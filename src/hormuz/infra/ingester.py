"""Data ingestion — Readwise articles + yfinance market data.

Converts raw external data into structured Observations.
"""

from __future__ import annotations

from datetime import datetime

import httpx
import yfinance as yf

from hormuz.core.types import Observation


# ── Readwise ──────────────────────────────────────────────────────────

HORMUZ_SOURCES = {
    "gCaptain",
    "Splash247",
    "Al-Monitor",
    "Reuters",
    "OilPrice.com",
    "Iran International",
}


async def fetch_readwise_articles(
    token: str,
    sources: set[str] | None = None,
    proxy: str | None = None,
    timeout: int = 30,
    limit: int = 50,
) -> list[dict]:
    """Fetch articles from Readwise Reader API, filtered by source site_name."""
    filter_sources = sources or HORMUZ_SOURCES
    headers = {"Authorization": f"Token {token}"}
    all_docs: list[dict] = []
    cursor: str | None = None

    import asyncio as _aio

    async with httpx.AsyncClient(proxy=proxy, timeout=timeout) as client:
        while len(all_docs) < limit:
            params: dict = {
                "page_size": min(limit - len(all_docs), 100),
                "location": "feed",
            }
            if cursor:
                params["pageCursor"] = cursor

            # Retry on connection errors (proxy/relay disconnects)
            for attempt in range(3):
                try:
                    resp = await client.get(
                        "https://readwise.io/api/v3/list/",
                        headers=headers,
                        params=params,
                    )
                    break
                except (httpx.RemoteProtocolError, httpx.ConnectError):
                    if attempt == 2:
                        raise
                    await _aio.sleep(2 ** attempt)

            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "60"))
                await _aio.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break

            for doc in results:
                if doc.get("site_name") in filter_sources:
                    all_docs.append(doc)

            cursor = data.get("nextPageCursor")
            if not cursor:
                break

    return all_docs


def parse_readwise_articles(raw: list[dict]) -> list[dict]:
    """Normalize article fields from Readwise response."""
    articles = []
    for item in raw:
        articles.append({
            "id": item.get("id"),
            "title": item.get("title", ""),
            "summary": item.get("summary", ""),
            "source": item.get("source", "unknown"),
            "url": item.get("url", ""),
            "published_date": item.get("published_date"),
        })
    return articles


# ── Market data ───────────────────────────────────────────────────────

async def fetch_market_data(proxy: str | None = None) -> dict:
    """Fetch Brent, OVX, and term structure from yfinance.

    These are calibration/result variables (PRD §3), not O-series.
    """
    result = {}

    # Brent crude front month
    brent = yf.Ticker("BZ=F")
    brent_price = brent.info.get("regularMarketPrice", 0.0)
    result["brent"] = brent_price

    # OVX (CBOE crude oil volatility)
    try:
        ovx = yf.Ticker("^OVX")
        result["ovx"] = ovx.info.get("regularMarketPrice", 0.0)
    except Exception:
        result["ovx"] = 0.0

    # BWET (Breakwave Tanker ETF) — TD3 VLCC freight proxy for O09
    try:
        bwet = yf.Ticker("BWET")
        bwet_price = bwet.info.get("regularMarketPrice", 0.0)
        if bwet_price and bwet_price > 0:
            result["bwet_price"] = bwet_price
    except Exception:
        pass

    return result


# ── O12: Fujairah-Singapore bunker spread ─────────────────────────────

import re

_BUNKER_URLS = {
    "fujairah": "https://shipandbunker.com/prices/emea/ae/ae-fjr-fujairah",
    "singapore": "https://shipandbunker.com/prices/apac/sg/sg-sin-singapore",
}


async def fetch_bunker_spread(
    proxy: str | None = None,
    timeout: int = 30,
) -> Observation | None:
    """Fetch VLSFO prices from Ship & Bunker for Fujairah and Singapore.

    Returns O12 observation with spread in $/mt.
    Normal spread <$20, crisis >$100 signals logistics breakdown.
    """
    prices = {}
    async with httpx.AsyncClient(proxy=proxy, timeout=timeout) as client:
        for port, url in _BUNKER_URLS.items():
            try:
                resp = await client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; HormuzDS/1.0)",
                })
                resp.raise_for_status()
                # Extract VLSFO price from page — pattern: "VLSFO" followed by price
                # Ship & Bunker format: price in table cells like "1,052.50"
                text = resp.text
                # Look for VLSFO price pattern in the HTML
                # The price appears after VLSFO mention in table structure
                m = re.search(
                    r'VLSFO.*?(?:Price|price|>\s*\$?\s*)([\d,]+\.?\d*)\s*(?:\$|/mt|<)',
                    text, re.DOTALL | re.IGNORECASE,
                )
                if m:
                    price_str = m.group(1).replace(",", "")
                    prices[port] = float(price_str)
            except Exception:
                continue

    if "fujairah" not in prices or "singapore" not in prices:
        return None

    spread = prices["singapore"] - prices["fujairah"]

    return Observation(
        id="O12",
        timestamp=datetime.now(),
        value=round(spread, 1),
        source="shipandbunker:vlsfo",
        noise_note=f"FUJ=${prices['fujairah']:.1f} SIN=${prices['singapore']:.1f}",
    )


# ── BWET → O09 mapping ───────────────────────────────────────────────

def bwet_to_vlcc_obs(market: dict, timestamp: datetime) -> Observation | None:
    """Map BWET ETF price to O09 (VLCC freight rate proxy).

    BWET tracks 90% TD3C FFA + 10% TD20. Price range:
    - Normal: ~$8-15 (WS40-80)
    - Elevated: ~$15-25 (WS100-250)
    - Crisis: ~$25-50+ (WS300-500+)

    We map to approximate WS points using linear interpolation.
    """
    bwet = market.get("bwet_price")
    if not bwet or bwet <= 0:
        return None

    # Approximate mapping: BWET $10→WS60, $20→WS200, $40→WS450
    # Linear pieces: [8,60] → [15,150] → [50,500]
    if bwet <= 15:
        ws = 60 + (bwet - 8) * (150 - 60) / (15 - 8)
    else:
        ws = 150 + (bwet - 15) * (500 - 150) / (50 - 15)
    ws = max(40, min(600, ws))

    return Observation(
        id="O09",
        timestamp=timestamp,
        value=round(ws, 0),
        source="yfinance:bwet_proxy",
        noise_note=f"BWET=${bwet:.2f} → WS{ws:.0f} (proxy)",
    )


# ── EIA SPR data ─────────────────────────────────────────────────────

async def fetch_spr_release(
    eia_api_key: str,
    proxy: str | None = None,
    timeout: int = 30,
) -> Observation | None:
    """Fetch latest SPR weekly stock change from EIA API v2, compute release rate (mbd).

    Returns O13 observation, or None if data unavailable.
    Requires free EIA API key: https://www.eia.gov/opendata/register.php
    """
    from datetime import datetime

    url = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
    params = {
        "api_key": eia_api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[product][]": "EPC0",  # crude oil
        "facets[process][]": "SAE",   # SPR ending stocks
        "facets[duoarea][]": "NUS",   # national US
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 2,  # need 2 weeks to compute delta
    }

    async with httpx.AsyncClient(proxy=proxy, timeout=timeout) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    rows = data.get("response", {}).get("data", [])
    if len(rows) < 2:
        return None

    # rows[0] = most recent, rows[1] = previous week
    current_stock = float(rows[0]["value"])  # thousand barrels
    previous_stock = float(rows[1]["value"])

    # Negative change = release (stock decreased)
    change_thousand = current_stock - previous_stock
    release_mbd = -change_thousand / 1000 / 7  # positive = release

    return Observation(
        id="O13",
        timestamp=datetime.now(),
        value=round(max(0.0, release_mbd), 2),  # clip negative (accumulation) to 0
        source="eia:weekly",
        noise_note=f"SPR stock change: {change_thousand:.0f}k bbl/week",
    )


def market_data_to_observations(
    market: dict,
    timestamp: datetime,
) -> list[Observation]:
    """Map market data to observations.

    Note: Brent price and OVX are calibration/result variables (PRD §3),
    NOT O-series observations. They go into StateVector, not ACH.
    O07-O13 (AP premium, P&I, VLCC rate, transit, pipeline, Fujairah, SPR)
    require specialized data sources (Lloyd's, Baltic Exchange, Vortexa, etc.)
    and are extracted from news via LLM as a fallback.
    """
    # No O-series observations from yfinance — Brent/OVX are calibration vars
    return []


def get_calibration_data(market: dict) -> dict:
    """Extract calibration variables from market data (not O-series).

    These are used for consistency checks (PRD §3) and StateVector updates,
    not fed into ACH.
    """
    return {
        "brent_price": market.get("brent", 0.0),
        "brent_front": market.get("brent_front", 0.0),
        "ovx": market.get("ovx", 0.0),
    }
