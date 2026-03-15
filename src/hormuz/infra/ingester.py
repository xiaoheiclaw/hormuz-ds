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
    updated_after: str | None = None,
) -> list[dict]:
    """Fetch articles from Readwise Reader API, optionally filtered by source site_name.

    updated_after: ISO datetime string (e.g. "2026-03-08T00:00:00") to fetch
    only articles updated after this time. Essential for backfill to reach
    older articles beyond the default feed window.
    """
    filter_sources = sources  # None = no filtering, fetch all
    headers = {"Authorization": f"Token {token}"}
    all_docs: list[dict] = []
    cursor: str | None = None

    import asyncio as _aio

    async with httpx.AsyncClient(proxy=proxy, timeout=timeout) as client:
        while len(all_docs) < limit:
            params: dict = {
                "page_size": min(limit - len(all_docs), 100),
                "location": "feed",
                "withHtmlContent": "true",
            }
            if updated_after:
                params["updatedAfter"] = updated_after
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
                if filter_sources is None or doc.get("site_name") in filter_sources:
                    all_docs.append(doc)

            cursor = data.get("nextPageCursor")
            if not cursor:
                break

    return all_docs


def _html_to_text(html: str | None) -> str:
    """Strip HTML tags to plain text. Lightweight, no external deps."""
    if not html:
        return ""
    import re
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.S)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(p|div|h[1-6]|li|tr)>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&#\d+;", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:3000]  # Cap at 3000 chars per article for LLM context


def parse_readwise_articles(raw: list[dict], relevance_filter: bool = True) -> list[dict]:
    """Normalize article fields from Readwise response.

    When relevance_filter=True, pre-filter articles by keyword relevance
    to avoid feeding irrelevant content (sports, entertainment) to LLM.
    """
    articles = []
    for item in raw:
        # Prefer html_content (full text) > content > summary
        body = _html_to_text(item.get("html_content")) or item.get("content") or item.get("summary") or ""
        articles.append({
            "id": item.get("id"),
            "title": item.get("title", ""),
            "summary": body,
            "source": item.get("site_name") or item.get("source", "unknown"),
            "url": item.get("url", ""),
            "published_date": item.get("published_date"),
        })
    if relevance_filter:
        articles = _filter_relevant(articles)
    return articles


# Two-tier relevance: STRONG keywords pass alone, WEAK require 2+ matches.
# This avoids "oil-rich country" or "peace prize" false positives.
_STRONG_KEYWORDS = {
    # Direct Hormuz / Gulf crisis
    "hormuz", "strait of hormuz", "persian gulf", "gulf of oman",
    "irgc", "islamic revolutionary guard",
    "bandar abbas", "fujairah", "ras tanura", "yanbu", "kharg island",
    # Maritime crisis
    "tanker attack", "shipping attack", "mine clearance", "minesweep",
    "war risk premium", "force majeure", "blockade",
    "vlcc", "freight surge", "convoy escort",
    # Specific military
    "centcom", "anti-ship missile", "naval mine",
    "gps spoofing", "ais spoofing", "electronic warfare",
    # Strategic energy
    "spr release", "strategic petroleum reserve", "oil embargo",
    "brent surge", "oil supply disruption",
}

_WEAK_KEYWORDS = {
    # Geography (too broad alone)
    "iran", "tehran", "saudi", "oman", "uae", "qatar", "bahrain", "kuwait",
    # General military (too broad alone)
    "attack", "strike", "missile", "drone", "naval", "escalat",
    "weapon", "ceasefire", "military",
    # General energy/maritime (too broad alone)
    "oil", "crude", "tanker", "shipping", "pipeline", "port",
    "brent", "opec", "petroleum", "freight",
    # Diplomacy (too broad alone)
    "mediation", "negotiat", "diplomat", "sanction",
}


def _filter_relevant(articles: list[dict]) -> list[dict]:
    """Keep articles matching STRONG keyword or 2+ WEAK keywords."""
    relevant = []
    for a in articles:
        text = (a.get("title", "") + " " + a.get("summary", "")).lower()
        # Any strong keyword → pass
        if any(kw in text for kw in _STRONG_KEYWORDS):
            relevant.append(a)
            continue
        # 2+ weak keywords → pass
        weak_hits = sum(1 for kw in _WEAK_KEYWORDS if kw in text)
        if weak_hits >= 2:
            relevant.append(a)
    return relevant


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
