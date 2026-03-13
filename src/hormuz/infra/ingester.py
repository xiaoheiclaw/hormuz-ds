"""Data ingestion — Readwise articles + yfinance market data.

Converts raw external data into structured Observations.
"""

from __future__ import annotations

from datetime import datetime

import httpx
import yfinance as yf

from hormuz.core.types import Observation


# ── Readwise ──────────────────────────────────────────────────────────

async def fetch_readwise_articles(
    token: str,
    tag: str = "hormuz",
    proxy: str | None = None,
    timeout: int = 30,
) -> list[dict]:
    """Fetch articles from Readwise Reader API filtered by tag."""
    headers = {"Authorization": f"Token {token}"}
    params = {"tag": tag}
    async with httpx.AsyncClient(proxy=proxy, timeout=timeout) as client:
        resp = await client.get(
            "https://readwise.io/api/v3/list/",
            headers=headers,
            params=params,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])


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
    """Fetch Brent and OVX from yfinance."""
    result = {}

    # Brent crude
    brent = yf.Ticker("BZ=F")
    brent_price = brent.info.get("regularMarketPrice", 0.0)
    result["brent"] = brent_price

    # OVX (CBOE crude oil volatility)
    try:
        ovx = yf.Ticker("^OVX")
        result["ovx"] = ovx.info.get("regularMarketPrice", 0.0)
    except Exception:
        result["ovx"] = 0.0

    return result


def market_data_to_observations(
    market: dict,
    timestamp: datetime,
) -> list[Observation]:
    """Map market data fields to O07/O09/O10 observations."""
    obs = []

    if "brent" in market:
        obs.append(Observation(
            id="O07",
            timestamp=timestamp,
            value=market["brent"],
            source="yfinance",
            noise_note="Brent crude spot price",
        ))

    if "ovx" in market:
        obs.append(Observation(
            id="O09",
            timestamp=timestamp,
            value=market["ovx"],
            source="yfinance",
            noise_note="CBOE crude oil volatility index",
        ))

    if "brent_term_structure" in market:
        ts = market["brent_term_structure"]
        if len(ts) >= 2:
            # Backwardation = front > back
            spread = ts[0] - ts[-1]
            obs.append(Observation(
                id="O10",
                timestamp=timestamp,
                value=spread,
                source="yfinance",
                noise_note="Brent term structure spread (front - back)",
            ))

    return obs
