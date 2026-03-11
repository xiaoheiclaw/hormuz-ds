"""Data ingestion from Readwise Reader API and market data via yfinance."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, UTC
from typing import Any

import httpx
import yfinance as yf

from hormuz.models import Observation

logger = logging.getLogger(__name__)

READWISE_API_URL = "https://readwise.io/api/v3/list/"


class ReadwiseIngester:
    """Fetch and filter articles from Readwise Reader API."""

    def __init__(self, config: dict):
        self.token: str = config["token"]
        self.tag: str = config["tag"]
        self.sources: set[str] = set(config["sources"])
        self.proxy: str | None = config.get("proxy")
        self.timeout: int = config.get("timeout", 30)

    def _matches(self, article: dict) -> bool:
        """Article matches if site_name in sources OR tag in article tags."""
        if article.get("site_name") in self.sources:
            return True
        tags = article.get("tags", {})
        if self.tag in tags:
            return True
        return False

    async def fetch(self, since: datetime | None = None) -> list[dict]:
        """Fetch articles with cursor pagination and source/tag filtering."""
        params: dict[str, Any] = {"location": "feed"}
        if since is not None:
            params["updatedAfter"] = since.isoformat()

        headers = {"Authorization": f"Token {self.token}"}
        proxy = self.proxy if self.proxy else None

        all_articles: list[dict] = []
        cursor: str | None = None

        async with httpx.AsyncClient(
            timeout=self.timeout,
            proxy=proxy,
            headers=headers,
        ) as client:
            while True:
                req_params = {**params}
                if cursor:
                    req_params["pageCursor"] = cursor

                resp = await client.get(READWISE_API_URL, params=req_params)

                # Rate limiting: respect 429 Retry-After
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", "5"))
                    logger.warning("Rate limited, retrying after %ds", retry_after)
                    await asyncio.sleep(retry_after)
                    continue

                resp.raise_for_status()
                data = resp.json()

                for article in data.get("results", []):
                    if self._matches(article):
                        all_articles.append({
                            "title": article.get("title", ""),
                            "summary": article.get("summary", ""),
                            "content": article.get("content", ""),
                            "source_url": article.get("source_url", ""),
                            "site_name": article.get("site_name", ""),
                            "tags": article.get("tags", {}),
                        })

                cursor = data.get("nextPageCursor")
                if not cursor:
                    break

        return all_articles


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

_TICKERS = [
    ("BZ=F", "brent_close"),
    ("^OVX", "ovx_close"),
]


class MarketIngester:
    """Fetch current market data via yfinance."""

    def __init__(self, config: dict | None = None):
        self.proxy: str | None = None
        if config:
            self.proxy = config.get("proxy")

    def fetch(self) -> list[Observation]:
        """Fetch latest close for Brent and OVX."""
        now = datetime.now(UTC)
        observations: list[Observation] = []

        for symbol, key in _TICKERS:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                if hist.empty:
                    logger.warning("No data for %s, skipping", symbol)
                    continue
                close = float(hist["Close"].iloc[-1])
                observations.append(Observation(
                    timestamp=now,
                    source="yfinance",
                    category="market",
                    key=key,
                    value=close,
                    metadata={"symbol": symbol},
                ))
            except Exception:
                logger.exception("Failed to fetch %s", symbol)

        return observations
