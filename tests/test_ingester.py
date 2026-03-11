"""Tests for ReadwiseIngester and MarketIngester."""

import pytest
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

from hormuz.ingester import ReadwiseIngester, MarketIngester
from hormuz.models import Observation


# ---------------------------------------------------------------------------
# ReadwiseIngester
# ---------------------------------------------------------------------------

class TestReadwiseIngester:
    @pytest.fixture
    def config(self):
        return {
            "token": "test_token",
            "tag": "hormuz",
            "sources": ["CENTCOM", "Reuters", "Lloyd's List"],
            "proxy": None,
            "timeout": 30,
        }

    def test_init(self, config):
        ingester = ReadwiseIngester(config)
        assert ingester.tag == "hormuz"
        assert "CENTCOM" in ingester.sources
        assert ingester.timeout == 30

    def _mock_response(self, results, next_cursor=None, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = {
            "results": results,
            "nextPageCursor": next_cursor,
        }
        resp.raise_for_status = MagicMock()
        return resp

    def _patch_client(self, responses):
        """Return a context-manager patch that yields responses in order."""
        client = AsyncMock()
        client.get = AsyncMock(side_effect=responses)
        mock_cls = patch("hormuz.ingester.httpx.AsyncClient")
        return mock_cls, client

    @pytest.mark.asyncio
    async def test_fetch_filters_by_source(self, config):
        """Articles from whitelisted sources are included."""
        results = [
            {"title": "CENTCOM Update", "summary": "s", "content": "c",
             "source_url": "https://centcom.mil/1", "site_name": "CENTCOM",
             "tags": {}},
            {"title": "Random Blog", "summary": "s", "content": "c",
             "source_url": "https://blog.com/1", "site_name": "Random Blog",
             "tags": {}},
        ]
        resp = self._mock_response(results)

        with patch("hormuz.ingester.httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get = AsyncMock(return_value=resp)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            ingester = ReadwiseIngester(config)
            articles = await ingester.fetch()

        assert len(articles) == 1
        assert articles[0]["site_name"] == "CENTCOM"

    @pytest.mark.asyncio
    async def test_fetch_filters_by_tag(self, config):
        """Articles with matching tag included even if source not whitelisted."""
        results = [
            {"title": "Tagged Article", "summary": "s", "content": "c",
             "source_url": "https://example.com/1", "site_name": "Unknown Source",
             "tags": {"hormuz": {}}},
        ]
        resp = self._mock_response(results)

        with patch("hormuz.ingester.httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get = AsyncMock(return_value=resp)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            ingester = ReadwiseIngester(config)
            articles = await ingester.fetch()

        assert len(articles) == 1
        assert articles[0]["title"] == "Tagged Article"

    @pytest.mark.asyncio
    async def test_fetch_pagination(self, config):
        """Cursor-based pagination fetches all pages."""
        page1 = self._mock_response(
            [{"title": "A1", "summary": "", "content": "", "source_url": "",
              "site_name": "CENTCOM", "tags": {}}],
            next_cursor="cursor2",
        )
        page2 = self._mock_response(
            [{"title": "A2", "summary": "", "content": "", "source_url": "",
              "site_name": "Reuters", "tags": {}}],
        )

        with patch("hormuz.ingester.httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get = AsyncMock(side_effect=[page1, page2])
            MockClient.return_value.__aenter__ = AsyncMock(return_value=client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            ingester = ReadwiseIngester(config)
            articles = await ingester.fetch()

        assert len(articles) == 2
        assert client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_with_since(self, config):
        """The since parameter is forwarded as updatedAfter."""
        resp = self._mock_response([])

        with patch("hormuz.ingester.httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get = AsyncMock(return_value=resp)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            ingester = ReadwiseIngester(config)
            since = datetime(2026, 3, 1, tzinfo=UTC)
            await ingester.fetch(since=since)

        call_kwargs = client.get.call_args
        params = call_kwargs.kwargs.get("params", call_kwargs[1].get("params", {}))
        assert "updatedAfter" in params

    @pytest.mark.asyncio
    async def test_fetch_returns_expected_keys(self, config):
        """Returned dicts contain required keys."""
        results = [
            {"title": "T", "summary": "S", "content": "C",
             "source_url": "http://x", "site_name": "CENTCOM",
             "tags": {"hormuz": {}}},
        ]
        resp = self._mock_response(results)

        with patch("hormuz.ingester.httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.get = AsyncMock(return_value=resp)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            ingester = ReadwiseIngester(config)
            articles = await ingester.fetch()

        required = {"title", "summary", "content", "source_url", "site_name", "tags"}
        assert required.issubset(articles[0].keys())


# ---------------------------------------------------------------------------
# MarketIngester
# ---------------------------------------------------------------------------

class TestMarketIngester:
    def _make_hist(self, close_value: float) -> pd.DataFrame:
        return pd.DataFrame(
            {"Close": [close_value]},
            index=pd.DatetimeIndex([datetime(2026, 3, 10)]),
        )

    def test_fetch_returns_observations(self):
        """Returns Observation objects for Brent and OVX."""
        hist_brent = self._make_hist(95.5)
        hist_ovx = self._make_hist(32.1)

        ticker_map = {
            "BZ=F": MagicMock(history=MagicMock(return_value=hist_brent)),
            "^OVX": MagicMock(history=MagicMock(return_value=hist_ovx)),
        }

        with patch("hormuz.ingester.yf") as mock_yf:
            mock_yf.Ticker.side_effect = lambda sym: ticker_map[sym]
            ingester = MarketIngester()
            obs = ingester.fetch()

        assert isinstance(obs, list)
        assert all(isinstance(o, Observation) for o in obs)
        assert len(obs) == 2

    def test_fetch_brent_value(self):
        """Brent observation has correct value."""
        hist_brent = self._make_hist(95.5)
        hist_ovx = self._make_hist(32.1)

        ticker_map = {
            "BZ=F": MagicMock(history=MagicMock(return_value=hist_brent)),
            "^OVX": MagicMock(history=MagicMock(return_value=hist_ovx)),
        }

        with patch("hormuz.ingester.yf") as mock_yf:
            mock_yf.Ticker.side_effect = lambda sym: ticker_map[sym]
            obs = MarketIngester().fetch()

        brent = [o for o in obs if o.key == "brent_close"]
        assert len(brent) == 1
        assert brent[0].value == 95.5
        assert brent[0].source == "yfinance"
        assert brent[0].category == "market"

    def test_fetch_ovx_value(self):
        """OVX observation has correct value."""
        hist_brent = self._make_hist(95.5)
        hist_ovx = self._make_hist(32.1)

        ticker_map = {
            "BZ=F": MagicMock(history=MagicMock(return_value=hist_brent)),
            "^OVX": MagicMock(history=MagicMock(return_value=hist_ovx)),
        }

        with patch("hormuz.ingester.yf") as mock_yf:
            mock_yf.Ticker.side_effect = lambda sym: ticker_map[sym]
            obs = MarketIngester().fetch()

        ovx = [o for o in obs if o.key == "ovx_close"]
        assert len(ovx) == 1
        assert ovx[0].value == 32.1

    def test_fetch_skips_empty_history(self):
        """If a ticker returns empty history, skip it gracefully."""
        hist_brent = self._make_hist(95.5)
        empty_hist = pd.DataFrame()

        ticker_map = {
            "BZ=F": MagicMock(history=MagicMock(return_value=hist_brent)),
            "^OVX": MagicMock(history=MagicMock(return_value=empty_hist)),
        }

        with patch("hormuz.ingester.yf") as mock_yf:
            mock_yf.Ticker.side_effect = lambda sym: ticker_map[sym]
            obs = MarketIngester().fetch()

        assert len(obs) == 1
        assert obs[0].key == "brent_close"
