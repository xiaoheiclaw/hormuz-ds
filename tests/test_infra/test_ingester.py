import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime


def test_parse_readwise_articles():
    from hormuz.infra.ingester import parse_readwise_articles
    raw = [
        {"id": 1, "title": "CENTCOM report", "summary": "3 attacks today", "published_date": "2026-03-12",
         "source": "CENTCOM", "url": "https://example.com/1"},
    ]
    articles = parse_readwise_articles(raw)
    assert len(articles) == 1
    assert articles[0]["source"] == "CENTCOM"


def test_market_data_returns_empty():
    """market_data_to_observations returns empty — Brent/OVX are calibration vars, not O-series."""
    from hormuz.infra.ingester import market_data_to_observations
    market = {"brent": 95.5, "ovx": 42.0}
    obs_list = market_data_to_observations(market, timestamp=datetime(2026, 3, 12))
    assert obs_list == []


def test_get_calibration_data():
    from hormuz.infra.ingester import get_calibration_data
    market = {"brent": 95.5, "ovx": 42.0, "brent_front": 95.5}
    calib = get_calibration_data(market)
    assert calib["brent_price"] == 95.5
    assert calib["ovx"] == 42.0


@pytest.mark.asyncio
async def test_fetch_market_data():
    from hormuz.infra.ingester import fetch_market_data
    with patch("hormuz.infra.ingester.yf") as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": 95.5}
        mock_ticker.history.return_value = MagicMock(empty=True)
        mock_yf.Ticker.return_value = mock_ticker
        result = await fetch_market_data(proxy=None)
        assert "brent" in result
