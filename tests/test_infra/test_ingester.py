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


def test_market_data_to_observations():
    from hormuz.infra.ingester import market_data_to_observations
    market = {"brent": 95.5, "ovx": 42.0, "brent_term_structure": [95.5, 93.0, 90.0]}
    obs_list = market_data_to_observations(market, timestamp=datetime(2026, 3, 12))
    ids = [o.id for o in obs_list]
    assert any("O07" in i or "O09" in i or "O10" in i for i in ids)


@pytest.mark.asyncio
async def test_fetch_market_data():
    from hormuz.infra.ingester import fetch_market_data
    with patch("hormuz.infra.ingester.yf") as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": 95.5}
        mock_yf.Ticker.return_value = mock_ticker
        result = await fetch_market_data(proxy=None)
        assert "brent" in result
