import pytest
from unittest.mock import AsyncMock
from datetime import datetime


@pytest.mark.asyncio
async def test_extract_observations():
    from hormuz.infra.analyzer import extract_observations
    mock_llm = AsyncMock()
    mock_llm.extract.return_value = {
        "observations": [
            {"id": "O01", "value": 3.0, "confidence": "high", "direction": "H1"},
            {"id": "O04", "value": 0.8, "confidence": "high", "direction": "H1"},
        ]
    }
    articles = [{"title": "CENTCOM update", "summary": "Attack frequency declining", "source": "CENTCOM"}]
    obs = await extract_observations(articles, llm=mock_llm)
    assert len(obs) == 2
    assert obs[0].id == "O01"


def test_llm_factory():
    from hormuz.infra.llm import create_llm_backend
    backend = create_llm_backend(backend_type="claude_api", model="claude-sonnet-4-6", api_key="test")
    assert hasattr(backend, "extract")
