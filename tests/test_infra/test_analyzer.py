import pytest
from unittest.mock import AsyncMock
from datetime import datetime


@pytest.mark.asyncio
async def test_extract_observations_with_signal_objects():
    """LLM returns signal objects with key + evidence."""
    from hormuz.infra.analyzer import extract_observations
    mock_llm = AsyncMock()
    mock_llm.extract.return_value = {
        "observations": [
            {"id": "O01", "value": 3.0, "confidence": "high", "direction": "H1"},
        ],
        "signals": [{"key": "external_mediation", "evidence": "high"}],
    }
    articles = [{"title": "CENTCOM update", "summary": "Attack frequency declining", "source": "CENTCOM"}]
    result = await extract_observations(articles, llm=mock_llm)
    assert len(result.observations) == 1
    assert len(result.signals) == 1
    assert result.signals[0].key == "external_mediation"
    assert result.signals[0].evidence == 1.0  # high → 1.0


@pytest.mark.asyncio
async def test_extract_observations_plain_string_compat():
    """Plain string signals get default medium evidence (backwards compat)."""
    from hormuz.infra.analyzer import extract_observations
    mock_llm = AsyncMock()
    mock_llm.extract.return_value = {
        "observations": [],
        "signals": ["irgc_escalation"],
    }
    articles = [{"title": "test", "summary": "test", "source": "test"}]
    result = await extract_observations(articles, llm=mock_llm)
    assert len(result.signals) == 1
    assert result.signals[0].key == "irgc_escalation"
    assert result.signals[0].evidence == 0.5  # default medium


@pytest.mark.asyncio
async def test_extract_dedup_signals_keeps_highest_evidence():
    """When same signal appears in multiple batches, keep highest evidence."""
    from hormuz.infra.analyzer import extract_observations
    mock_llm = AsyncMock()
    # Two batches return same signal with different evidence
    mock_llm.extract.side_effect = [
        {"observations": [], "signals": [{"key": "external_mediation", "evidence": "low"}]},
        {"observations": [], "signals": [{"key": "external_mediation", "evidence": "high"}]},
    ]
    articles = [{"title": f"art{i}", "summary": "x", "source": "s"} for i in range(2)]
    result = await extract_observations(articles, llm=mock_llm, batch_size=1)
    assert len(result.signals) == 1
    assert result.signals[0].evidence == 1.0  # high wins


def test_llm_factory():
    from hormuz.infra.llm import create_llm_backend
    backend = create_llm_backend(backend_type="claude_api", model="claude-sonnet-4-6", api_key="test")
    assert hasattr(backend, "extract")
