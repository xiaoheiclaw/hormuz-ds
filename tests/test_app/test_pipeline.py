import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
from datetime import datetime


@pytest.fixture
def db_path(tmp_path):
    from hormuz.infra.db import init_db
    p = tmp_path / "test.db"
    init_db(p)
    return p


@pytest.mark.asyncio
async def test_pipeline_full_run(db_path, tmp_path):
    from hormuz.app.pipeline import run_pipeline
    config = {
        "db": {"path": str(db_path)},
        "configs_dir": str(Path(__file__).parents[2] / "configs"),
        "readwise": {"token": "fake", "tag": "test", "proxy": None, "timeout": 10},
        "llm": {"backend": "claude_api", "claude_api": {"model": "test", "api_key": "fake"}},
        "conflict": {"start_date": "2026-03-01"},
        "output_dir": str(tmp_path),
    }
    from hormuz.infra.analyzer import ExtractionResult
    mock_extraction = ExtractionResult(observations=[], signals=[])
    with patch("hormuz.app.pipeline.fetch_readwise_articles", new_callable=AsyncMock, return_value=[]), \
         patch("hormuz.app.pipeline.fetch_market_data", new_callable=AsyncMock, return_value={"brent": 95.0}), \
         patch("hormuz.app.pipeline.extract_observations", new_callable=AsyncMock, return_value=mock_extraction):
        result = await run_pipeline(config)
    assert result["steps_completed"] >= 4
    assert "system_output" in result


def test_engine_run_pure():
    """Engine run with canned data, no IO"""
    from hormuz.app.pipeline import engine_run
    from hormuz.core.types import Parameters, ACHPosterior
    from hormuz.core.variables import load_constants
    constants = load_constants(Path(__file__).parents[2] / "configs" / "constants.yaml")
    params = Parameters()
    observations = []
    controls = []
    so, _mc = engine_run(constants, params, observations, controls, events={}, mc_n=100, seed=42)
    assert so.gross_gap_mbd == pytest.approx(16.0, abs=0.2)
    assert so.path_probabilities.a + so.path_probabilities.b + so.path_probabilities.c == pytest.approx(1.0)
