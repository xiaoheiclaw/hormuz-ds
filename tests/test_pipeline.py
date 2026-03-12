"""Tests for the pipeline orchestrator."""
import pytest
from pathlib import Path
from datetime import datetime, UTC
from unittest.mock import AsyncMock, patch, MagicMock
from hormuz.pipeline import Pipeline
from hormuz.models import Observation


@pytest.fixture
def pipeline_dirs(tmp_path):
    """Create directory structure for pipeline."""
    config_dir = Path(__file__).parent.parent / "configs"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    template_dir = Path(__file__).parent.parent / "templates"
    return config_dir, data_dir, reports_dir, template_dir


def _make_pipeline(config_dir, data_dir, reports_dir, template_dir):
    """Create pipeline with mocked external dependencies."""
    with patch("hormuz.pipeline.ReadwiseIngester") as MockReadwise, \
         patch("hormuz.pipeline.MarketIngester") as MockMarket, \
         patch("hormuz.pipeline.get_backend") as MockBackend:

        mock_rw = AsyncMock()
        mock_rw.fetch.return_value = []
        MockReadwise.return_value = mock_rw

        mock_market = MagicMock()
        mock_market.fetch.return_value = []
        MockMarket.return_value = mock_market

        mock_backend = AsyncMock()
        mock_backend.complete.return_value = '{"observations": []}'
        MockBackend.return_value = mock_backend

        pipeline = Pipeline(config_dir, data_dir, reports_dir, template_dir)
    return pipeline


class TestPipelineInit:
    def test_initializes_all_components(self, pipeline_dirs):
        config_dir, data_dir, reports_dir, template_dir = pipeline_dirs
        pipeline = _make_pipeline(config_dir, data_dir, reports_dir, template_dir)
        assert pipeline.db is not None
        assert pipeline.signals is not None
        assert pipeline.ach is not None
        assert pipeline.physical is not None
        assert pipeline.schelling is not None
        assert pipeline.reporter is not None
        assert pipeline.analyzer is not None

    def test_loads_config(self, pipeline_dirs):
        config_dir, data_dir, reports_dir, template_dir = pipeline_dirs
        pipeline = _make_pipeline(config_dir, data_dir, reports_dir, template_dir)
        assert "conflict" in pipeline.config
        assert "paths" in pipeline.parameters


class TestPipelineExecution:
    @pytest.mark.asyncio
    async def test_run_completes_without_crash(self, pipeline_dirs):
        """Full pipeline run with mocked external APIs."""
        config_dir, data_dir, reports_dir, template_dir = pipeline_dirs

        with patch("hormuz.pipeline.ReadwiseIngester") as MockReadwise, \
             patch("hormuz.pipeline.MarketIngester") as MockMarket, \
             patch("hormuz.pipeline.get_backend") as MockBackend:

            mock_rw = AsyncMock()
            mock_rw.fetch.return_value = []
            MockReadwise.return_value = mock_rw

            mock_market = MagicMock()
            mock_market.fetch.return_value = [
                Observation(timestamp=datetime.now(UTC), source="yfinance",
                           category="market", key="brent_close", value=95.0)
            ]
            MockMarket.return_value = mock_market

            mock_backend = AsyncMock()
            mock_backend.complete.return_value = '{"observations": []}'
            MockBackend.return_value = mock_backend

            pipeline = Pipeline(config_dir, data_dir, reports_dir, template_dir)
            summary = await pipeline.run()

        assert isinstance(summary, dict)
        assert "steps_completed" in summary

    @pytest.mark.asyncio
    async def test_step_failure_doesnt_crash_pipeline(self, pipeline_dirs):
        """If ingestion fails, pipeline continues to other steps."""
        config_dir, data_dir, reports_dir, template_dir = pipeline_dirs

        with patch("hormuz.pipeline.ReadwiseIngester") as MockReadwise, \
             patch("hormuz.pipeline.MarketIngester") as MockMarket, \
             patch("hormuz.pipeline.get_backend") as MockBackend:

            mock_rw = AsyncMock()
            mock_rw.fetch.side_effect = Exception("Network error")
            MockReadwise.return_value = mock_rw

            mock_market = MagicMock()
            mock_market.fetch.side_effect = Exception("yfinance error")
            MockMarket.return_value = mock_market

            mock_backend = AsyncMock()
            MockBackend.return_value = mock_backend

            pipeline = Pipeline(config_dir, data_dir, reports_dir, template_dir)
            summary = await pipeline.run()

        assert summary is not None
        assert "errors" in summary
        assert len(summary["errors"]) > 0

    @pytest.mark.asyncio
    async def test_signals_before_engine(self, pipeline_dirs):
        """Verify signals (step 4) runs before engine (step 5)."""
        config_dir, data_dir, reports_dir, template_dir = pipeline_dirs
        execution_order = []

        with patch("hormuz.pipeline.ReadwiseIngester") as MockReadwise, \
             patch("hormuz.pipeline.MarketIngester") as MockMarket, \
             patch("hormuz.pipeline.get_backend") as MockBackend:

            mock_rw = AsyncMock()
            mock_rw.fetch.return_value = []
            MockReadwise.return_value = mock_rw
            mock_market = MagicMock()
            mock_market.fetch.return_value = []
            MockMarket.return_value = mock_market
            mock_backend = AsyncMock()
            MockBackend.return_value = mock_backend

            pipeline = Pipeline(config_dir, data_dir, reports_dir, template_dir)

            orig_scan = pipeline.signals.scan
            orig_engine = pipeline._step_engine

            def mock_scan(*args, **kwargs):
                execution_order.append("signals")
                return orig_scan(*args, **kwargs)

            def mock_engine(*args, **kwargs):
                execution_order.append("engine")
                return orig_engine(*args, **kwargs)

            pipeline.signals.scan = mock_scan
            pipeline._step_engine = mock_engine

            await pipeline.run()

        assert "signals" in execution_order
        assert "engine" in execution_order
        assert execution_order.index("signals") < execution_order.index("engine")

    @pytest.mark.asyncio
    async def test_market_observations_written_to_db(self, pipeline_dirs):
        """Market observations from ingester are persisted to DB."""
        config_dir, data_dir, reports_dir, template_dir = pipeline_dirs

        with patch("hormuz.pipeline.ReadwiseIngester") as MockReadwise, \
             patch("hormuz.pipeline.MarketIngester") as MockMarket, \
             patch("hormuz.pipeline.get_backend") as MockBackend:

            mock_rw = AsyncMock()
            mock_rw.fetch.return_value = []
            MockReadwise.return_value = mock_rw

            now = datetime.now(UTC)
            mock_market = MagicMock()
            mock_market.fetch.return_value = [
                Observation(timestamp=now, source="yfinance",
                           category="market", key="brent_close", value=92.5),
                Observation(timestamp=now, source="yfinance",
                           category="market", key="ovx_close", value=35.0),
            ]
            MockMarket.return_value = mock_market

            mock_backend = AsyncMock()
            MockBackend.return_value = mock_backend

            pipeline = Pipeline(config_dir, data_dir, reports_dir, template_dir)
            await pipeline.run()

        # Check observations were written
        from datetime import timedelta
        obs = pipeline.db.get_observations_since(now - timedelta(hours=1), category="market")
        assert len(obs) == 2
        keys = {o.key for o in obs}
        assert "brent_close" in keys
        assert "ovx_close" in keys

    @pytest.mark.asyncio
    async def test_summary_tracks_steps(self, pipeline_dirs):
        """Summary dict tracks which steps completed and which errored."""
        config_dir, data_dir, reports_dir, template_dir = pipeline_dirs
        pipeline = _make_pipeline(config_dir, data_dir, reports_dir, template_dir)
        summary = await pipeline.run()

        assert "steps_completed" in summary
        assert isinstance(summary["steps_completed"], list)
        # At minimum, revert_check, signals, engine, report should complete
        assert len(summary["steps_completed"]) >= 4
