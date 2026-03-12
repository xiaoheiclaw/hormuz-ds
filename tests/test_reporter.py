"""Tests for the HTML report generator."""
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hormuz.db import HormuzDB
from hormuz.models import (
    MCParams,
    MCResult,
    Observation,
    PathWeights,
    PositionSignal,
    Regime,
    RegimeType,
    Signal,
    SignalStatus,
)
from hormuz.reporter import Reporter


@pytest.fixture
def reporter(tmp_db, tmp_path) -> Reporter:
    db = HormuzDB(tmp_db)
    template_dir = Path(__file__).parent.parent / "templates"
    config_dir = Path(__file__).parent.parent / "configs"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    return Reporter(db, template_dir, output_dir, reports_dir, config_dir=config_dir)


def _seed_basic(db: HormuzDB) -> None:
    """Insert minimal data for a meaningful report."""
    now = datetime.now(UTC)
    db.insert_regime(Regime(
        timestamp=now, question="q1",
        regime=RegimeType.wide, trigger="initial"))
    db.insert_regime(Regime(
        timestamp=now, question="q2",
        regime=RegimeType.wide, trigger="initial"))
    db.insert_mc_params(MCParams(
        timestamp=now,
        irgc_decay_mean=6.0, convoy_start_mean=5.0,
        pipeline_max=4.0,
        pipeline_ramp_weeks=2.5, spr_rate_mean=2.5,
        spr_delay_weeks=2.5, surplus_buffer=2.5,
        path_weights=PathWeights(a=0.3, b=0.5, c=0.2),
        trigger="initial"))


class TestStatusReport:
    def test_generates_html_file(self, reporter):
        path = reporter.update_status()
        assert path.exists()
        assert path.suffix == ".html"

    def test_html_contains_panels(self, reporter):
        _seed_basic(reporter.db)
        path = reporter.update_status()
        content = path.read_text()
        assert "ABC 路径权重" in content
        assert "ACH 制度" in content
        assert "物理参数" in content
        assert "Grabo 绊线" in content
        assert "近期观测" in content

    def test_shows_path_weights(self, reporter):
        _seed_basic(reporter.db)
        path = reporter.update_status()
        content = path.read_text()
        assert "30" in content  # path A weight %
        assert "50" in content  # path B weight %
        assert "20" in content  # path C weight %

    def test_shows_regime_badges(self, reporter):
        _seed_basic(reporter.db)
        path = reporter.update_status()
        content = path.read_text()
        assert "wide" in content.lower()

    def test_shows_phys_params(self, reporter):
        _seed_basic(reporter.db)
        path = reporter.update_status()
        content = path.read_text()
        assert "IRGC 衰减均值" in content
        assert "护航启动均值" in content

    def test_shows_signal_status(self, reporter):
        reporter.db.insert_signal(Signal(
            timestamp=datetime.now(UTC), signal_id="E3",
            status=SignalStatus.triggered,
            action_taken="convoyStartMean上调1周"))
        path = reporter.update_status()
        content = path.read_text()
        assert "E3" in content

    def test_shows_path_descriptions(self, reporter):
        _seed_basic(reporter.db)
        path = reporter.update_status()
        content = path.read_text()
        assert "快速解决" in content
        assert "拉锯消耗" in content
        assert "升级扩大" in content

    def test_shows_observations(self, reporter):
        reporter.db.insert_observation(Observation(
            timestamp=datetime.now(UTC), source="yfinance",
            category="market", key="brent_close", value=88.5))
        path = reporter.update_status()
        content = path.read_text()
        assert "88.5" in content

    def test_empty_db_renders(self, reporter):
        """Template handles all-None data gracefully."""
        path = reporter.update_status()
        content = path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "ABC 路径权重" in content


class TestWeeklyArchive:
    def test_archive_creates_file(self, reporter):
        _seed_basic(reporter.db)
        path = reporter.archive_weekly(force=True)
        assert path is not None
        assert path.exists()
        assert "W" in path.stem

    def test_archive_filename_format(self, reporter):
        _seed_basic(reporter.db)
        path = reporter.archive_weekly(force=True)
        # e.g. 2026-W11.html
        assert path.stem.startswith("20")
        assert path.suffix == ".html"

    def test_archive_returns_none_if_not_wednesday(self, reporter):
        """Without force=True, returns None on non-Wednesday."""
        result = reporter.archive_weekly(force=False)
        # Could be None or a path depending on what day the test runs
        # We just verify it doesn't crash
        if datetime.now(UTC).weekday() != 2:
            assert result is None

    def test_archive_contains_weekly_extras(self, reporter):
        _seed_basic(reporter.db)
        # Add ACH evidence this week
        from hormuz.models import ACHEvidence
        reporter.db.insert_ach_evidence(ACHEvidence(
            timestamp=datetime.now(UTC), question="q1",
            evidence_id=1, direction="h1", confidence="high",
            notes="ASBM launch detected"))
        path = reporter.archive_weekly(force=True)
        content = path.read_text()
        assert "ASBM" in content


class TestGatherData:
    def test_gather_returns_dict(self, reporter):
        data = reporter._gather_status_data()
        assert isinstance(data, dict)
        assert "q1_regime" in data
        assert "q2_regime" in data
        assert "phys_params" in data
        assert "signals" in data

    def test_gather_with_data(self, reporter):
        _seed_basic(reporter.db)
        data = reporter._gather_status_data()
        assert data["q1_regime"] is not None
        assert data["path_weights"] is not None


class TestRender:
    def test_render_returns_html_string(self, reporter):
        data = reporter._gather_status_data()
        html = reporter._render("status.html.jinja", data)
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html
