import pytest
import sqlite3
from pathlib import Path
from datetime import datetime


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


def test_init_db(db_path):
    from hormuz.infra.db import init_db
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "observations" in tables
    assert "ach_evidence" in tables
    assert "state_snapshots" in tables
    assert "controls" in tables
    assert "mc_runs" in tables
    assert "system_outputs" in tables
    assert "position_signals" in tables
    assert "parameters_override" in tables
    conn.close()


def test_insert_observation(db_path):
    from hormuz.infra.db import init_db, insert_observation, get_observations
    from hormuz.core.types import Observation
    init_db(db_path)
    obs = Observation(id="O01", timestamp=datetime(2026, 3, 12), value=3.5, source="CENTCOM")
    insert_observation(db_path, obs)
    results = get_observations(db_path, since=datetime(2026, 3, 11))
    assert len(results) == 1
    assert results[0].id == "O01"


def test_insert_control(db_path):
    from hormuz.infra.db import init_db, insert_control, get_controls
    from hormuz.core.types import Control
    init_db(db_path)
    ctrl = Control(id="D02", actor="WHITE_HOUSE", triggered=True, trigger_time=datetime(2026, 3, 12))
    insert_control(db_path, ctrl)
    results = get_controls(db_path)
    assert len(results) == 1
    assert results[0].triggered


def test_save_system_output(db_path):
    from hormuz.infra.db import init_db, save_system_output, get_latest_output
    from hormuz.core.types import SystemOutput, ACHPosterior, PathWeights
    init_db(db_path)
    so = SystemOutput(
        timestamp=datetime(2026, 3, 12),
        ach_posterior=ACHPosterior(h1=0.5, h2=0.5, h3=None),
        t1_percentiles={"p50": 21}, t2_percentiles={"p50": 35},
        t_total_percentiles={"p50": 63},
        buffer_trajectory=[(0, 0.0)], gross_gap_mbd=16.0,
        net_gap_trajectories={}, path_probabilities=PathWeights(),
        path_total_gaps={"A": 270.0, "B": 833.0, "C": 2500.0},
        expected_total_gap=700.0, consistency_flags=[],
    )
    save_system_output(db_path, so)
    latest = get_latest_output(db_path)
    assert latest is not None
    assert latest.gross_gap_mbd == 16.0


def test_save_parameter_override(db_path):
    from hormuz.infra.db import init_db, save_parameter_override, get_parameter_overrides
    init_db(db_path)
    save_parameter_override(db_path, param="mines_in_water_range", old_value="(20, 100)", new_value="(30, 80)")
    overrides = get_parameter_overrides(db_path)
    assert len(overrides) == 1
