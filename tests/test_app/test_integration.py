"""Full pipeline integration test with fixture data, no external IO."""
import pytest
from pathlib import Path
from datetime import datetime


@pytest.fixture
def setup(tmp_path):
    from hormuz.infra.db import init_db
    db = tmp_path / "test.db"
    init_db(db)
    return {"db_path": db, "output_dir": tmp_path, "configs_dir": Path(__file__).parents[2] / "configs"}


def test_engine_run_produces_valid_output(setup):
    from hormuz.app.pipeline import engine_run
    from hormuz.core.variables import load_constants, load_parameters
    from hormuz.core.types import SystemOutput
    constants = load_constants(setup["configs_dir"] / "constants.yaml")
    params = load_parameters(setup["configs_dir"] / "parameters.yaml")
    so = engine_run(constants, params, observations=[], controls=[], events={}, mc_n=100, seed=42)
    assert isinstance(so, SystemOutput)
    assert so.gross_gap_mbd > 15
    assert so.path_probabilities.a + so.path_probabilities.b + so.path_probabilities.c == pytest.approx(1.0)
    assert so.path_total_gaps["A"] < so.path_total_gaps["B"] < so.path_total_gaps["C"]


def test_full_roundtrip(setup):
    """Engine run -> save to DB -> load from DB -> render HTML"""
    from hormuz.app.pipeline import engine_run
    from hormuz.app.reporter import render_status
    from hormuz.infra.db import save_system_output, get_latest_output
    from hormuz.core.variables import load_constants, load_parameters
    from hormuz.core.mc import run_monte_carlo
    from hormuz.core.types import ACHPosterior

    constants = load_constants(setup["configs_dir"] / "constants.yaml")
    params = load_parameters(setup["configs_dir"] / "parameters.yaml")
    so = engine_run(constants, params, [], [], {}, mc_n=100, seed=42)

    # Save + load
    save_system_output(setup["db_path"], so)
    loaded = get_latest_output(setup["db_path"])
    assert loaded is not None
    assert loaded.gross_gap_mbd == so.gross_gap_mbd

    # Render
    mc_result = run_monte_carlo(
        ACHPosterior(h1=0.5, h2=0.5, h3=None), params, {}, n=100, seed=42
    )
    out_html = setup["output_dir"] / "status.html"
    render_status(so, mc_result, params, output_path=out_html,
                  conflict_start="2026-03-01", brent_price=95.0)
    assert out_html.exists()
    assert len(out_html.read_text()) > 1000


def test_all_core_tests_pass():
    """Meta: ensure pytest collects and passes all core tests"""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/test_core/", "-v", "--tb=short"],
        capture_output=True, text=True, cwd=str(Path(__file__).parents[2])
    )
    assert result.returncode == 0
