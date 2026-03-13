import pytest
from pathlib import Path
from datetime import datetime
from hormuz.core.types import SystemOutput, ACHPosterior, PathWeights, Parameters
from hormuz.core.mc import MCResult
import numpy as np


def make_system_output():
    return SystemOutput(
        timestamp=datetime(2026, 3, 12),
        ach_posterior=ACHPosterior(h1=0.6, h2=0.4, h3=None),
        t1_percentiles={"p10": 14, "p25": 17, "p50": 21, "p75": 28, "p90": 35},
        t2_percentiles={"p10": 20, "p25": 28, "p50": 35, "p75": 42, "p90": 56},
        t_total_percentiles={"p10": 42, "p25": 52, "p50": 63, "p75": 77, "p90": 98},
        buffer_trajectory=[(d, min(d * 0.5, 7.0)) for d in range(91)],
        gross_gap_mbd=16.0,
        net_gap_trajectories={"A": [(0, 16.0), (28, 5.0)], "B": [(0, 16.0), (84, 9.0)]},
        path_probabilities=PathWeights(a=0.30, b=0.50, c=0.20),
        path_total_gaps={"A": 270.0, "B": 833.0, "C": 2500.0},
        expected_total_gap=700.0,
        consistency_flags=["AP declining but S06 still high"],
    )


def test_render_status(tmp_path):
    from hormuz.app.reporter import render_status
    so = make_system_output()
    params = Parameters()
    mc_result = MCResult(
        t_samples=np.random.default_rng(42).normal(63, 20, 100),
        total_gap_samples=np.random.default_rng(42).normal(700, 200, 100),
        t_percentiles={"p10": 42, "p50": 63, "p90": 98},
        path_frequencies={"A": 0.25, "B": 0.55, "C": 0.20},
        path_total_gap_means={"A": 280.0, "B": 850.0, "C": 2600.0},
    )
    output_path = tmp_path / "status.html"
    render_status(so, mc_result, params, output_path=output_path,
                  conflict_start="2026-03-01", brent_price=95.0)
    assert output_path.exists()
    html = output_path.read_text()
    assert "16.0" in html  # gross gap
    assert "270" in html   # path A gap
    assert "参数" in html   # parameter section in Chinese


def test_render_has_all_sections(tmp_path):
    from hormuz.app.reporter import render_status
    so = make_system_output()
    params = Parameters()
    mc_result = MCResult(
        t_samples=np.ones(10), total_gap_samples=np.ones(10),
        t_percentiles={"p10": 1, "p50": 1, "p90": 1},
        path_frequencies={"A": 0.3, "B": 0.5, "C": 0.2},
        path_total_gap_means={"A": 1, "B": 1, "C": 1},
    )
    out = tmp_path / "status.html"
    render_status(so, mc_result, params, output_path=out,
                  conflict_start="2026-03-01", brent_price=95.0)
    html = out.read_text()
    # Check all 9 sections present
    for section in ["状态总览", "核心公式", "物理层", "博弈层", "路径",
                    "MC", "仓位", "参数", "校验"]:
        assert section in html, f"Missing section: {section}"
