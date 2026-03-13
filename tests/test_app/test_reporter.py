"""Reporter tests — render_status produces valid two-tab HTML."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from hormuz.core.types import (
    ACHPosterior,
    Parameters,
    PathWeights,
    SystemOutput,
)
from hormuz.core.mc import MCResult
from hormuz.core.m5_game import SignalEvidence
from hormuz.app.reporter import render_status, _build_signal_display


@pytest.fixture
def system_output():
    return SystemOutput(
        timestamp=datetime(2026, 3, 13, 12, 0),
        ach_posterior=ACHPosterior(h1=0.72, h2=0.28),
        t1_percentiles={"p10": 10, "p50": 25, "p90": 60},
        t2_percentiles={"p10": 5, "p50": 15, "p90": 40},
        t_total_percentiles={"p10": 20, "p50": 45, "p90": 100},
        buffer_trajectory=[(0, 0.0), (7, 1.0), (30, 4.0)],
        gross_gap_mbd=16.0,
        net_gap_trajectories={"A": [(0, 16.0), (28, 12.0)]},
        path_probabilities=PathWeights(a=0.35, b=0.45, c=0.20),
        path_total_gaps={"A": 300, "B": 900, "C": 2000},
        expected_total_gap=810,
        consistency_flags=[],
    )


@pytest.fixture
def mc_result():
    rng = np.random.default_rng(42)
    t = rng.uniform(10, 150, size=100)
    gap = rng.uniform(100, 2000, size=100)
    return MCResult(
        t_samples=t,
        total_gap_samples=gap,
        t_percentiles={"p10": 20, "p50": 60, "p90": 130},
        path_frequencies={"A": 0.30, "B": 0.50, "C": 0.20},
        path_total_gap_means={"A": 300, "B": 900, "C": 2000},
    )


def test_render_produces_html(system_output, mc_result, tmp_path):
    out = tmp_path / "status.html"
    render_status(
        system_output=system_output,
        mc_result=mc_result,
        params=Parameters(),
        output_path=out,
        brent_price=95.0,
    )
    assert out.exists()
    html = out.read_text()
    assert "霍尔木兹决策系统" in html
    assert "实时状态" in html
    assert "框架参考" in html


def test_render_with_signals(system_output, mc_result, tmp_path):
    out = tmp_path / "status.html"
    signals = [
        SignalEvidence("external_mediation", 0.8),
        SignalEvidence("irgc_escalation", 1.0),
    ]
    render_status(
        system_output=system_output,
        mc_result=mc_result,
        params=Parameters(),
        output_path=out,
        game_signals=signals,
    )
    html = out.read_text()
    assert "第三方斡旋" in html
    assert "IRGC" in html


def test_render_with_flags(system_output, mc_result, tmp_path):
    system_output.consistency_flags = ["gross_gap unexpectedly low: 5.0 mbd"]
    out = tmp_path / "status.html"
    render_status(
        system_output=system_output,
        mc_result=mc_result,
        params=Parameters(),
        output_path=out,
    )
    html = out.read_text()
    assert "gross_gap unexpectedly low" in html


def test_build_signal_display():
    signals = [SignalEvidence("costly_self_binding", 1.0)]
    rows = _build_signal_display(signals)
    assert len(rows) == 1
    assert rows[0]["direction"] == "A"
    assert rows[0]["evidence_label"] == "high"


def test_build_signal_display_unknown():
    rows = _build_signal_display([SignalEvidence("nonexistent", 1.0)])
    assert len(rows) == 0
