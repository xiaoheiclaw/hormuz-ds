import pytest
from hormuz.core.types import PathWeights, SystemOutput, ACHPosterior
from datetime import datetime


def make_output(**kwargs):
    defaults = dict(
        timestamp=datetime(2026, 3, 12),
        ach_posterior=ACHPosterior(h1=0.5, h2=0.5, h3=None),
        t1_percentiles={"p50": 21}, t2_percentiles={"p50": 35},
        t_total_percentiles={"p50": 63},
        buffer_trajectory=[], gross_gap_mbd=16.0,
        net_gap_trajectories={}, path_probabilities=PathWeights(),
        path_total_gaps={"A": 270, "B": 833, "C": 2500},
        expected_total_gap=700, consistency_flags=[],
    )
    defaults.update(kwargs)
    return SystemOutput(**defaults)


def test_base_positions():
    from hormuz.app.positions import evaluate_positions
    so = make_output()
    signals = []
    result = evaluate_positions(so, brent_price=95.0, signals=signals)
    assert result.energy_pct == 15
    assert result.vol_pct == 3
    assert result.recession_pct == 2


def test_t_end_exit():
    """Transit up 3 days + AP < 1% -> unwind"""
    from hormuz.app.positions import evaluate_positions
    so = make_output()
    result = evaluate_positions(so, brent_price=85.0, signals=[], t_end_confirmed=True)
    assert result.energy_pct < 15  # reducing
    assert result.vol_pct == 0     # closed


def test_demand_destruction():
    """Brent > 150 -> clear energy, double recession"""
    from hormuz.app.positions import evaluate_positions
    so = make_output()
    result = evaluate_positions(so, brent_price=155.0, signals=[])
    assert result.energy_pct == 0
    assert result.recession_pct == 4


def test_system_failure():
    """Brent < 80 for 3 days -> force close all"""
    from hormuz.app.positions import evaluate_positions
    so = make_output()
    result = evaluate_positions(so, brent_price=78.0, signals=[], brent_below_80_days=3)
    assert result.energy_pct == 0
    assert result.vol_pct == 0


def test_tripwire_override():
    """T1a signal -> vol×2"""
    from hormuz.app.positions import evaluate_positions
    so = make_output()
    result = evaluate_positions(so, brent_price=95.0, signals=[{"action": "vol_double"}])
    assert result.vol_pct == 6
