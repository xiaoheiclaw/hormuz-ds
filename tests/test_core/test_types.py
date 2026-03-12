import pytest
from datetime import datetime


def test_constants_immutable():
    from hormuz.core.types import Constants
    c = Constants()
    assert c.exposed_supply_mbd == 20.1
    assert c.strait_width_km == 9.0


def test_parameters_defaults():
    from hormuz.core.types import Parameters
    p = Parameters()
    assert p.mines_in_water_range == (20, 100)
    assert p.spr_pump_min_days == 13
    assert p.pipeline_max_mbd == 4.0


def test_observation():
    from hormuz.core.types import Observation
    o = Observation(id="O01", timestamp=datetime(2026, 3, 12), value=3.5, source="CENTCOM")
    assert o.id == "O01"


def test_state_vector_defaults():
    from hormuz.core.types import StateVector
    sv = StateVector()
    assert sv.disruption_rate == 0.80
    assert sv.buffer_mbd == 0.0


def test_control():
    from hormuz.core.types import Control
    c = Control(id="D01", actor="US_NAVY", triggered=False)
    assert not c.triggered


def test_ach_posterior_validation():
    from hormuz.core.types import ACHPosterior
    p = ACHPosterior(h1=0.6, h2=0.4, h3=None)
    assert p.dominant == "inconclusive"
    p2 = ACHPosterior(h1=0.75, h2=0.25, h3=None)
    assert p2.dominant == "H1"


def test_path_weights_normalize():
    from hormuz.core.types import PathWeights
    pw = PathWeights(a=0.50, b=0.40, c=0.30)
    pw = pw.normalized()
    assert abs(pw.a + pw.b + pw.c - 1.0) < 1e-9


def test_path_weights_clip():
    from hormuz.core.types import PathWeights
    pw = PathWeights(a=0.95, b=0.04, c=0.01)
    pw = pw.normalized()
    assert pw.a <= 0.85
    assert pw.c >= 0.05


def test_system_output():
    from hormuz.core.types import SystemOutput, ACHPosterior, PathWeights
    so = SystemOutput(
        timestamp=datetime(2026, 3, 12),
        ach_posterior=ACHPosterior(h1=0.5, h2=0.5, h3=None),
        t1_percentiles={"p50": 21},
        t2_percentiles={"p50": 35},
        t_total_percentiles={"p50": 63},
        buffer_trajectory=[(0, 0.0), (14, 1.5), (30, 7.0)],
        gross_gap_mbd=16.0,
        net_gap_trajectories={"A": [(0, 16.0), (14, 14.5)]},
        path_probabilities=PathWeights(),
        path_total_gaps={"A": 270.0, "B": 833.0, "C": 2500.0},
        expected_total_gap=700.0,
        consistency_flags=[],
    )
    assert so.gross_gap_mbd == 16.0
