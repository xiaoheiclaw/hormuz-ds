import pytest
from hormuz.core.types import Parameters, Control


def test_buffer_day_0():
    from hormuz.core.m3_buffer import compute_buffer
    params = Parameters()
    assert compute_buffer(day=0, params=params) == pytest.approx(0.0, abs=0.1)


def test_buffer_day_2():
    """Before ADCOP kicks in"""
    from hormuz.core.m3_buffer import compute_buffer
    params = Parameters()
    assert compute_buffer(day=2, params=params) == pytest.approx(0.0, abs=0.1)


def test_buffer_day_10():
    """During pipeline ramp"""
    from hormuz.core.m3_buffer import compute_buffer
    params = Parameters()
    b = compute_buffer(day=10, params=params)
    assert 0.5 < b < 3.0  # partial ramp


def test_buffer_day_30():
    """Steady state"""
    from hormuz.core.m3_buffer import compute_buffer
    params = Parameters()
    b = compute_buffer(day=30, params=params)
    assert 5.0 < b < 9.0  # ~7 mbd steady state


def test_pipeline_component():
    from hormuz.core.m3_buffer import pipeline_buffer
    params = Parameters()
    assert pipeline_buffer(day=2, params=params) == pytest.approx(0.0, abs=0.01)
    assert pipeline_buffer(day=4, params=params) > 0  # ADCOP starting
    p14 = pipeline_buffer(day=14, params=params)
    assert p14 > 2.0  # ADCOP + Saudi pipeline near steady


def test_spr_component_no_trigger():
    from hormuz.core.m3_buffer import spr_buffer
    params = Parameters()
    assert spr_buffer(day=20, params=params, spr_trigger_day=None) == 0.0


def test_spr_component_with_trigger():
    from hormuz.core.m3_buffer import spr_buffer
    params = Parameters()
    # Triggered day 1, 13 day delay, so arrives ~day 14
    assert spr_buffer(day=10, params=params, spr_trigger_day=1) == 0.0
    assert spr_buffer(day=20, params=params, spr_trigger_day=1) > 0.5


def test_cape_component():
    from hormuz.core.m3_buffer import cape_buffer
    assert cape_buffer(day=10) == pytest.approx(0.0, abs=0.01)
    assert cape_buffer(day=15) > 0  # first arrivals
    assert cape_buffer(day=60) > 1.0  # steady


def test_buffer_trajectory():
    """Generate full trajectory"""
    from hormuz.core.m3_buffer import compute_buffer_trajectory
    params = Parameters()
    traj = compute_buffer_trajectory(max_day=90, params=params)
    assert len(traj) == 91  # day 0..90
    assert traj[0][1] == pytest.approx(0.0, abs=0.1)
    assert traj[-1][1] > 5.0
