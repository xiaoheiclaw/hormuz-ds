import pytest
from hormuz.core.types import Constants, StateVector


def test_gross_gap():
    from hormuz.core.m4_gap import compute_gross_gap
    c = Constants()
    sv = StateVector()
    assert compute_gross_gap(c, sv) == pytest.approx(16.08, abs=0.1)


def test_net_gap_day_0():
    from hormuz.core.m4_gap import compute_net_gap
    assert compute_net_gap(gross_gap=16.0, buffer=0.0) == 16.0


def test_net_gap_day_30():
    from hormuz.core.m4_gap import compute_net_gap
    assert compute_net_gap(gross_gap=16.0, buffer=7.0) == 9.0


def test_total_gap_path_a():
    """Path A: T~28 days -> ~270 mbd·days"""
    from hormuz.core.m4_gap import integrate_total_gap
    buffer_traj = [(d, 0.0 if d < 3 else 1.5 if d < 14 else 7.0) for d in range(29)]
    tg = integrate_total_gap(gross_gap=16.0, buffer_trajectory=buffer_traj, t_end=28)
    assert 200 < tg < 350


def test_total_gap_path_b():
    """Path B: T~84 days -> ~833 mbd·days"""
    from hormuz.core.m4_gap import integrate_total_gap
    buffer_traj = [(d, 0.0 if d < 3 else 1.5 if d < 14 else 7.0) for d in range(85)]
    tg = integrate_total_gap(gross_gap=16.0, buffer_trajectory=buffer_traj, t_end=84)
    assert 700 < tg < 950


def test_net_gap_trajectory():
    from hormuz.core.m4_gap import compute_net_gap_trajectory
    buffer_traj = [(0, 0.0), (14, 1.5), (30, 7.0)]
    traj = compute_net_gap_trajectory(gross_gap=16.0, buffer_trajectory=buffer_traj)
    assert traj[0][1] == pytest.approx(16.0)
    assert traj[1][1] == pytest.approx(14.5)
    assert traj[2][1] == pytest.approx(9.0)


def test_path_total_gaps():
    """Compute TotalGap for all three paths"""
    from hormuz.core.m4_gap import compute_path_total_gaps
    from hormuz.core.types import Parameters
    params = Parameters()
    gaps = compute_path_total_gaps(gross_gap=16.0, params=params)
    assert "A" in gaps and "B" in gaps and "C" in gaps
    assert gaps["A"] < gaps["B"] < gaps["C"]
