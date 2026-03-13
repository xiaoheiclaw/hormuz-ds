import pytest
from datetime import datetime, timedelta
from hormuz.core.types import Observation


def make_obs(obs_id, value, ts=None):
    return Observation(id=obs_id, timestamp=ts or datetime(2026, 3, 12), value=value, source="test")


def test_no_signals():
    from hormuz.app.signals import scan_signals
    obs = [make_obs("O01", 3.0)]
    result = scan_signals(obs, signal_state={})
    assert len(result.triggered) == 0


def test_t1a_triggered():
    """GPS up + attack freq rising -> T1a"""
    from hormuz.app.signals import scan_signals
    obs = [
        make_obs("O05", 0.8),  # GPS spoofing high
        make_obs("O01", 5.0),  # attack freq high
    ]
    result = scan_signals(obs, signal_state={}, o01_trend="rising")
    assert "T1a" in result.triggered


def test_t1b_triggered():
    """GPS up + attack freq falling -> T1b"""
    from hormuz.app.signals import scan_signals
    obs = [make_obs("O05", 0.8), make_obs("O01", 1.5)]
    result = scan_signals(obs, signal_state={}, o01_trend="falling")
    assert "T1b" in result.triggered


def test_48h_revert():
    """T1a should revert after 48h"""
    from hormuz.app.signals import check_reverts
    state = {"T1a": {"triggered_at": datetime(2026, 3, 10, 0, 0)}}
    now = datetime(2026, 3, 12, 1, 0)  # 49h later
    reverted = check_reverts(state, now=now)
    assert "T1a" in reverted


def test_e3_persistent():
    """E3 mine strike is persistent (no revert)"""
    from hormuz.app.signals import scan_signals
    result = scan_signals([], signal_state={}, events={"E3": True})
    assert "E3" in result.triggered


def test_signal_result_has_position_actions():
    from hormuz.app.signals import scan_signals
    result = scan_signals([], signal_state={}, events={"E1": True})
    assert len(result.position_actions) > 0  # E1 -> vol×2 + recession 5%
