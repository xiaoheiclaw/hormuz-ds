"""Tests for the physical layer: state equations and regime→parameter mapping."""
import pytest

from hormuz.engine.physical import PhysicalLayer
from hormuz.models import RegimeType


@pytest.fixture
def physical(parameters) -> PhysicalLayer:
    return PhysicalLayer(parameters["physical"])


class TestRegimeToParams:
    def test_wide_keeps_defaults(self, physical):
        params = physical.update_params(q1_regime=RegimeType.wide, q2_regime=RegimeType.wide)
        assert params["irgc_decay_mean"] == 6.0
        assert params["convoy_start_mean"] == 5.0

    def test_lean_h1_speeds_decay(self, physical):
        params = physical.update_params(q1_regime=RegimeType.lean_h1, q2_regime=RegimeType.wide)
        assert params["irgc_decay_mean"] < 6.0  # should be ~4.0

    def test_lean_h2_no_change(self, physical):
        params = physical.update_params(q1_regime=RegimeType.lean_h2, q2_regime=RegimeType.wide)
        assert params["irgc_decay_mean"] == 6.0

    def test_confirmed_h3_extends(self, physical):
        params = physical.update_params(q1_regime=RegimeType.confirmed_h3, q2_regime=RegimeType.wide)
        assert params["irgc_decay_mean"] > 12.0  # should be ~15.0

    def test_q2_lean_h1_shortens_convoy(self, physical):
        params = physical.update_params(q1_regime=RegimeType.wide, q2_regime=RegimeType.lean_h1)
        assert params["convoy_start_mean"] < 5.0  # should be ~4.0

    def test_q2_lean_h2_extends_convoy(self, physical):
        params = physical.update_params(q1_regime=RegimeType.wide, q2_regime=RegimeType.lean_h2)
        assert params["convoy_start_mean"] > 5.0  # should be ~6.5

    def test_all_params_present(self, physical):
        params = physical.update_params(q1_regime=RegimeType.wide, q2_regime=RegimeType.wide)
        for key in [
            "irgc_decay_mean", "convoy_start_mean", "disruption_range",
            "pipeline_max", "pipeline_ramp_weeks", "spr_rate_mean",
            "spr_delay_weeks", "surplus_buffer",
        ]:
            assert key in params


class TestQ1Decay:
    def test_exponential_decay(self, physical):
        cap_w0 = physical.q1_capability(week=0, decay_mean=6.0)
        cap_w6 = physical.q1_capability(week=6, decay_mean=6.0)
        assert abs(cap_w0 - 1.0) < 0.01
        assert abs(cap_w6 / cap_w0 - 0.368) < 0.01  # e^(-1)

    def test_faster_decay_lower_capability(self, physical):
        cap_fast = physical.q1_capability(week=4, decay_mean=4.0)
        cap_slow = physical.q1_capability(week=4, decay_mean=8.0)
        assert cap_fast < cap_slow


class TestQ2NetFlow:
    def test_positive_flow_mines_accumulating(self, physical):
        flow = physical.q2_net_flow(deploy_rate=5.0, sweep_rate=3.0, q1_attack_freq=1.0)
        assert flow == 2.0  # 5 - 3

    def test_q1_gate_blocks_sweeping(self, physical):
        flow = physical.q2_net_flow(deploy_rate=5.0, sweep_rate=3.0, q1_attack_freq=3.0)
        assert flow == 5.0  # sweep blocked, all deploy goes through

    def test_negative_flow_mines_clearing(self, physical):
        flow = physical.q2_net_flow(deploy_rate=1.0, sweep_rate=3.0, q1_attack_freq=0.5)
        assert flow == -2.0


class TestQ3Buffer:
    def test_buffer_ramp(self, physical):
        buf_d1 = physical.q3_buffer(day=1)
        buf_d14 = physical.q3_buffer(day=14)
        buf_d30 = physical.q3_buffer(day=30)
        assert buf_d1 < buf_d14 < buf_d30
        assert buf_d30 <= 10.0

    def test_spr_delay(self, physical):
        """SPR doesn't kick in until spr_delay_weeks * 7 days."""
        buf_early = physical.q3_buffer(day=5)  # before SPR
        buf_after_spr = physical.q3_buffer(day=25)  # after SPR delay
        # The jump from SPR should be significant
        assert buf_after_spr - buf_early > 1.0

    def test_day_zero(self, physical):
        buf = physical.q3_buffer(day=0)
        # Day 0: only surplus buffer available
        assert buf > 0  # surplus_buffer = 2.5 mbd
