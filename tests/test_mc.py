"""Tests for MC model Phase 1 (analytical approximation)."""

from datetime import datetime

import pytest

from hormuz.engine.mc import MCModel
from hormuz.models import MCParams, MCResult, PathWeights


@pytest.fixture
def mc() -> MCModel:
    return MCModel()


@pytest.fixture
def base_params() -> MCParams:
    return MCParams(
        timestamp=datetime(2026, 3, 11),
        irgc_decay_mean=6.0,
        convoy_start_mean=5.0,
        disruption_range=(0.55, 0.90),
        pipeline_max=4.0,
        pipeline_ramp_weeks=2.5,
        spr_rate_mean=2.5,
        spr_delay_weeks=2.5,
        surplus_buffer=2.5,
        path_weights=PathWeights(a=0.3, b=0.5, c=0.2),
        trigger="test",
    )


class TestPhase1:
    def test_produces_result(self, mc, base_params):
        result = mc.run(base_params)
        assert isinstance(result, MCResult)
        assert result.price_p10 < result.price_p50 < result.price_p90

    def test_path_a_lower_than_c(self, mc, base_params):
        result = mc.run(base_params)
        assert result.path_a_price < result.path_c_price

    def test_path_b_between_a_and_c(self, mc, base_params):
        result = mc.run(base_params)
        assert result.path_a_price <= result.path_b_price <= result.path_c_price

    def test_weighted_mean_between_paths(self, mc, base_params):
        result = mc.run(base_params)
        assert result.path_a_price <= result.price_mean <= result.path_c_price

    def test_higher_c_weight_raises_mean(self, mc, base_params):
        result_base = mc.run(base_params)
        high_c = MCParams(
            **{**base_params.model_dump(), "path_weights": PathWeights(a=0.1, b=0.4, c=0.5)}
        )
        result_high_c = mc.run(high_c)
        assert result_high_c.price_mean > result_base.price_mean

    def test_longer_convoy_raises_prices(self, mc, base_params):
        result_base = mc.run(base_params)
        long_convoy = MCParams(**{**base_params.model_dump(), "convoy_start_mean": 10.0})
        result_long = mc.run(long_convoy)
        assert result_long.price_mean > result_base.price_mean

    def test_all_prices_above_base(self, mc, base_params):
        result = mc.run(base_params)
        assert result.path_a_price > MCModel.BASE_PRICE
        assert result.path_b_price > MCModel.BASE_PRICE
        assert result.path_c_price > MCModel.BASE_PRICE

    def test_prices_realistic_range(self, mc, base_params):
        """Prices should be in realistic range for Hormuz crisis."""
        result = mc.run(base_params)
        assert 80 < result.price_mean < 200
        assert 70 < result.price_p10 < 150
        assert 90 < result.price_p90 < 250
