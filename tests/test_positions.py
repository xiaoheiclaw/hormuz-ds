"""Tests for position rules engine."""
from datetime import datetime, timedelta, UTC
from pathlib import Path

import pytest

from hormuz.db import HormuzDB
from hormuz.engine.positions import PositionEngine
from hormuz.models import Observation, PositionSignal


@pytest.fixture
def pos_engine(tmp_db, parameters) -> PositionEngine:
    db = HormuzDB(tmp_db)
    return PositionEngine(db, parameters["positions"])


class TestHardStopLoss:
    def test_brent_below_80_three_days(self, pos_engine):
        """v5.4: <$80 = external shock (logic paradox with net gap)."""
        obs = [
            Observation(timestamp=datetime(2026, 3, 9, tzinfo=UTC), source="yfinance", category="market", key="brent_price", value=79.0),
            Observation(timestamp=datetime(2026, 3, 10, tzinfo=UTC), source="yfinance", category="market", key="brent_price", value=78.0),
            Observation(timestamp=datetime(2026, 3, 11, tzinfo=UTC), source="yfinance", category="market", key="brent_price", value=77.0),
        ]
        signals = pos_engine.evaluate(observations=obs)
        assert any("强制平所有能源超配" in s.action for s in signals)

    def test_brent_above_80_no_stop(self, pos_engine):
        obs = [
            Observation(timestamp=datetime(2026, 3, 9, tzinfo=UTC), source="yfinance", category="market", key="brent_price", value=85.0),
            Observation(timestamp=datetime(2026, 3, 10, tzinfo=UTC), source="yfinance", category="market", key="brent_price", value=83.0),
            Observation(timestamp=datetime(2026, 3, 11, tzinfo=UTC), source="yfinance", category="market", key="brent_price", value=81.0),
        ]
        signals = pos_engine.evaluate(observations=obs)
        assert not any("强制平" in s.action for s in signals)

    def test_brent_only_two_days_no_stop(self, pos_engine):
        """Only 2 days below 80 -> no stop loss."""
        obs = [
            Observation(timestamp=datetime(2026, 3, 10, tzinfo=UTC), source="yfinance", category="market", key="brent_price", value=79.0),
            Observation(timestamp=datetime(2026, 3, 11, tzinfo=UTC), source="yfinance", category="market", key="brent_price", value=78.0),
        ]
        signals = pos_engine.evaluate(observations=obs)
        assert not any("强制平" in s.action for s in signals)

    def test_brent_above_150_demand_destruction(self, pos_engine):
        """v5.4: >$150 = demand destruction terminal (path C endgame)."""
        obs = [
            Observation(timestamp=datetime(2026, 3, 11, tzinfo=UTC), source="yfinance", category="market", key="brent_price", value=155.0),
        ]
        signals = pos_engine.evaluate(observations=obs)
        assert any("清能源多头" in s.action for s in signals)

    def test_brent_below_150_no_demand_destruction(self, pos_engine):
        """$140 should NOT trigger demand destruction."""
        obs = [
            Observation(timestamp=datetime(2026, 3, 11, tzinfo=UTC), source="yfinance", category="market", key="brent_price", value=140.0),
        ]
        signals = pos_engine.evaluate(observations=obs)
        assert not any("清能源多头" in s.action for s in signals)

    def test_portfolio_loss_exceeds_max(self, pos_engine):
        """Total portfolio loss > 8% -> halve all positions."""
        obs = [
            Observation(timestamp=datetime(2026, 3, 11, tzinfo=UTC), source="manual", category="market", key="portfolio_loss_pct", value=9.0),
        ]
        signals = pos_engine.evaluate(observations=obs)
        assert any("减半" in s.action for s in signals)

    def test_portfolio_loss_within_limit(self, pos_engine):
        """Loss <= 8% -> no signal."""
        obs = [
            Observation(timestamp=datetime(2026, 3, 11, tzinfo=UTC), source="manual", category="market", key="portfolio_loss_pct", value=7.0),
        ]
        signals = pos_engine.evaluate(observations=obs)
        assert not any("减半" in s.action for s in signals)


class TestMCDrivenRules:
    def test_high_attack_freq_adds_energy(self, pos_engine):
        obs = []
        for day in range(7):
            obs.append(Observation(
                timestamp=datetime(2026, 3, 5 + day, tzinfo=UTC),
                source="centcom", category="q1_attack", key="attack_frequency", value=4.0,
            ))
            obs.append(Observation(
                timestamp=datetime(2026, 3, 5 + day, tzinfo=UTC),
                source="readwise", category="market", key="transit_volume", value=2.5,
            ))
        signals = pos_engine.evaluate(observations=obs)
        assert any("22%" in s.action for s in signals)

    def test_high_attack_but_transit_ok_no_signal(self, pos_engine):
        """Attack freq > 3/day for 7 days but transit >= 3 mbd -> no add."""
        obs = []
        for day in range(7):
            obs.append(Observation(
                timestamp=datetime(2026, 3, 5 + day, tzinfo=UTC),
                source="centcom", category="q1_attack", key="attack_frequency", value=4.0,
            ))
            obs.append(Observation(
                timestamp=datetime(2026, 3, 5 + day, tzinfo=UTC),
                source="readwise", category="market", key="transit_volume", value=5.0,
            ))
        signals = pos_engine.evaluate(observations=obs)
        assert not any("22%" in s.action for s in signals)

    def test_low_attack_freq_signals_decay_adjust(self, pos_engine):
        obs = [
            Observation(
                timestamp=datetime(2026, 3, 7 + day, tzinfo=UTC),
                source="centcom", category="q1_attack", key="attack_frequency", value=0.5,
            )
            for day in range(5)
        ]
        signals = pos_engine.evaluate(observations=obs)
        assert any("irgcDecayMean" in s.action for s in signals)

    def test_low_attack_only_four_days_no_signal(self, pos_engine):
        """Attack freq < 1/day for only 4 days -> no signal."""
        obs = [
            Observation(
                timestamp=datetime(2026, 3, 8 + day, tzinfo=UTC),
                source="centcom", category="q1_attack", key="attack_frequency", value=0.5,
            )
            for day in range(4)
        ]
        signals = pos_engine.evaluate(observations=obs)
        assert not any("irgcDecayMean" in s.action for s in signals)

    def test_transit_partial_recovery(self, pos_engine):
        """Transit > 8 mbd for 3 days -> unwind half energy overweight."""
        obs = [
            Observation(
                timestamp=datetime(2026, 3, 9 + day, tzinfo=UTC),
                source="readwise", category="market", key="transit_volume", value=9.0,
            )
            for day in range(3)
        ]
        signals = pos_engine.evaluate(observations=obs)
        assert any("1/2" in s.action for s in signals)

    def test_transit_recovery_triggers_unwind(self, pos_engine):
        """Transit > 12 mbd for 5 days -> unwind all."""
        obs = [
            Observation(
                timestamp=datetime(2026, 3, 7 + day, tzinfo=UTC),
                source="readwise", category="market", key="transit_volume", value=13.0,
            )
            for day in range(5)
        ]
        signals = pos_engine.evaluate(observations=obs)
        assert any("基准" in s.action or "全部" in s.action for s in signals)

    def test_no_observations_no_signals(self, pos_engine):
        """Empty observations -> no signals."""
        signals = pos_engine.evaluate(observations=[])
        assert signals == []

    def test_signals_have_executed_false(self, pos_engine):
        """All signals returned should have executed=False."""
        obs = [
            Observation(timestamp=datetime(2026, 3, 9, tzinfo=UTC), source="yfinance", category="market", key="brent_price", value=79.0),
            Observation(timestamp=datetime(2026, 3, 10, tzinfo=UTC), source="yfinance", category="market", key="brent_price", value=78.0),
            Observation(timestamp=datetime(2026, 3, 11, tzinfo=UTC), source="yfinance", category="market", key="brent_price", value=77.0),
        ]
        signals = pos_engine.evaluate(observations=obs)
        assert all(s.executed is False for s in signals)

    def test_signals_are_position_signal_type(self, pos_engine):
        """All returned items should be PositionSignal instances."""
        obs = [
            Observation(timestamp=datetime(2026, 3, 9, tzinfo=UTC), source="yfinance", category="market", key="brent_price", value=79.0),
            Observation(timestamp=datetime(2026, 3, 10, tzinfo=UTC), source="yfinance", category="market", key="brent_price", value=78.0),
            Observation(timestamp=datetime(2026, 3, 11, tzinfo=UTC), source="yfinance", category="market", key="brent_price", value=77.0),
        ]
        signals = pos_engine.evaluate(observations=obs)
        assert all(isinstance(s, PositionSignal) for s in signals)
