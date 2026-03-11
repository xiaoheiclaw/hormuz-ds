"""Tests for the database layer."""
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hormuz.db import HormuzDB
from hormuz.models import (
    ACHEvidence,
    MCParams,
    MCResult,
    Observation,
    PathWeights,
    PositionSignal,
    Regime,
    RegimeType,
    Signal,
    SignalStatus,
)


class TestDBInit:
    def test_creates_tables(self, tmp_db: Path):
        db = HormuzDB(tmp_db)
        tables = db.list_tables()
        expected = [
            "observations",
            "ach_evidence",
            "regimes",
            "signals",
            "mc_params",
            "mc_results",
            "position_signals",
        ]
        for t in expected:
            assert t in tables, f"missing table: {t}"


class TestObservations:
    def test_insert_and_query(self, tmp_db: Path):
        db = HormuzDB(tmp_db)
        now = datetime.now(UTC)
        obs = Observation(
            timestamp=now,
            source="yfinance",
            category="market",
            key="brent_front",
            value=85.5,
            metadata={"unit": "USD/bbl"},
        )
        row_id = db.insert_observation(obs)
        assert row_id >= 1

        results = db.get_observations_since(now - timedelta(seconds=1))
        assert len(results) == 1
        r = results[0]
        assert r.id == row_id
        assert r.key == "brent_front"
        assert r.value == 85.5
        assert r.metadata == {"unit": "USD/bbl"}

    def test_query_by_category(self, tmp_db: Path):
        db = HormuzDB(tmp_db)
        now = datetime.now(UTC)
        db.insert_observation(Observation(
            timestamp=now, source="yfinance", category="market",
            key="brent", value=85.0,
        ))
        db.insert_observation(Observation(
            timestamp=now, source="centcom", category="q1_attack",
            key="centcom_press", value=1.0,
        ))

        market_only = db.get_observations_since(
            now - timedelta(seconds=1), category="market"
        )
        assert len(market_only) == 1
        assert market_only[0].category == "market"


class TestACHEvidence:
    def test_insert_and_query_by_question(self, tmp_db: Path):
        db = HormuzDB(tmp_db)
        now = datetime.now(UTC)
        ev = ACHEvidence(
            timestamp=now,
            question="q1",
            evidence_id=1,
            direction="h1",
            confidence="high",
            notes="test evidence",
        )
        row_id = db.insert_ach_evidence(ev)
        assert row_id >= 1

        results = db.get_ach_evidence("q1")
        assert len(results) == 1
        assert results[0].direction == "h1"

    def test_get_recent_high_confidence(self, tmp_db: Path):
        db = HormuzDB(tmp_db)
        now = datetime.now(UTC)
        db.insert_ach_evidence(ACHEvidence(
            timestamp=now, question="q1", evidence_id=1,
            direction="h1", confidence="high",
        ))
        db.insert_ach_evidence(ACHEvidence(
            timestamp=now, question="q1", evidence_id=2,
            direction="h2", confidence="low",
        ))

        high_only = db.get_ach_evidence("q1", confidence="high")
        assert len(high_only) == 1
        assert high_only[0].confidence == "high"


class TestSignals:
    def test_insert_and_get_active(self, tmp_db: Path):
        db = HormuzDB(tmp_db)
        now = datetime.now(UTC)
        sig = Signal(
            timestamp=now,
            signal_id="T1",
            status=SignalStatus.triggered,
            action_taken="alert",
        )
        row_id = db.insert_signal(sig)
        assert row_id >= 1

        active = db.get_active_signals()
        assert len(active) == 1
        assert active[0].signal_id == "T1"

    def test_update_signal_status(self, tmp_db: Path):
        db = HormuzDB(tmp_db)
        now = datetime.now(UTC)
        row_id = db.insert_signal(Signal(
            timestamp=now,
            signal_id="T1",
            status=SignalStatus.triggered,
            action_taken="alert",
        ))

        db.update_signal_status(row_id, SignalStatus.reverted)
        active = db.get_active_signals()
        assert len(active) == 0


class TestRegimes:
    def test_get_latest_regime(self, tmp_db: Path):
        db = HormuzDB(tmp_db)
        old = datetime(2025, 1, 1, tzinfo=UTC)
        new = datetime(2025, 6, 1, tzinfo=UTC)
        db.insert_regime(Regime(
            timestamp=old, question="q1",
            regime=RegimeType.wide, trigger="init",
        ))
        db.insert_regime(Regime(
            timestamp=new, question="q1",
            regime=RegimeType.lean_h1, trigger="ach_convergence",
        ))

        latest = db.get_latest_regime("q1")
        assert latest is not None
        assert latest.regime == RegimeType.lean_h1
        assert latest.timestamp == new


class TestMCParams:
    def test_insert_and_get_latest(self, tmp_db: Path):
        db = HormuzDB(tmp_db)
        now = datetime.now(UTC)
        params = MCParams(
            timestamp=now,
            irgc_decay_mean=14.0,
            convoy_start_mean=7.0,
            disruption_range=(0.3, 0.7),
            pipeline_max=1.5,
            pipeline_ramp_weeks=4.0,
            spr_rate_mean=1.0,
            spr_delay_weeks=2.0,
            surplus_buffer=1.5,
            path_weights=PathWeights(a=0.3, b=0.5, c=0.2),
            trigger="regime_change",
        )
        row_id = db.insert_mc_params(params)
        assert row_id >= 1

        latest = db.get_latest_mc_params()
        assert latest is not None
        assert latest.id == row_id
        assert latest.irgc_decay_mean == 14.0
        assert latest.disruption_range == (0.3, 0.7)
        assert latest.path_weights.a == pytest.approx(0.3)
        assert latest.path_weights.b == pytest.approx(0.5)
        assert latest.path_weights.c == pytest.approx(0.2)
        assert latest.trigger == "regime_change"


class TestMCResults:
    def test_insert_returns_id(self, tmp_db: Path):
        db = HormuzDB(tmp_db)
        now = datetime.now(UTC)
        # Need a params row first for the FK
        params = MCParams(
            timestamp=now,
            irgc_decay_mean=14.0,
            convoy_start_mean=7.0,
            disruption_range=(0.3, 0.7),
            pipeline_max=1.5,
            pipeline_ramp_weeks=4.0,
            spr_rate_mean=1.0,
            spr_delay_weeks=2.0,
            surplus_buffer=1.5,
            path_weights=PathWeights(a=0.3, b=0.5, c=0.2),
            trigger="regime_change",
        )
        params_id = db.insert_mc_params(params)

        result = MCResult(
            timestamp=now,
            params_id=params_id,
            price_mean=95.0,
            price_p10=80.0,
            price_p50=93.0,
            price_p90=115.0,
            path_a_price=88.0,
            path_b_price=95.0,
            path_c_price=120.0,
            key_dates={"escalation": now, "peak": now + timedelta(days=30)},
        )
        row_id = db.insert_mc_result(result)
        assert row_id >= 1


class TestPositionSignals:
    def test_get_unexecuted(self, tmp_db: Path):
        db = HormuzDB(tmp_db)
        now = datetime.now(UTC)

        id1 = db.insert_position_signal(PositionSignal(
            timestamp=now, trigger="mc_output",
            action="buy_call_spread", executed=False,
        ))
        db.insert_position_signal(PositionSignal(
            timestamp=now, trigger="tripwire",
            action="hedge_delta", executed=True,
        ))

        unexecuted = db.get_unexecuted_position_signals()
        assert len(unexecuted) == 1
        assert unexecuted[0].action == "buy_call_spread"

        db.mark_position_executed(id1)
        assert len(db.get_unexecuted_position_signals()) == 0
