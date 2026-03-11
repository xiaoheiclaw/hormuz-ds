"""Tests for hormuz data models."""
from datetime import datetime, timedelta

import pytest

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


class TestObservation:
    def test_create_market_observation(self):
        obs = Observation(
            timestamp=datetime(2026, 3, 10, 12, 0),
            source="yfinance",
            category="market",
            key="CL1",
            value=82.5,
        )
        assert obs.source == "yfinance"
        assert obs.category == "market"
        assert obs.value == 82.5
        assert obs.id is None
        assert obs.metadata is None

    def test_create_attack_observation_with_metadata(self):
        obs = Observation(
            timestamp=datetime(2026, 3, 10, 12, 0),
            source="centcom",
            category="q1_attack",
            key="houthi_strike",
            value=1.0,
            metadata={"type": "drone", "target": "tanker", "region": "bab_el_mandeb"},
        )
        assert obs.metadata["type"] == "drone"
        assert obs.metadata["target"] == "tanker"
        assert obs.metadata["region"] == "bab_el_mandeb"

    def test_observation_source_validation(self):
        with pytest.raises(ValueError):
            Observation(
                timestamp=datetime(2026, 3, 10),
                source="bloomberg",
                category="market",
                key="test",
                value=1.0,
            )


class TestACHEvidence:
    def test_create_q1_evidence(self):
        ev = ACHEvidence(
            timestamp=datetime(2026, 3, 10),
            question="q1",
            evidence_id=1,
            direction="h1",
            confidence="high",
        )
        assert ev.question == "q1"
        assert ev.direction == "h1"

    def test_q1_allows_h3(self):
        ev = ACHEvidence(
            timestamp=datetime(2026, 3, 10),
            question="q1",
            evidence_id=2,
            direction="h3",
            confidence="medium",
            notes="Escalation signal",
        )
        assert ev.direction == "h3"

    def test_q2_rejects_h3(self):
        with pytest.raises(ValueError):
            ACHEvidence(
                timestamp=datetime(2026, 3, 10),
                question="q2",
                evidence_id=3,
                direction="h3",
                confidence="low",
            )


class TestPathWeights:
    def test_weights_must_sum_to_one(self):
        pw = PathWeights(a=0.30, b=0.50, c=0.20)
        assert pw.a == 0.30
        assert pw.b == 0.50
        assert pw.c == 0.20

    def test_invalid_weights_rejected(self):
        with pytest.raises(ValueError):
            PathWeights(a=0.50, b=0.50, c=0.50)

    def test_apply_delta(self):
        pw = PathWeights(a=0.30, b=0.50, c=0.20)
        new = pw.apply_delta(a_delta=0.05, c_delta=-0.05)
        assert abs(new.a - 0.35) < 1e-9
        assert abs(new.c - 0.15) < 1e-9
        assert abs(new.b - 0.50) < 1e-9
        assert abs(new.a + new.b + new.c - 1.0) < 1e-9

    def test_delta_clamped_to_bounds(self):
        pw = PathWeights(a=0.05, b=0.90, c=0.05)
        new = pw.apply_delta(a_delta=-0.10, c_delta=0.50)
        assert new.a >= 0.0
        assert new.c <= 1.0
        assert abs(new.a + new.b + new.c - 1.0) < 1e-9


class TestSignal:
    def test_grabo_tripwire_has_revert_deadline(self):
        deadline = datetime(2026, 3, 17)
        sig = Signal(
            timestamp=datetime(2026, 3, 10),
            signal_id="T1",
            status=SignalStatus.triggered,
            revert_deadline=deadline,
            action_taken="hedge_50pct",
        )
        assert sig.revert_deadline == deadline
        assert sig.status == SignalStatus.triggered

    def test_event_trigger_no_revert(self):
        sig = Signal(
            timestamp=datetime(2026, 3, 10),
            signal_id="E3",
            status=SignalStatus.confirmed,
            revert_deadline=None,
            action_taken="full_position",
        )
        assert sig.revert_deadline is None


class TestMCParams:
    def test_create_params_with_weights(self):
        pw = PathWeights(a=0.30, b=0.50, c=0.20)
        params = MCParams(
            timestamp=datetime(2026, 3, 10),
            irgc_decay_mean=14.0,
            convoy_start_mean=7.0,
            disruption_range=(0.3, 0.7),
            pipeline_max=1.5,
            pipeline_ramp_weeks=8.0,
            spr_rate_mean=1.0,
            spr_delay_weeks=2.0,
            surplus_buffer=1.5,
            path_weights=pw,
            trigger="weekly_update",
        )
        assert params.path_weights.a == 0.30
        assert params.disruption_range == (0.3, 0.7)
        assert params.trigger == "weekly_update"
