"""Tests for the signals engine (Grabo tripwires, event triggers, confirmations)."""
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hormuz.db import HormuzDB
from hormuz.engine.signals import SignalEngine
from hormuz.models import Observation, Signal, SignalStatus


@pytest.fixture
def signal_engine(tmp_db: Path) -> SignalEngine:
    db = HormuzDB(tmp_db)
    return SignalEngine(db)


def _obs(
    category: str = "q1_attack",
    key: str = "general",
    value: float = 1.0,
    metadata: dict | None = None,
    ts: datetime | None = None,
) -> Observation:
    """Helper to build observations quickly."""
    return Observation(
        timestamp=ts or datetime(2026, 3, 10, 12, 0, tzinfo=UTC),
        source="manual",
        category=category,
        key=key,
        value=value,
        metadata=metadata,
    )


# --- Tripwire Detection ---


class TestTripwireDetection:
    def test_e3_mine_strike_detected(self, signal_engine: SignalEngine):
        """Mine strike -> E3 triggered, no revert deadline."""
        obs = _obs(category="q2_mine", key="mine_strike")
        triggered = signal_engine.scan([obs])

        assert len(triggered) == 1
        sig = triggered[0]
        assert sig.signal_id == "E3"
        assert sig.status == SignalStatus.triggered
        assert sig.revert_deadline is None

    def test_e1_infrastructure_attack(self, signal_engine: SignalEngine):
        """Attack on pipeline -> E1 triggered."""
        obs = _obs(category="q1_attack", key="attack", metadata={"target": "pipeline"})
        triggered = signal_engine.scan([obs])

        assert len(triggered) == 1
        assert triggered[0].signal_id == "E1"
        assert triggered[0].revert_deadline is None

    def test_no_false_positive_on_normal_attack(self, signal_engine: SignalEngine):
        """Normal tanker attack does NOT trigger E1."""
        obs = _obs(category="q1_attack", key="attack", metadata={"target": "tanker"})
        triggered = signal_engine.scan([obs])

        # Should not trigger E1 (tanker not in infrastructure targets)
        e1_signals = [s for s in triggered if s.signal_id == "E1"]
        assert len(e1_signals) == 0

    def test_e2_minesweeper_attack(self, signal_engine: SignalEngine):
        """Attack on minesweeper -> E2 triggered."""
        obs = _obs(category="q1_attack", key="attack", metadata={"target": "minesweeper"})
        triggered = signal_engine.scan([obs])

        assert len(triggered) == 1
        assert triggered[0].signal_id == "E2"
        assert triggered[0].revert_deadline is None

    def test_e4_new_area_mining(self, signal_engine: SignalEngine):
        """New area mining signs -> E4 triggered."""
        obs = _obs(category="q2_mine", key="new_area_mining")
        triggered = signal_engine.scan([obs])

        assert len(triggered) == 1
        assert triggered[0].signal_id == "E4"
        assert triggered[0].revert_deadline is None

    def test_c1_non_iranian_weapon(self, signal_engine: SignalEngine):
        """Non-Iranian weapon -> C1 triggered."""
        obs = _obs(
            category="q1_attack",
            key="weapon_analysis",
            metadata={"non_iranian_weapon": True},
        )
        triggered = signal_engine.scan([obs])

        assert len(triggered) == 1
        assert triggered[0].signal_id == "C1"
        assert triggered[0].revert_deadline is None

    def test_c2_mine_in_cleared_lane(self, signal_engine: SignalEngine):
        """Mine in cleared lane -> C2 triggered."""
        obs = _obs(category="q2_mine", key="mine_in_cleared_lane")
        triggered = signal_engine.scan([obs])

        assert len(triggered) == 1
        assert triggered[0].signal_id == "C2"
        assert triggered[0].revert_deadline is None

    def test_t1_platform_movement(self, signal_engine: SignalEngine):
        """Missile platform movement -> T1 triggered with 48h revert."""
        obs = _obs(category="q1_attack", key="t1_platform_movement")
        triggered = signal_engine.scan([obs])

        assert len(triggered) == 1
        sig = triggered[0]
        assert sig.signal_id == "T1"
        assert sig.status == SignalStatus.triggered
        assert sig.revert_deadline is not None
        # Revert deadline should be ~48h after trigger
        delta = sig.revert_deadline - sig.timestamp
        assert abs(delta.total_seconds() - 48 * 3600) < 1

    def test_t2_coastal_positions(self, signal_engine: SignalEngine):
        """Multi-region coastal positions -> T2 triggered."""
        obs = _obs(category="q1_attack", key="t2_multi_region")
        triggered = signal_engine.scan([obs])

        assert len(triggered) == 1
        assert triggered[0].signal_id == "T2"
        assert triggered[0].revert_deadline is not None

    def test_t3_minelayer_departure(self, signal_engine: SignalEngine):
        """Multiple minelayers depart -> T3 triggered."""
        obs = _obs(category="q2_mine", key="t3_mining_boats")
        triggered = signal_engine.scan([obs])

        assert len(triggered) == 1
        assert triggered[0].signal_id == "T3"
        assert triggered[0].revert_deadline is not None

    def test_no_duplicate_trigger(self, signal_engine: SignalEngine):
        """Same signal doesn't trigger twice if already active."""
        obs = _obs(category="q2_mine", key="mine_strike")

        first = signal_engine.scan([obs])
        assert len(first) == 1

        # Scan again with same type of observation
        obs2 = _obs(
            category="q2_mine",
            key="mine_strike",
            ts=datetime(2026, 3, 11, 12, 0, tzinfo=UTC),
        )
        second = signal_engine.scan([obs2])
        assert len([s for s in second if s.signal_id == "E3"]) == 0

    def test_multiple_signals_from_batch(self, signal_engine: SignalEngine):
        """A batch of observations can trigger multiple different signals."""
        obs_list = [
            _obs(category="q2_mine", key="mine_strike"),
            _obs(category="q1_attack", key="attack", metadata={"target": "refinery"}),
        ]
        triggered = signal_engine.scan(obs_list)

        ids = {s.signal_id for s in triggered}
        assert "E3" in ids
        assert "E1" in ids


# --- Revert ---


class TestRevert:
    def test_tripwire_reverts_after_48h(self, signal_engine: SignalEngine):
        """T-class reverts after 48h with no confirmation."""
        obs = _obs(category="q1_attack", key="t1_platform_movement")
        signal_engine.scan([obs])

        # Before 48h: no reverts
        now_before = datetime(2026, 3, 10, 12, 0, tzinfo=UTC) + timedelta(hours=47)
        reverted = signal_engine.check_reverts(now_before)
        assert len(reverted) == 0

        # After 48h: reverted
        now_after = datetime(2026, 3, 10, 12, 0, tzinfo=UTC) + timedelta(hours=49)
        reverted = signal_engine.check_reverts(now_after)
        assert len(reverted) == 1
        assert reverted[0].signal_id == "T1"
        assert reverted[0].status == SignalStatus.reverted

    def test_event_trigger_never_reverts(self, signal_engine: SignalEngine):
        """E-class never reverts even after long time."""
        obs = _obs(category="q2_mine", key="mine_strike")
        signal_engine.scan([obs])

        far_future = datetime(2027, 1, 1, tzinfo=UTC)
        reverted = signal_engine.check_reverts(far_future)
        assert len(reverted) == 0

    def test_confirmation_never_reverts(self, signal_engine: SignalEngine):
        """C-class never reverts even after long time."""
        obs = _obs(
            category="q1_attack",
            key="weapon_analysis",
            metadata={"non_iranian_weapon": True},
        )
        signal_engine.scan([obs])

        far_future = datetime(2027, 1, 1, tzinfo=UTC)
        reverted = signal_engine.check_reverts(far_future)
        assert len(reverted) == 0


# --- Position Signal Generation ---


class TestPositionSignalGeneration:
    def test_e3_generates_position_signal(self, signal_engine: SignalEngine):
        """Mine strike -> position signal with convoy action."""
        obs = _obs(category="q2_mine", key="mine_strike")
        signal_engine.scan([obs])

        pos_signals = signal_engine.db.get_unexecuted_position_signals()
        assert len(pos_signals) == 1
        assert "convoy" in pos_signals[0].action.lower() or "上调1周" in pos_signals[0].action
        assert pos_signals[0].executed is False

    def test_e1_generates_volatility_signal(self, signal_engine: SignalEngine):
        """Infrastructure attack -> position signal with vol+recession action."""
        obs = _obs(category="q1_attack", key="attack", metadata={"target": "pipeline"})
        signal_engine.scan([obs])

        pos_signals = signal_engine.db.get_unexecuted_position_signals()
        assert len(pos_signals) == 1
        ps = pos_signals[0]
        assert "波动率" in ps.action or "vol" in ps.action.lower()
        assert ps.executed is False

    def test_t1_generates_position_signal(self, signal_engine: SignalEngine):
        """T1 tripwire -> position signal for vol doubling."""
        obs = _obs(category="q1_attack", key="t1_platform_movement")
        signal_engine.scan([obs])

        pos_signals = signal_engine.db.get_unexecuted_position_signals()
        assert len(pos_signals) == 1
        assert pos_signals[0].executed is False

    def test_c1_generates_position_signal(self, signal_engine: SignalEngine):
        """C1 confirmation -> position signal for ACH->H3."""
        obs = _obs(
            category="q1_attack",
            key="weapon_analysis",
            metadata={"non_iranian_weapon": True},
        )
        signal_engine.scan([obs])

        pos_signals = signal_engine.db.get_unexecuted_position_signals()
        assert len(pos_signals) == 1
        assert "H3" in pos_signals[0].action or "持续力" in pos_signals[0].action

    def test_signals_persisted_to_db(self, signal_engine: SignalEngine):
        """Triggered signals are persisted to the signals table."""
        obs = _obs(category="q2_mine", key="mine_strike")
        signal_engine.scan([obs])

        active = signal_engine.db.get_active_signals()
        assert len(active) == 1
        assert active[0].signal_id == "E3"
