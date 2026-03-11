"""Tests for ACH Matrix Engine."""
from datetime import UTC, datetime, timedelta

import pytest

from hormuz.db import HormuzDB
from hormuz.engine.ach import ACHEngine
from hormuz.models import ACHEvidence, RegimeType


@pytest.fixture
def ach_engine(tmp_db, constants) -> ACHEngine:
    db = HormuzDB(tmp_db)
    return ACHEngine(db, constants["ach"])


def _ev(question: str, evidence_id: int, direction: str, ts: datetime | None = None) -> ACHEvidence:
    """Helper to create ACHEvidence with sensible defaults."""
    return ACHEvidence(
        timestamp=ts or datetime.now(UTC),
        question=question,
        evidence_id=evidence_id,
        direction=direction,
        confidence="high",
        notes="test",
    )


class TestAddEvidence:
    def test_add_returns_row_id(self, ach_engine):
        ev = _ev("q1", 1, "h1")
        row_id = ach_engine.add_evidence(ev)
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_add_multiple_returns_incrementing_ids(self, ach_engine):
        id1 = ach_engine.add_evidence(_ev("q1", 1, "h1"))
        id2 = ach_engine.add_evidence(_ev("q1", 2, "h1"))
        assert id2 > id1


class TestConvergenceRules:
    def test_three_high_disc_same_direction_updates_regime(self, ach_engine):
        """>=3 high-disc evidence all pointing h1 -> LEAN_H1."""
        # Q1 high-disc evidence_ids: 1 (q1_e1), 2 (q1_e2), 3 (q1_e3), 8 (q1_e8)
        ach_engine.add_evidence(_ev("q1", 1, "h1"))
        ach_engine.add_evidence(_ev("q1", 2, "h1"))
        ach_engine.add_evidence(_ev("q1", 3, "h1"))
        assert ach_engine.evaluate_regime("q1") == RegimeType.lean_h1

    def test_three_high_disc_h3_gives_confirmed(self, ach_engine):
        """>=3 high-disc pointing h3 -> CONFIRMED_H3 (Q1 only)."""
        ach_engine.add_evidence(_ev("q1", 1, "h3"))
        ach_engine.add_evidence(_ev("q1", 2, "h3"))
        ach_engine.add_evidence(_ev("q1", 3, "h3"))
        assert ach_engine.evaluate_regime("q1") == RegimeType.confirmed_h3

    def test_three_high_disc_h2_gives_lean_h2(self, ach_engine):
        """>=3 high-disc pointing h2 -> LEAN_H2."""
        ach_engine.add_evidence(_ev("q1", 1, "h2"))
        ach_engine.add_evidence(_ev("q1", 2, "h2"))
        ach_engine.add_evidence(_ev("q1", 8, "h2"))
        assert ach_engine.evaluate_regime("q1") == RegimeType.lean_h2

    def test_only_medium_low_does_not_update(self, ach_engine):
        """Even many medium/low evidence -> stays WIDE."""
        # Q1 medium: 4,5,6,7; low: 9
        for eid in [4, 5, 6, 7, 9]:
            ach_engine.add_evidence(_ev("q1", eid, "h1"))
        assert ach_engine.evaluate_regime("q1") == RegimeType.wide

    def test_single_contrary_high_disc_reverts(self, ach_engine):
        """After establishing LEAN_H1, one contrary high-disc -> back to WIDE."""
        # Establish LEAN_H1 with 3 high-disc h1
        ach_engine.add_evidence(_ev("q1", 1, "h1"))
        ach_engine.add_evidence(_ev("q1", 2, "h1"))
        ach_engine.add_evidence(_ev("q1", 3, "h1"))
        assert ach_engine.evaluate_regime("q1") == RegimeType.lean_h1

        # Add 1 high-disc evidence pointing h3
        ach_engine.add_evidence(_ev("q1", 8, "h3"))
        assert ach_engine.evaluate_regime("q1") == RegimeType.wide

    def test_neutral_does_not_contribute(self, ach_engine):
        """Neutral high-disc evidence doesn't count toward any direction."""
        ach_engine.add_evidence(_ev("q1", 1, "neutral"))
        ach_engine.add_evidence(_ev("q1", 2, "neutral"))
        ach_engine.add_evidence(_ev("q1", 3, "neutral"))
        assert ach_engine.evaluate_regime("q1") == RegimeType.wide

    def test_no_evidence_returns_wide(self, ach_engine):
        """No evidence at all -> WIDE."""
        assert ach_engine.evaluate_regime("q1") == RegimeType.wide

    def test_mixed_high_disc_below_threshold(self, ach_engine):
        """2 high-disc h1 + 1 high-disc h2 -> not enough in any direction -> WIDE."""
        ach_engine.add_evidence(_ev("q1", 1, "h1"))
        ach_engine.add_evidence(_ev("q1", 2, "h1"))
        ach_engine.add_evidence(_ev("q1", 3, "h2"))
        assert ach_engine.evaluate_regime("q1") == RegimeType.wide


class TestStaleness:
    def test_stale_evidence_decays(self, ach_engine):
        """Evidence >2 weeks old without refresh -> excluded."""
        old = datetime.now(UTC) - timedelta(weeks=3)
        ach_engine.add_evidence(_ev("q1", 1, "h1", ts=old))
        ach_engine.add_evidence(_ev("q1", 2, "h1", ts=old))
        ach_engine.add_evidence(_ev("q1", 3, "h1", ts=old))
        assert ach_engine.evaluate_regime("q1", as_of=datetime.now(UTC)) == RegimeType.wide

    def test_fresh_evidence_overrides_stale(self, ach_engine):
        """Fresh evidence for same evidence_id keeps it active."""
        old = datetime.now(UTC) - timedelta(weeks=3)
        fresh = datetime.now(UTC) - timedelta(days=1)

        # evidence_id=1: stale then fresh
        ach_engine.add_evidence(_ev("q1", 1, "h1", ts=old))
        ach_engine.add_evidence(_ev("q1", 1, "h1", ts=fresh))

        # evidence_id=2 and 3: fresh
        ach_engine.add_evidence(_ev("q1", 2, "h1", ts=fresh))
        ach_engine.add_evidence(_ev("q1", 3, "h1", ts=fresh))

        # Fresh entries should be used -> LEAN_H1
        assert ach_engine.evaluate_regime("q1", as_of=datetime.now(UTC)) == RegimeType.lean_h1

    def test_stale_evidence_with_as_of_none_uses_now(self, ach_engine):
        """as_of=None defaults to now for staleness check."""
        old = datetime.now(UTC) - timedelta(weeks=3)
        ach_engine.add_evidence(_ev("q1", 1, "h1", ts=old))
        ach_engine.add_evidence(_ev("q1", 2, "h1", ts=old))
        ach_engine.add_evidence(_ev("q1", 3, "h1", ts=old))
        # as_of=None -> uses now -> all stale
        assert ach_engine.evaluate_regime("q1") == RegimeType.wide

    def test_evidence_exactly_two_weeks_not_stale(self, ach_engine):
        """Evidence exactly at the 2-week boundary is not stale."""
        now = datetime.now(UTC)
        boundary = now - timedelta(weeks=2)
        ach_engine.add_evidence(_ev("q1", 1, "h1", ts=boundary))
        ach_engine.add_evidence(_ev("q1", 2, "h1", ts=boundary))
        ach_engine.add_evidence(_ev("q1", 3, "h1", ts=boundary))
        assert ach_engine.evaluate_regime("q1", as_of=now) == RegimeType.lean_h1


class TestQ2ACH:
    def test_q2_only_two_hypotheses(self, ach_engine):
        """Q2 has h1/h2 only. Single high-disc doesn't reach >=3 threshold."""
        # Q2 high-disc: 1 (q2_e1), 2 (q2_e2), 3 (q2_e3), 4 (q2_e4)
        ach_engine.add_evidence(_ev("q2", 1, "h1"))
        assert ach_engine.evaluate_regime("q2") == RegimeType.wide

    def test_q2_convergence_with_three_high_disc(self, ach_engine):
        """Q2 with >=3 high-disc same direction -> converges."""
        ach_engine.add_evidence(_ev("q2", 1, "h1"))
        ach_engine.add_evidence(_ev("q2", 2, "h1"))
        ach_engine.add_evidence(_ev("q2", 3, "h1"))
        assert ach_engine.evaluate_regime("q2") == RegimeType.lean_h1

    def test_q2_medium_only_stays_wide(self, ach_engine):
        """Q2 with only medium/low evidence stays WIDE."""
        for eid in [5, 6, 7]:
            ach_engine.add_evidence(_ev("q2", eid, "h1"))
        assert ach_engine.evaluate_regime("q2") == RegimeType.wide

    def test_q2_contrary_reverts(self, ach_engine):
        """Q2 contrary high-disc reverts to WIDE."""
        ach_engine.add_evidence(_ev("q2", 1, "h1"))
        ach_engine.add_evidence(_ev("q2", 2, "h1"))
        ach_engine.add_evidence(_ev("q2", 3, "h1"))
        assert ach_engine.evaluate_regime("q2") == RegimeType.lean_h1

        ach_engine.add_evidence(_ev("q2", 4, "h2"))
        assert ach_engine.evaluate_regime("q2") == RegimeType.wide
