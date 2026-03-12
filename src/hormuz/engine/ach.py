"""ACH (Analysis of Competing Hypotheses) Matrix Engine.

v5.4 changes:
- H3 suspension: Mehrabad airport destroyed, Russian supply cut.
  When H3 suspended, its prior (10%) is redistributed +5% to H1, +5% to H2.
  Posterior updates run binary Bayesian between H1/H2 only.
- Q1 evidence updated to non-visual high-frequency version (5 items).
"""
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from hormuz.db import HormuzDB
from hormuz.models import ACHEvidence, RegimeType

# Staleness threshold: evidence older than this without newer confirmation is excluded.
_STALENESS_WEEKS = 2


class ACHEngine:
    """ACH matrix evaluation engine.

    Takes evidence from the database, applies staleness decay and convergence
    rules, and returns regime judgments.

    v5.4: H3 suspension support — when h3_suspended=True, H3 evidence is
    excluded and regime evaluation only considers H1/H2 directions.
    """

    def __init__(self, db: HormuzDB, ach_config: dict, h3_suspended: bool = True) -> None:
        """Load evidence definitions from ach_config (from constants.yaml ach section).

        Builds a lookup: {question: {evidence_id(int): discriminating_power}}
        Evidence IDs are 1-based integers derived from the list index.
        """
        self.db = db
        self.ach_config = ach_config
        self.h3_suspended = h3_suspended
        # Build disc power lookup: question -> evidence_id -> power
        self._disc_power: dict[str, dict[int, str]] = {}
        for question in ("q1", "q2"):
            q_config = ach_config.get(question, {})
            power_map: dict[int, str] = {}
            for i, ev_def in enumerate(q_config.get("evidence", []), start=1):
                power_map[i] = ev_def["discriminating_power"]
            self._disc_power[question] = power_map

    def add_evidence(self, evidence: ACHEvidence) -> int:
        """Write evidence to database. Returns evidence row id."""
        return self.db.insert_ach_evidence(evidence)

    def evaluate_regime(
        self, question: str, as_of: datetime | None = None
    ) -> RegimeType:
        """Evaluate current regime based on accumulated evidence.

        Algorithm:
        1. Get all evidence for this question from DB
        2. Apply staleness decay: for each evidence_id, keep only the most recent
           entry; if that entry is >2 weeks old relative to as_of, exclude it
        3. Separate remaining evidence by discriminating power
        4. Count high-disc evidence by direction
        5. Apply convergence rules:
           a. If >=3 high-disc evidence point same direction -> lean toward that hypothesis
           b. If the would-be regime is non-WIDE and any high-disc evidence points
              in a DIFFERENT direction -> revert to WIDE
           c. If only medium/low evidence -> stay WIDE
        """
        if as_of is None:
            as_of = datetime.now(UTC)

        # Ensure as_of is timezone-aware
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)

        # 1. Get all evidence for this question
        all_evidence = self.db.get_ach_evidence(question)
        if not all_evidence:
            return RegimeType.wide

        # 2. Staleness: for each evidence_id, keep only the most recent entry
        latest_by_eid: dict[int, ACHEvidence] = {}
        for ev in all_evidence:
            eid = ev.evidence_id
            if eid not in latest_by_eid or ev.timestamp > latest_by_eid[eid].timestamp:
                latest_by_eid[eid] = ev

        # Filter out stale evidence (>2 weeks old relative to as_of)
        cutoff = as_of - timedelta(weeks=_STALENESS_WEEKS)
        active_evidence: list[ACHEvidence] = []
        for ev in latest_by_eid.values():
            ev_ts = ev.timestamp
            if ev_ts.tzinfo is None:
                ev_ts = ev_ts.replace(tzinfo=UTC)
            if ev_ts >= cutoff:
                active_evidence.append(ev)

        if not active_evidence:
            return RegimeType.wide

        # 3. Separate by discriminating power
        power_map = self._disc_power.get(question, {})
        high_disc: list[ACHEvidence] = []
        for ev in active_evidence:
            power = power_map.get(ev.evidence_id, "low")
            if power == "high":
                high_disc.append(ev)

        # 5c. If no high-disc evidence -> stay WIDE
        if not high_disc:
            return RegimeType.wide

        # 4. Count high-disc evidence by direction (excluding neutral)
        # v5.4: when H3 suspended for Q1, exclude h3 direction evidence
        excluded_directions = {"neutral"}
        if self.h3_suspended and question == "q1":
            excluded_directions.add("h3")

        direction_counts: dict[str, int] = defaultdict(int)
        for ev in high_disc:
            if ev.direction not in excluded_directions:
                direction_counts[ev.direction] += 1

        if not direction_counts:
            return RegimeType.wide

        # 5a. Find the dominant direction
        max_count = max(direction_counts.values())
        dominant_dirs = [d for d, c in direction_counts.items() if c == max_count]

        # Need >=3 high-disc in the same direction
        if max_count < 3:
            return RegimeType.wide

        # Pick the dominant direction (should be unique if >=3)
        dominant = dominant_dirs[0]

        # 5b. Check for contrary evidence: any high-disc pointing a different direction
        other_directions = {d for d in direction_counts if d != dominant}
        if other_directions:
            return RegimeType.wide

        # Map direction to regime
        return _direction_to_regime(dominant, question)


def _direction_to_regime(direction: str, question: str) -> RegimeType:
    """Map a direction string to the corresponding RegimeType."""
    if direction == "h1":
        return RegimeType.lean_h1
    elif direction == "h2":
        return RegimeType.lean_h2
    elif direction == "h3" and question == "q1":
        return RegimeType.confirmed_h3
    else:
        return RegimeType.wide
