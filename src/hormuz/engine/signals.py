"""Grabo tripwire / event trigger / confirmation signal engine.

Signals bypass the normal ACH/MC pipeline and generate position signals immediately.
Three classes:
- T (Tripwire): 48h auto-revert if not confirmed
- E (Event): permanent, no revert
- C (Confirmation): permanent, no revert
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, timedelta
from datetime import datetime
from typing import Callable

from hormuz.db import HormuzDB
from hormuz.models import Observation, PositionSignal, Signal, SignalStatus

# Infrastructure targets that trigger E1
INFRASTRUCTURE_TARGETS = {"pipeline", "refinery", "desalination", "infrastructure"}

# Tripwire revert window
TRIPWIRE_REVERT_HOURS = 48


@dataclass
class SignalDef:
    """Definition of a signal detection rule."""

    signal_id: str
    detect: Callable[[Observation], bool]
    action: str
    has_revert: bool  # True for T-class


# --- Detection functions ---

def _detect_t1(obs: Observation) -> bool:
    return obs.category == "q1_attack" and obs.key == "t1_platform_movement"


def _detect_t2(obs: Observation) -> bool:
    return obs.category == "q1_attack" and obs.key == "t2_coastal_activation"


def _detect_t3(obs: Observation) -> bool:
    return obs.category == "q2_mine" and obs.key == "t3_minelayer_departure"


def _detect_e1(obs: Observation) -> bool:
    return (
        obs.category == "q1_attack"
        and obs.metadata is not None
        and obs.metadata.get("target") in INFRASTRUCTURE_TARGETS
    )


def _detect_e2(obs: Observation) -> bool:
    return (
        obs.category == "q1_attack"
        and obs.metadata is not None
        and obs.metadata.get("target") == "minesweeper"
    )


def _detect_e3(obs: Observation) -> bool:
    return obs.category == "q2_mine" and obs.key == "mine_strike"


def _detect_e4(obs: Observation) -> bool:
    return obs.category == "q2_mine" and obs.key == "new_area_mining"


def _detect_c1(obs: Observation) -> bool:
    return (
        obs.category == "q1_attack"
        and obs.metadata is not None
        and bool(obs.metadata.get("non_iranian_weapon"))
    )


def _detect_c2(obs: Observation) -> bool:
    return obs.category == "q2_mine" and obs.key == "mine_in_cleared_lane"


# Signal definitions in priority order (C > E > T to avoid conflicts)
SIGNAL_DEFS: list[SignalDef] = [
    # Confirmations (highest priority)
    SignalDef("C1", _detect_c1, "ACH→H3确认，持续力无上界", has_revert=False),
    SignalDef("C2", _detect_c2, "convoyStartMean上调3周，H2终极确认", has_revert=False),
    # Event triggers
    SignalDef("E1", _detect_e1, "波动率加倍+衰退对冲5%", has_revert=False),
    SignalDef("E2", _detect_e2, "convoyStartMean上调2周", has_revert=False),
    SignalDef("E3", _detect_e3, "convoyStartMean上调1周", has_revert=False),
    SignalDef("E4", _detect_e4, "Q2时间线延长", has_revert=False),
    # Tripwires (48h auto-revert)
    SignalDef("T1", _detect_t1, "波动率头寸加倍", has_revert=True),
    SignalDef("T2", _detect_t2, "路径C权重大幅上调", has_revert=True),
    SignalDef("T3", _detect_t3, "convoyStartMean上调2周", has_revert=True),
]


class SignalEngine:
    """Scans observations for signal triggers and manages signal lifecycle."""

    def __init__(self, db: HormuzDB) -> None:
        self.db = db

    def _get_active_signal_ids(self) -> set[str]:
        """Return signal_ids of all currently active (triggered/confirmed) signals."""
        active = self.db.get_active_signals()
        return {s.signal_id for s in active}

    def scan(self, observations: list[Observation]) -> list[Signal]:
        """Scan observations for signal triggers.

        For each triggered signal:
        1. Write to signals table
        2. Generate and write position_signal
        3. Return list of triggered signals

        Skips signals already in TRIGGERED/CONFIRMED status (no duplicates).
        """
        active_ids = self._get_active_signal_ids()
        triggered: list[Signal] = []
        # Track newly triggered in this batch to avoid dups within one scan
        triggered_in_batch: set[str] = set()

        for obs in observations:
            for sdef in SIGNAL_DEFS:
                if sdef.signal_id in active_ids or sdef.signal_id in triggered_in_batch:
                    continue
                if not sdef.detect(obs):
                    continue

                now = obs.timestamp
                revert_deadline = (
                    now + timedelta(hours=TRIPWIRE_REVERT_HOURS)
                    if sdef.has_revert
                    else None
                )

                signal = Signal(
                    timestamp=now,
                    signal_id=sdef.signal_id,
                    status=SignalStatus.triggered,
                    revert_deadline=revert_deadline,
                    action_taken=sdef.action,
                )
                sig_id = self.db.insert_signal(signal)
                signal.id = sig_id

                # Generate position signal
                pos = PositionSignal(
                    timestamp=now,
                    trigger=sdef.signal_id,
                    action=sdef.action,
                    executed=False,
                )
                self.db.insert_position_signal(pos)

                triggered.append(signal)
                triggered_in_batch.add(sdef.signal_id)

        return triggered

    def check_reverts(self, now: datetime) -> list[Signal]:
        """Check active signals past their revert_deadline.

        T-class: if now > revert_deadline, set status to REVERTED.
        E-class and C-class: never revert (revert_deadline is None).

        Returns list of reverted signals.
        """
        active = self.db.get_active_signals()
        reverted: list[Signal] = []

        for sig in active:
            if sig.revert_deadline is None:
                continue
            # Compare timezone-aware
            deadline = sig.revert_deadline
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=UTC)
            check_time = now
            if check_time.tzinfo is None:
                check_time = check_time.replace(tzinfo=UTC)

            if check_time > deadline:
                self.db.update_signal_status(sig.id, SignalStatus.reverted)
                sig.status = SignalStatus.reverted
                reverted.append(sig)

        return reverted
