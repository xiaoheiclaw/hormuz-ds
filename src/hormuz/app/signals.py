"""Grabo tripwire signal system — T1-T3, E1-E4, C1-C2.

Signals bypass normal ACH pipeline and map directly to position actions.
T1a/T1b/T2/T3/E4 auto-revert after 48h; E1-E3/C1-C2 are persistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from hormuz.core.types import Observation

# ── Signal → position action mapping ─────────────────────────────────

_SIGNAL_ACTIONS: dict[str, list[dict]] = {
    "T1a": [{"action": "vol_double", "desc": "T1a offensive H2 — vol×2"}],
    "T1b": [{"action": "vol_add_50pct", "desc": "T1b defensive H2 — vol×1.5"}],
    "T2":  [{"action": "vol_double", "desc": "T2 multi-region — vol×2"}],
    "T3":  [{"action": "energy_add_5", "desc": "T3 mining boats — energy+5%"},
            {"action": "vol_double", "desc": "T3 mining boats — vol×2"}],
    "E1":  [{"action": "vol_double", "desc": "E1 infrastructure attack — vol×2"},
            {"action": "recession_5", "desc": "E1 infrastructure attack — recession 5%"}],
    "E2":  [{"action": "energy_add_5", "desc": "E2 minesweeper attack — energy+5%"}],
    "E3":  [{"action": "energy_add_5", "desc": "E3 mine strike — energy+5%"}],
    "E4":  [{"action": "vol_add_50pct", "desc": "E4 new area mining — vol×1.5"}],
    "C1":  [{"action": "escalation", "desc": "C1 non-Iranian weapons — escalation signal"}],
    "C2":  [{"action": "energy_add_5", "desc": "C2 re-mining — energy+5%"},
            {"action": "vol_double", "desc": "C2 re-mining — vol×2"}],
}

# Signals with 48h auto-revert
_REVERTABLE = {"T1a", "T1b", "T2", "T3", "E4"}
_REVERT_HOURS = 48


@dataclass
class SignalResult:
    triggered: list[str] = field(default_factory=list)
    reverted: list[str] = field(default_factory=list)
    position_actions: list[dict] = field(default_factory=list)
    events: dict = field(default_factory=dict)


def scan_signals(
    observations: list[Observation],
    signal_state: dict,
    o01_trend: str | None = None,
    events: dict[str, bool] | None = None,
) -> SignalResult:
    """Scan observations and events for tripwire signals.

    Returns triggered signals and their mapped position actions.
    """
    events = events or {}
    triggered: list[str] = []
    actions: list[dict] = []

    # Index observations by id
    obs_by_id: dict[str, Observation] = {}
    for o in observations:
        obs_by_id[o.id] = o

    # ── T1a / T1b: GPS spoofing + attack frequency co-occurrence ──
    o05 = obs_by_id.get("O05")
    if o05 and o05.value > 0.5:
        if o01_trend == "rising":
            triggered.append("T1a")
        elif o01_trend == "falling":
            triggered.append("T1b")

    # ── T2: Multi-region activation (simplified: O06 mosaic high) ──
    o06 = obs_by_id.get("O06")
    if o06 and o06.value > 0.7:
        triggered.append("T2")

    # ── T3: Mining boats (would come from specific intel, check events) ──
    if events.get("T3"):
        triggered.append("T3")

    # ── E1-E4: Event-based triggers ──
    for eid in ["E1", "E2", "E3", "E4"]:
        if events.get(eid):
            triggered.append(eid)

    # ── C1-C2: Confirmations ──
    for cid in ["C1", "C2"]:
        if events.get(cid):
            triggered.append(cid)

    # Collect position actions
    for sig in triggered:
        actions.extend(_SIGNAL_ACTIONS.get(sig, []))

    return SignalResult(
        triggered=triggered,
        reverted=[],
        position_actions=actions,
        events=events,
    )


def check_reverts(
    signal_state: dict[str, dict],
    now: datetime | None = None,
) -> list[str]:
    """Check which signals should auto-revert (48h expiry).

    signal_state: {signal_id: {"triggered_at": datetime}}
    Returns list of signal IDs that should revert.
    """
    now = now or datetime.now()
    reverted = []

    for sig_id, state in signal_state.items():
        if sig_id not in _REVERTABLE:
            continue
        triggered_at = state.get("triggered_at")
        if triggered_at and (now - triggered_at) > timedelta(hours=_REVERT_HOURS):
            reverted.append(sig_id)

    return reverted
