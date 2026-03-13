"""Position rules engine — base positions + exit rules.

executed field is the human confirmation boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hormuz.core.types import SystemOutput


@dataclass
class PositionResult:
    energy_pct: int
    vol_pct: int
    recession_pct: int
    actions: list[str] = field(default_factory=list)
    executed: bool = False  # human confirmation boundary


# Base positions from PRD
_BASE_ENERGY = 15
_BASE_VOL = 3
_BASE_RECESSION = 2


def evaluate_positions(
    system_output: SystemOutput,
    brent_price: float,
    t_end_confirmed: bool = False,
    brent_below_80_days: int = 0,
) -> PositionResult:
    """Evaluate position recommendations.

    1. Start with base positions (15/3/2)
    2. Check exit rules: T end / $150 / $80
    """
    energy = _BASE_ENERGY
    vol = _BASE_VOL
    recession = _BASE_RECESSION
    actions: list[str] = []

    # ── Exit rules (override base) ──────────────────────────────────

    # System failure: Brent < $80 for 3+ days
    if brent_below_80_days >= 3:
        energy = 0
        vol = 0
        recession = 0
        actions = ["FORCE CLOSE ALL — Brent < $80 × 3 days (system failure)"]
        return PositionResult(energy_pct=energy, vol_pct=vol, recession_pct=recession, actions=actions)

    # Demand destruction: Brent > $150
    if brent_price > 150:
        energy = 0
        recession = _BASE_RECESSION * 2  # double recession hedge
        actions.append("CLEAR energy — Brent > $150 (demand destruction)")
        actions.append(f"recession→{recession}% (doubled)")
        return PositionResult(energy_pct=energy, vol_pct=vol, recession_pct=recession, actions=actions)

    # T-end confirmed: unwind
    if t_end_confirmed:
        energy = max(0, energy - 10)  # reduce to ~5%
        vol = 0  # close vol
        actions.append("T-end confirmed — reducing energy, closing vol")

    return PositionResult(
        energy_pct=energy,
        vol_pct=vol,
        recession_pct=recession,
        actions=actions,
    )
