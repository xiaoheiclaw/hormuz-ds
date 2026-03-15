"""M5: Schelling credibility-based game theory path adjuster.

Signals carry evidence strength from LLM extraction.
Adjustment = credibility × evidence × sensitivity, with focal convergence.

Credibility = cost × 0.6 + observability × 0.4 (structural, per signal type).
Focal convergence: N same-direction signals → non-linear amplification.
"""

from __future__ import annotations

from dataclasses import dataclass
from hormuz.core.types import PathWeights

# ── Base sensitivity — global knob for all signal effects ────────────
BASE_SENSITIVITY = 0.15

# ── Focal convergence bonus per additional same-direction signal ─────
FOCAL_BONUS = 0.4


@dataclass(frozen=True)
class SignalEvidence:
    """A signal observation with evidence strength from LLM."""
    key: str
    evidence: float  # 0-1: high=1.0, medium=0.5, low=0.2


@dataclass(frozen=True)
class SignalDef:
    """Structural definition of a Schelling signal."""
    direction: str       # "A" | "B" | "C" — which path this makes focal
    cost: float          # 0-1: how costly to fake
    observability: float # 0-1: how publicly verifiable

    @property
    def credibility(self) -> float:
        return self.cost * 0.6 + self.observability * 0.4


_SIGNAL_DEFS: dict[str, SignalDef] = {
    # Diplomatic talk — low cost, moderately public
    "external_mediation":   SignalDef(direction="A", cost=0.3, observability=0.5),
    # Contradictory US messaging — very cheap, very public
    "us_inconsistency":     SignalDef(direction="B", cost=0.2, observability=0.9),
    # Costly commitment to de-escalation — costly by definition, moderately public
    "costly_self_binding":  SignalDef(direction="A", cost=0.8, observability=0.7),
    # IRGC infrastructure escalation — very costly (irreversible), moderately public
    "irgc_escalation":      SignalDef(direction="C", cost=0.9, observability=0.6),
    # IRGC internal disagreement — moderate cost, hard to verify
    "irgc_fragmentation":   SignalDef(direction="A", cost=0.5, observability=0.4),
}

# Combo requirements: signal requires all listed prerequisites
_COMBO_REQUIRES: dict[str, list[str]] = {
    "irgc_fragmentation": ["us_inconsistency"],
}


def adjust_path_weights(
    base: PathWeights,
    signals: list[SignalEvidence],
) -> PathWeights:
    """Apply Schelling credibility-based adjustment to path weights.

    1. Filter: skip unknown signals, check combo prereqs
    2. Compute effective strength per signal: credibility × evidence × BASE_SENSITIVITY
    3. Group by direction, apply focal convergence (non-linear for same-direction)
    4. Shift target paths up, redistribute from others, normalize + clip
    """
    if not signals:
        return base

    signal_keys = {s.key for s in signals}

    # Filter valid signals with prereq check, deduplicate by key (keep max evidence)
    best_by_key: dict[str, tuple[SignalDef, float]] = {}
    for sig in signals:
        sdef = _SIGNAL_DEFS.get(sig.key)
        if sdef is None:
            continue
        prereqs = _COMBO_REQUIRES.get(sig.key)
        if prereqs and not all(p in signal_keys for p in prereqs):
            continue
        if sig.evidence <= 0:
            continue
        if sig.key not in best_by_key or sig.evidence > best_by_key[sig.key][1]:
            best_by_key[sig.key] = (sdef, sig.evidence)

    active = list(best_by_key.values())
    if not active:
        return base

    # Group strengths by direction
    dir_strengths: dict[str, list[float]] = {"A": [], "B": [], "C": []}
    for sdef, evidence in active:
        strength = sdef.credibility * evidence * BASE_SENSITIVITY
        dir_strengths[sdef.direction].append(strength)

    # Compute total shift per direction with focal convergence
    dir_shift: dict[str, float] = {}
    for d, strengths in dir_strengths.items():
        if not strengths:
            dir_shift[d] = 0.0
            continue
        raw_sum = sum(strengths)
        n = len(strengths)
        focal = 1.0 + FOCAL_BONUS * (n - 1) if n > 1 else 1.0
        dir_shift[d] = raw_sum * focal

    # Compute boost and damp factors from original weights (order-independent)
    boost = {}
    damp = {}
    for d in ("A", "B", "C"):
        shift = dir_shift.get(d, 0.0)
        boost[d] = 1.0 + shift if shift > 0 else 1.0
        damp[d] = max(0.1, 1.0 - shift * 0.3) if shift > 0 else 1.0

    # Apply simultaneously: each path gets its own boost × product of damp from other directions
    weights = {}
    for d in ("A", "B", "C"):
        others = [x for x in ("A", "B", "C") if x != d]
        factor = boost[d]
        for o in others:
            factor *= damp[o]
        weights[d] = max(0.001, getattr(base, d.lower()) * factor)

    return PathWeights(a=weights["A"], b=weights["B"], c=weights["C"]).normalized()
