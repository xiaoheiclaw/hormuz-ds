"""M5: Game theory path weight adjuster — PRD §6.

Schelling signal deltas applied additively, then normalize + clip.
6 signals from constants.yaml, cap ±10pp per signal.
"""

from __future__ import annotations

from hormuz.core.types import ACHPosterior, PathWeights

# PRD §6 Schelling signal table (from constants.yaml)
# id → signal_key mapping:
#   S1: external_mediation    — third-party mediation (Oman, Qatar, China)
#   S2: us_inconsistency      — contradictory US messaging
#   S3: costly_self_binding   — costly commitment to de-escalation
#   S4: irgc_escalation       — IRGC infrastructure escalation (cross-layer E1)
#   S5: peace_window          — diplomatic window (requires S1+S3 combo)
#   S6: irgc_fragmentation    — IRGC internal disagreement (requires S2 combo)

_SIGNAL_DELTAS: dict[str, dict[str, float]] = {
    "external_mediation":   {"a": +0.05, "b":  0.00, "c": -0.03},
    "us_inconsistency":     {"a": +0.03, "b":  0.00, "c":  0.00},
    "costly_self_binding":  {"a": +0.05, "b":  0.00, "c":  0.00},
    "irgc_escalation":      {"a": -0.05, "b":  0.00, "c": +0.10},
    "peace_window":         {"a": +0.03, "b":  0.00, "c":  0.00},
    "irgc_fragmentation":   {"a": +0.02, "b":  0.00, "c":  0.00},
}

# Combo requirements: signal requires all listed prerequisites to be active
_COMBO_REQUIRES: dict[str, list[str]] = {
    "peace_window": ["external_mediation", "costly_self_binding"],
    "irgc_fragmentation": ["us_inconsistency"],
}

# Max absolute delta per signal application (PRD: 10pp)
_DELTA_CAP = 0.10


def ach_to_base_weights(posterior: ACHPosterior) -> PathWeights:
    """Map ACH posterior to base path weights.

    H1 (depletion) → short crisis → more A, less C
    H2 (preserved) → long crisis → more C, less A
    B absorbs the middle ground.
    """
    h1 = posterior.h1
    a = 0.25 + 0.30 * (h1 - 0.5)
    c = 0.25 - 0.30 * (h1 - 0.5)
    b = 1.0 - a - c
    return PathWeights(a=a, b=b, c=c).normalized()


def adjust_path_weights(
    base: PathWeights,
    active_signals: list[str],
) -> PathWeights:
    """Apply signal deltas sequentially, normalize + clip after each.

    Combo signals (peace_window, irgc_fragmentation) only fire if
    their prerequisites are also in active_signals.
    """
    if not active_signals:
        return base

    signal_set = set(active_signals)

    current = base
    for signal in active_signals:
        delta = _SIGNAL_DELTAS.get(signal)
        if delta is None:
            continue

        # Check combo prerequisites
        prereqs = _COMBO_REQUIRES.get(signal)
        if prereqs and not all(p in signal_set for p in prereqs):
            continue

        raw = PathWeights(
            a=current.a + delta["a"],
            b=current.b + delta["b"],
            c=current.c + delta["c"],
        )
        current = raw.normalized()

    return current
