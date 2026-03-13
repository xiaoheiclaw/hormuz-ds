"""M5: Game theory path weight adjuster — PRD §6.

Schelling signal multipliers applied sequentially, then normalize + clip.
Multiplicative logic: consistent relative impact regardless of base weight.
"""

from __future__ import annotations

from hormuz.core.types import ACHPosterior, PathWeights

# PRD §6 Schelling signal table
# Multipliers: >1 = more likely, <1 = less likely
# e.g., a=1.3 means "A path becomes 30% more likely"
#
# Design principles:
#   - Multiplicative: same signal = same relative impact at any base
#   - Combo signals stronger than prerequisites (higher evidence bar → bigger payoff)
#   - Escalation signals have larger multipliers than de-escalation (bad news > good news)
#   - B is explicitly targeted, not just residual

_SIGNAL_MULTIPLIERS: dict[str, dict[str, float]] = {
    # S1: third-party mediation (Oman, Qatar, China)
    "external_mediation":   {"a": 1.25, "b": 1.05, "c": 0.80},
    # S2: contradictory US messaging — uncertainty increases B and C
    "us_inconsistency":     {"a": 1.05, "b": 1.10, "c": 1.05},
    # S3: costly commitment to de-escalation
    "costly_self_binding":  {"a": 1.30, "b": 1.00, "c": 0.80},
    # S4: IRGC infrastructure escalation — strongest signal
    "irgc_escalation":      {"a": 0.70, "b": 0.85, "c": 1.60},
    # S5: diplomatic window (combo: mediation + self_binding)
    "peace_window":         {"a": 1.50, "b": 0.90, "c": 0.60},
    # S6: IRGC internal disagreement (combo: us_inconsistency)
    "irgc_fragmentation":   {"a": 1.30, "b": 1.10, "c": 0.75},
}

# Combo requirements: signal requires all listed prerequisites to be active
_COMBO_REQUIRES: dict[str, list[str]] = {
    "peace_window": ["external_mediation", "costly_self_binding"],
    "irgc_fragmentation": ["us_inconsistency"],
}


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
    """Apply signal multipliers sequentially, normalize + clip after each.

    Multiplicative: weight *= multiplier, then renormalize.
    Combo signals only fire if prerequisites are also in active_signals.
    """
    if not active_signals:
        return base

    signal_set = set(active_signals)

    current = base
    for signal in active_signals:
        mults = _SIGNAL_MULTIPLIERS.get(signal)
        if mults is None:
            continue

        # Check combo prerequisites
        prereqs = _COMBO_REQUIRES.get(signal)
        if prereqs and not all(p in signal_set for p in prereqs):
            continue

        raw = PathWeights(
            a=current.a * mults["a"],
            b=current.b * mults["b"],
            c=current.c * mults["c"],
        )
        current = raw.normalized()

    return current
