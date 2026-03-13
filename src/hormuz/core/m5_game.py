"""M5: Game theory path weight adjuster — PRD §6.

Schelling signal deltas applied additively, then normalize + clip.
"""

from __future__ import annotations

from hormuz.core.types import ACHPosterior, PathWeights

# Signal delta table: signal_name -> {a: delta, b: delta, c: delta}
_SIGNAL_DELTAS: dict[str, dict[str, float]] = {
    "mediation":             {"a": +0.15, "b": -0.05, "c": -0.10},
    "commitment_softening":  {"a": +0.10, "b": -0.10, "c":  0.00},
    "commitment_lock":       {"a": +0.05, "b": -0.05, "c":  0.00},
    "escalation":            {"a": -0.10, "b": -0.05, "c": +0.15},
}


def ach_to_base_weights(posterior: ACHPosterior) -> PathWeights:
    """Map ACH posterior to base path weights.

    H1 (depletion) → short crisis → more A, less C
    H2 (preserved) → long crisis → more C, less A
    B absorbs the middle ground.

    At h1=h2=0.5 (no info): A=0.25, B=0.50, C=0.25
    At h1=1.0: A=0.55, B=0.35, C=0.10
    At h2=1.0: A=0.10, B=0.35, C=0.55
    """
    h1 = posterior.h1
    a = 0.25 + 0.30 * (h1 - 0.5)   # [0.10, 0.55]
    c = 0.25 - 0.30 * (h1 - 0.5)   # [0.55, 0.10]
    b = 1.0 - a - c                  # 0.35 at extremes, 0.50 at center
    return PathWeights(a=a, b=b, c=c).normalized()


def adjust_path_weights(
    base: PathWeights,
    active_signals: list[str],
) -> PathWeights:
    """Apply signal deltas sequentially, normalize + clip after each."""
    if not active_signals:
        return base

    current = base
    for signal in active_signals:
        delta = _SIGNAL_DELTAS.get(signal)
        if delta is None:
            continue
        raw = PathWeights(
            a=current.a + delta["a"],
            b=current.b + delta["b"],
            c=current.c + delta["c"],
        )
        current = raw.normalized()

    return current
