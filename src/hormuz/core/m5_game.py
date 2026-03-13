"""M5: Game theory path weight adjuster — PRD §6.

Schelling signal deltas applied additively, then normalize + clip.
"""

from __future__ import annotations

from hormuz.core.types import PathWeights

# Signal delta table: signal_name -> {a: delta, b: delta, c: delta}
_SIGNAL_DELTAS: dict[str, dict[str, float]] = {
    "mediation":             {"a": +0.15, "b": -0.05, "c": -0.10},
    "commitment_softening":  {"a": +0.10, "b": -0.10, "c":  0.00},
    "commitment_lock":       {"a": +0.05, "b": -0.05, "c":  0.00},
    "escalation":            {"a": -0.10, "b": -0.05, "c": +0.15},
}


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
