"""Schelling Signal Sheet — game theory delta adjustments to path weights.

Six fixed checks that output delta adjustments to path A (accommodation)
and path C (confrontation) weights. Path B is residual, not returned.

Class A signals (hard observable) trigger independently.
Class B signals (pattern inference) require combo conditions.
"""
from __future__ import annotations


class SchellingSheet:
    """Compute path-weight deltas from active Schelling signals."""

    def __init__(self, config: dict) -> None:
        """Load signal definitions from constants.yaml["schelling"].

        Args:
            config: The schelling section of constants.yaml.
        """
        self.min_week: int = config.get("min_week", 4)
        self.delta_cap: float = config.get("delta_cap", 0.10)

        # Build lookup: signal_id -> signal definition
        self._signals: dict[int, dict] = {}
        for sig in config["signals"]:
            self._signals[sig["id"]] = sig

    def compute_delta(
        self,
        active_signals: dict[int, bool],
        current_week: int | None = None,
    ) -> dict[str, float]:
        """Compute path weight delta adjustments.

        Rules:
        1. Before min_week: return {a: 0, c: 0} — only record baseline
        2. Class A signals trigger independently
        3. Class B signals only trigger if their combo condition is met
        4. Each triggered signal contributes its fixed delta
        5. Total delta capped at +/-delta_cap per call

        Args:
            active_signals: {signal_id: is_active} mapping.
            current_week: Current crisis week. None = auto-detect (>= min_week).

        Returns:
            {"a": float, "c": float} — delta adjustments.
        """
        zero = {"a": 0.0, "c": 0.0}

        # Week gating: before min_week, no delta output
        if current_week is not None and current_week < self.min_week:
            return zero

        # Collect active signal ids
        active_ids = {sid for sid, active in active_signals.items() if active}
        if not active_ids:
            return zero

        total_a = 0.0
        total_c = 0.0

        for sid in active_ids:
            sig_def = self._signals.get(sid)
            if sig_def is None:
                continue

            # Class B: check combo condition
            combo = sig_def.get("requires_combo")
            if combo is not None:
                # At least one of the combo signals must be active
                if not any(req_id in active_ids for req_id in combo):
                    continue

            # Accumulate deltas
            delta = sig_def.get("delta", {})
            total_a += delta.get("a", 0.0)
            total_c += delta.get("c", 0.0)

        # Cap at +/-delta_cap
        total_a = max(-self.delta_cap, min(self.delta_cap, total_a))
        total_c = max(-self.delta_cap, min(self.delta_cap, total_c))

        return {"a": total_a, "c": total_c}
