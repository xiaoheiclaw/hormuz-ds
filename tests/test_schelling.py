"""Tests for Schelling Signal Sheet — game theory delta adjustments."""
from __future__ import annotations

import pytest

from hormuz.engine.schelling import SchellingSheet


@pytest.fixture
def schelling(constants) -> SchellingSheet:
    return SchellingSheet(constants["schelling"])


class TestDeltaOutput:
    def test_class_a_signal_triggers_alone(self, schelling):
        """Signal 1 alone -> a up, c down."""
        delta = schelling.compute_delta({1: True}, current_week=5)
        assert delta["a"] > 0
        assert delta["c"] < 0

    def test_class_b_needs_combo(self, schelling):
        """Signal 5 alone -> no delta; Signal 5 + 1 -> delta."""
        delta_alone = schelling.compute_delta({5: True}, current_week=5)
        assert delta_alone["a"] == 0

        delta_combo = schelling.compute_delta({5: True, 1: True}, current_week=5)
        assert delta_combo["a"] > 0

    def test_signal_6_needs_signal_2(self, schelling):
        """Signal 6 alone -> no delta; Signal 6 + 2 -> delta."""
        delta_alone = schelling.compute_delta({6: True}, current_week=5)
        assert delta_alone["a"] == 0

        delta_combo = schelling.compute_delta({6: True, 2: True}, current_week=5)
        assert delta_combo["a"] > 0

    def test_delta_capped_at_10pp(self, schelling):
        """All A-class signals active -> still capped at +/-0.10."""
        delta = schelling.compute_delta(
            {1: True, 2: True, 3: True, 4: True}, current_week=5
        )
        assert abs(delta["a"]) <= 0.10
        assert abs(delta["c"]) <= 0.10

    def test_no_signals_no_delta(self, schelling):
        delta = schelling.compute_delta({}, current_week=5)
        assert delta["a"] == 0
        assert delta["c"] == 0

    def test_signal_4_escalation_boosts_c(self, schelling):
        """Signal 4 (IRGC escalation) -> C up big, A down."""
        delta = schelling.compute_delta({4: True}, current_week=5)
        assert delta["c"] > 0
        assert delta["a"] < 0

    def test_all_signals_combined(self, schelling):
        """All 6 signals active — B conditions met, capping applies."""
        delta = schelling.compute_delta(
            {1: True, 2: True, 3: True, 4: True, 5: True, 6: True},
            current_week=5,
        )
        assert abs(delta["a"]) <= 0.10
        assert abs(delta["c"]) <= 0.10

    def test_signal_5_needs_either_1_or_3(self, schelling):
        """Signal 5 fires if signal 3 is active (not only signal 1)."""
        delta = schelling.compute_delta({5: True, 3: True}, current_week=5)
        assert delta["a"] > 0

    def test_inactive_signals_ignored(self, schelling):
        """Signals marked False are not counted."""
        delta = schelling.compute_delta({1: False, 2: False}, current_week=5)
        assert delta["a"] == 0
        assert delta["c"] == 0


class TestWeekGating:
    def test_before_w4_returns_zero(self, schelling):
        """Before W4 -> only record, no delta."""
        delta = schelling.compute_delta({1: True}, current_week=2)
        assert delta["a"] == 0
        assert delta["c"] == 0

    def test_w4_onwards_outputs_delta(self, schelling):
        delta = schelling.compute_delta({1: True}, current_week=4)
        assert delta["a"] > 0

    def test_week_3_returns_zero(self, schelling):
        delta = schelling.compute_delta({1: True, 4: True}, current_week=3)
        assert delta["a"] == 0
        assert delta["c"] == 0

    def test_none_week_outputs_delta(self, schelling):
        """current_week=None should produce delta (auto-detect >= W4)."""
        delta = schelling.compute_delta({1: True}, current_week=None)
        assert delta["a"] > 0
