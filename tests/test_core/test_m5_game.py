import pytest
from hormuz.core.types import PathWeights


def test_no_signals():
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    result = adjust_path_weights(base, active_signals=[])
    assert result.a == pytest.approx(0.30)


def test_mediation_signal():
    """external_mediation -> A+=0.05, C-=0.03"""
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    result = adjust_path_weights(base, active_signals=["external_mediation"])
    assert result.a > base.a
    assert result.c < base.c


def test_escalation_signal():
    """irgc_escalation -> C+=0.10, A-=0.05"""
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    result = adjust_path_weights(base, active_signals=["irgc_escalation"])
    assert result.c > base.c
    assert result.a < base.a


def test_multiple_signals():
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    result = adjust_path_weights(base, active_signals=["external_mediation", "costly_self_binding"])
    assert result.a > base.a


def test_always_normalized():
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    for signals in [["external_mediation"], ["irgc_escalation"],
                    ["external_mediation", "irgc_escalation"]]:
        result = adjust_path_weights(base, active_signals=signals)
        assert abs(result.a + result.b + result.c - 1.0) < 1e-9


def test_clip_bounds():
    """Extreme signals should still clip to [0.05, 0.85]"""
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    result = adjust_path_weights(base, active_signals=[
        "external_mediation", "costly_self_binding", "us_inconsistency",
    ])
    assert result.a <= 0.85
    assert result.c >= 0.05


def test_combo_signal_requires_prereqs():
    """peace_window requires both external_mediation + costly_self_binding"""
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()

    # peace_window alone — should be skipped (missing prereqs)
    result_alone = adjust_path_weights(base, active_signals=["peace_window"])
    assert result_alone.a == pytest.approx(base.a)

    # peace_window with prereqs — should fire
    result_combo = adjust_path_weights(base, active_signals=[
        "external_mediation", "costly_self_binding", "peace_window",
    ])
    assert result_combo.a > base.a


def test_unknown_signal_ignored():
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    result = adjust_path_weights(base, active_signals=["unknown_signal_xyz"])
    assert result.a == pytest.approx(base.a)
