import pytest
from hormuz.core.types import PathWeights


def test_no_signals():
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    result = adjust_path_weights(base, active_signals=[])
    assert result.a == pytest.approx(0.30)


def test_mediation_signal():
    """D03 mediation -> A+=0.15, C-=0.10, B-=0.05"""
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    result = adjust_path_weights(base, active_signals=["mediation"])
    assert result.a > base.a
    assert result.c < base.c


def test_escalation_signal():
    """E1 target spillover -> C+=0.15, A-=0.10, B-=0.05"""
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    result = adjust_path_weights(base, active_signals=["escalation"])
    assert result.c > base.c
    assert result.a < base.a


def test_multiple_signals():
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    result = adjust_path_weights(base, active_signals=["mediation", "commitment_softening"])
    assert result.a > base.a + 0.1


def test_always_normalized():
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    for signals in [["mediation"], ["escalation"], ["mediation", "escalation"]]:
        result = adjust_path_weights(base, active_signals=signals)
        assert abs(result.a + result.b + result.c - 1.0) < 1e-9


def test_clip_bounds():
    """Extreme signals should still clip to [0.05, 0.85]"""
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    result = adjust_path_weights(base, active_signals=["mediation", "commitment_softening", "commitment_lock"])
    assert result.a <= 0.85
    assert result.c >= 0.05
