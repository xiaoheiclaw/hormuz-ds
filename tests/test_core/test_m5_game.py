"""M5 Schelling credibility-based game theory tests.

Tests define the API contract:
  - SignalEvidence(key, evidence) replaces plain string signals
  - Adjustment strength = credibility × evidence × sensitivity
  - Focal convergence: same-direction signals amplify non-linearly
  - Combo prereqs preserved
"""
import pytest
from hormuz.core.types import PathWeights


# ── Helpers ──────────────────────────────────────────────────────────

def _adj(base, signals):
    from hormuz.core.m5_game import adjust_path_weights, SignalEvidence
    sigs = [SignalEvidence(k, e) for k, e in signals]
    return adjust_path_weights(base, sigs)


BASE = PathWeights()  # default A=0.30, B=0.50, C=0.20


# ── 1. Basic contract ───────────────────────────────────────────────

def test_no_signals():
    result = _adj(BASE, [])
    assert result.a == pytest.approx(0.30)
    assert result.b == pytest.approx(0.50)
    assert result.c == pytest.approx(0.20)


def test_unknown_signal_ignored():
    result = _adj(BASE, [("unknown_xyz", 1.0)])
    assert result.a == pytest.approx(BASE.a)


def test_always_normalized():
    cases = [
        [("external_mediation", 1.0)],
        [("irgc_escalation", 1.0)],
        [("external_mediation", 0.8), ("irgc_escalation", 0.6)],
    ]
    for sigs in cases:
        result = _adj(BASE, sigs)
        assert abs(result.a + result.b + result.c - 1.0) < 1e-9


def test_clip_bounds():
    """Even extreme inputs stay within [0.05, 0.85]."""
    result = _adj(BASE, [
        ("external_mediation", 1.0),
        ("costly_self_binding", 1.0),
    ])
    assert result.a <= 0.85
    assert result.c >= 0.05


# ── 2. Evidence modulates strength ──────────────────────────────────

def test_high_evidence_moves_more_than_low():
    """Same signal, high evidence should shift more than low evidence."""
    r_high = _adj(BASE, [("irgc_escalation", 1.0)])
    r_low = _adj(BASE, [("irgc_escalation", 0.2)])

    shift_high = r_high.c - BASE.c
    shift_low = r_low.c - BASE.c
    assert shift_high > shift_low > 0


def test_zero_evidence_no_change():
    """Evidence=0 should produce no change regardless of credibility."""
    result = _adj(BASE, [("irgc_escalation", 0.0)])
    assert result.c == pytest.approx(BASE.c)


def test_evidence_roughly_proportional():
    """Double evidence ≈ double the shift (not exact due to normalization)."""
    r_half = _adj(BASE, [("costly_self_binding", 0.5)])
    r_full = _adj(BASE, [("costly_self_binding", 1.0)])

    shift_half = r_half.a - BASE.a
    shift_full = r_full.a - BASE.a
    ratio = shift_full / shift_half if shift_half > 0 else 0
    assert 1.5 < ratio < 2.5  # roughly 2x


# ── 3. Credibility matters ──────────────────────────────────────────

def test_costly_signal_stronger_than_cheap():
    """At same evidence, costly_self_binding (high cost) should shift A
    more than external_mediation (low cost). Both push A direction."""
    r_costly = _adj(BASE, [("costly_self_binding", 1.0)])
    r_cheap = _adj(BASE, [("external_mediation", 1.0)])

    shift_costly = r_costly.a - BASE.a
    shift_cheap = r_cheap.a - BASE.a
    assert shift_costly > shift_cheap > 0


def test_escalation_direction():
    """irgc_escalation pushes C up, A down."""
    result = _adj(BASE, [("irgc_escalation", 1.0)])
    assert result.c > BASE.c
    assert result.a < BASE.a


def test_mediation_direction():
    """external_mediation pushes A up, C down."""
    result = _adj(BASE, [("external_mediation", 1.0)])
    assert result.a > BASE.a
    assert result.c < BASE.c


def test_us_inconsistency_pushes_b():
    """us_inconsistency direction=B, should increase B."""
    result = _adj(BASE, [("us_inconsistency", 1.0)])
    assert result.b > BASE.b


# ── 4. Focal convergence ────────────────────────────────────────────

def test_focal_convergence_nonlinear():
    """Two same-direction signals should shift more than sum of individuals.

    mediation + self_binding both push A.
    Combined shift > mediation_shift + self_binding_shift (non-linear amplification).
    """
    r_med = _adj(BASE, [("external_mediation", 1.0)])
    r_self = _adj(BASE, [("costly_self_binding", 1.0)])
    r_both = _adj(BASE, [("external_mediation", 1.0), ("costly_self_binding", 1.0)])

    shift_sum = (r_med.a - BASE.a) + (r_self.a - BASE.a)
    shift_combined = r_both.a - BASE.a
    assert shift_combined > shift_sum  # focal amplification


def test_opposing_signals_partially_cancel():
    """mediation (→A) + escalation (→C) should partially cancel out."""
    r_both = _adj(BASE, [("external_mediation", 1.0), ("irgc_escalation", 1.0)])

    # Neither A nor C should shift as much as when alone
    r_med_only = _adj(BASE, [("external_mediation", 1.0)])
    r_esc_only = _adj(BASE, [("irgc_escalation", 1.0)])

    assert r_both.a < r_med_only.a  # less A than mediation alone
    assert r_both.c < r_esc_only.c  # less C than escalation alone


# ── 5. Combo prereqs ────────────────────────────────────────────────

def test_irgc_fragmentation_requires_us_inconsistency():
    r_alone = _adj(BASE, [("irgc_fragmentation", 1.0)])
    assert r_alone.a == pytest.approx(BASE.a)

    r_combo = _adj(BASE, [
        ("us_inconsistency", 1.0),
        ("irgc_fragmentation", 1.0),
    ])
    # fragmentation pushes A, us_inconsistency pushes B
    # net: A should still increase (fragmentation is stronger toward A)
    assert r_combo.a > BASE.a


# ── 6. Sensitivity sanity ───────────────────────────────────────────

def test_single_low_evidence_cheap_signal_tiny():
    """Cheap talk + low evidence → near-zero shift (< 2pp)."""
    result = _adj(BASE, [("external_mediation", 0.2)])
    assert abs(result.a - BASE.a) < 0.02


def test_single_high_evidence_costly_signal_moderate():
    """Costly + high evidence → moderate shift (3-15pp)."""
    result = _adj(BASE, [("costly_self_binding", 1.0)])
    shift = result.a - BASE.a
    assert 0.03 < shift < 0.15


def test_full_convergence_large_shift():
    """Two same-direction high-evidence signals → focal convergence shift."""
    result = _adj(BASE, [
        ("external_mediation", 1.0),
        ("costly_self_binding", 1.0),
    ])
    shift = result.a - BASE.a
    assert shift > 0.04
