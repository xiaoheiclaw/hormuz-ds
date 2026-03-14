"""ACH scenario backtesting — verify posterior direction against known/expected outcomes.

Each test feeds simulated observations for a known scenario and asserts
the posterior converges in the correct direction. We do NOT assert exact
values — only directional correctness and rough ranges.
"""

import pytest
from datetime import datetime

from hormuz.core.types import Observation
from hormuz.core.m1_ach import run_ach


def _obs(obs_id: str, value: float) -> Observation:
    """Helper to create an observation with minimal fields."""
    return Observation(id=obs_id, timestamp=datetime(2026, 3, 14), value=value, source="test")


# ── Scenario 1: 2019 Oil Tanker Attacks (June 2019) ──────────────────
# IRGC limpet mines on tankers in Gulf of Oman. Deliberate, coordinated,
# capability fully intact. 4 tankers attacked in 2 incidents.
# Expected: H2 dominant (capability preserved, strategic choice)

class TestScenario2019TankerAttacks:
    """June 2019: IRGC limpet mine attacks on tankers — H2 should dominate."""

    def test_h2_dominates(self):
        obs = [
            _obs("O01", 0.3),   # sporadic (2 incidents, not sustained barrage)
            _obs("O02", 0.5),   # stable (no prior baseline, first attacks)
            _obs("O03", 0.7),   # coordinated (multi-target, synchronized timing)
            _obs("O04", 0.6),   # mixed weapons (limpet mines = moderate sophistication)
            _obs("O05", 0.4),   # some GPS interference but not primary tactic
            _obs("O06", 0.7),   # multi-node capability (attacks from different vectors)
        ]
        result = run_ach(obs)
        assert result.h2 > result.h1, f"Expected H2 > H1, got H1={result.h1:.2f} H2={result.h2:.2f}"
        assert result.h2 > 0.55, f"H2 should be >55%, got {result.h2:.0%}"

    def test_insurance_confirms(self):
        """Adding insurance/freight signals should reinforce H2."""
        obs = [
            _obs("O01", 0.3),
            _obs("O03", 0.7),
            _obs("O04", 0.6),
            _obs("O06", 0.7),
            # B-group confirms blockade is effective
            _obs("O07", 2.0),   # war risk premium jumped to ~2%
            _obs("O08", 0.5),   # P&I surcharges
            _obs("O10", 0.7),   # some traffic reduction but not halted
        ]
        result = run_ach(obs)
        assert result.h2 > result.h1


# ── Scenario 2: 2019 Attacks Stop (Sept 2019) ────────────────────────
# After Abqaiq attack and diplomatic pressure, Gulf tanker attacks stopped.
# Key: this was political/strategic stop, NOT capability depletion.
# Test: does ACH correctly shift toward H1 when attacks decline?
# (Even though real cause was political — ACH can't distinguish)

class TestScenario2019AttacksStop:
    """Sept 2019: attacks stop — H1 should rise (ACH sees declining capability signal)."""

    def test_h1_rises_on_decline(self):
        obs = [
            _obs("O01", 0.05),  # attacks near-zero
            _obs("O02", 0.8),   # sharp decline from previous level
            _obs("O03", 0.2),   # no coordinated attacks to observe
            _obs("O04", 0.1),   # no weapon use at all
            _obs("O05", 0.2),   # GPS spoofing reduced
            _obs("O06", 0.3),   # network appears fragmented (or just quiet)
        ]
        result = run_ach(obs)
        assert result.h1 > result.h2, f"Expected H1 > H2, got H1={result.h1:.2f} H2={result.h2:.2f}"

    def test_sensitivity_o02_is_strongest_driver(self):
        """O02 (decline rate) should be the strongest H1 signal."""
        # Only O02 saying "sharp decline" — should still push toward H1
        obs_only_o02 = [_obs("O02", 0.9)]
        result = run_ach(obs_only_o02)
        assert result.h1 > 0.55, f"O02 alone should push H1 > 55%, got {result.h1:.0%}"


# ── Scenario 3: Hypothetical Day 30 Full Depletion ────────────────────
# All IRGC capability exhausted. No attacks, no EW, fragmented network.
# Expected: H1 >> H2 (>85%)

class TestScenarioFullDepletion:
    """Hypothetical: complete IRGC capability depletion — H1 should dominate strongly."""

    def test_h1_strongly_dominant(self):
        obs = [
            _obs("O01", 0.0),   # zero attacks
            _obs("O02", 1.0),   # attacks collapsed completely
            _obs("O03", 0.0),   # no coordination at all
            _obs("O04", 0.0),   # no weapons being used
            _obs("O05", 0.0),   # GPS spoofing gone
            _obs("O06", 0.0),   # network collapsed
        ]
        result = run_ach(obs)
        assert result.h1 > 0.85, f"Full depletion: H1 should be >85%, got {result.h1:.0%}"
        assert result.h2 < 0.15, f"Full depletion: H2 should be <15%, got {result.h2:.0%}"


# ── Scenario 4: Hypothetical Day 30 Sustained Blockade ────────────────
# IRGC maintaining operations, coordinated attacks, advanced weapons in use.
# Expected: H2 >> H1 (>70%)

class TestScenarioSustainedBlockade:
    """Hypothetical: sustained IRGC capability — H2 should dominate."""

    def test_h2_strongly_dominant(self):
        obs = [
            _obs("O01", 0.5),   # moderate sustained attacks
            _obs("O02", 0.4),   # slightly declining but not collapsing
            _obs("O03", 0.7),   # still coordinated
            _obs("O04", 0.6),   # still using advanced weapons
            _obs("O05", 0.7),   # complex GPS spoofing maintained
            _obs("O06", 0.8),   # distributed multi-node network
        ]
        result = run_ach(obs, o01_trend="stable")
        assert result.h2 > 0.70, f"Sustained blockade: H2 should be >70%, got {result.h2:.0%}"

    def test_with_b_group_confirmation(self):
        """Full A+B group observation set for sustained crisis."""
        obs = [
            _obs("O01", 0.5),
            _obs("O02", 0.4),
            _obs("O03", 0.7),
            _obs("O04", 0.6),
            _obs("O05", 0.7),
            _obs("O06", 0.8),
            _obs("O07", 3.0),   # war risk 3% — extreme
            _obs("O08", 0.9),   # near full P&I exclusion
            _obs("O10", 0.2),   # only ~12 ships/day (heavily reduced)
            _obs("O11", 0.8),   # Yanbu loading active (pipeline diversion)
        ]
        result = run_ach(obs, o01_trend="stable")
        assert result.h2 > 0.70, f"Full sustained: H2 should be >70%, got {result.h2:.0%}"


# ── Scenario 5: 2024 Red Sea Crisis (Houthi) ─────────────────────────
# Houthis attacking Red Sea shipping. Sustained over months, using drones
# and missiles. No capability depletion despite US strikes.
# Mapped to Hormuz framework: persistent H2 analog.

class TestScenario2024RedSea:
    """2024 Red Sea (Houthi analog): sustained attacks despite strikes — H2 should hold."""

    def test_h2_dominant_after_months(self):
        """Note: O04/O05 mapped differently for Houthi vs IRGC.
        Houthis inherently lack advanced ASCM/EW capability, so O04/O05 are
        not diagnostic of depletion. We only use observations that ARE
        diagnostic for non-state actors: O01 (frequency), O02 (trend),
        O03 (coordination), O06 (geographic distribution).
        """
        obs = [
            _obs("O01", 0.5),   # steady ~3-5 attacks/week
            _obs("O02", 0.5),   # stable, no decline despite US strikes
            _obs("O03", 0.6),   # coordinated multi-target ops
            # O04/O05 omitted — not diagnostic for Houthi (no advanced arsenal baseline)
            _obs("O06", 0.7),   # multiple launch sites across Yemen coast
        ]
        result = run_ach(obs)
        assert result.h2 > result.h1, f"Red Sea: H2 should > H1, got H1={result.h1:.2f} H2={result.h2:.2f}"


# ── Scenario 6: T1a/T1b Unbinding ────────────────────────────────────
# Verify O05 interpretation changes based on O01 trend.

class TestT1aT1bUnbinding:
    """O05 GPS spoofing interpretation depends on attack frequency trend."""

    def test_t1a_offensive_gps_plus_rising_attacks(self):
        """GPS spoofing + rising attacks = offensive H2 (strongest H2 signal)."""
        obs = [
            _obs("O05", 0.8),   # complex spoofing active
        ]
        result = run_ach(obs, o01_trend="rising")
        assert result.h2 > result.h1, "T1a: GPS+rising should favor H2"

    def test_t1b_defensive_gps_plus_falling_attacks(self):
        """GPS spoofing + falling attacks = defensive H2 (weaker H2 signal)."""
        obs = [
            _obs("O05", 0.8),
        ]
        result_rising = run_ach(obs, o01_trend="rising")
        result_falling = run_ach(obs, o01_trend="falling")
        # Both should favor H2, but rising should be stronger
        assert result_rising.h2 > result_falling.h2, \
            f"T1a should give stronger H2 than T1b: rising={result_rising.h2:.2f} falling={result_falling.h2:.2f}"

    def test_gps_degrading_favors_h1(self):
        """Low GPS spoofing = EW capability lost → H1."""
        obs = [_obs("O05", 0.2)]
        result = run_ach(obs)
        assert result.h1 > result.h2, "Low GPS spoofing should favor H1"


# ── Sensitivity Analysis ──────────────────────────────────────────────

class TestSensitivity:
    """Verify which observations have the strongest impact on posterior."""

    def test_strong_discriminators_move_posterior_more(self):
        """O02-O05 (strong) should shift posterior more than O01/O06 (moderate)."""
        # Single strong observation (O03 coordination, high = H2)
        result_strong = run_ach([_obs("O03", 0.8)])
        shift_strong = abs(result_strong.h2 - 0.5)

        # Single moderate observation (O01 attack freq, high = H2)
        result_moderate = run_ach([_obs("O01", 0.8)])
        shift_moderate = abs(result_moderate.h2 - 0.5)

        assert shift_strong > shift_moderate, \
            f"Strong discriminator should shift more: strong={shift_strong:.3f} moderate={shift_moderate:.3f}"

    def test_cap_at_95_percent(self):
        """Even with all observations pointing one way, posterior shouldn't exceed ~95%."""
        obs = [
            _obs("O01", 0.0),
            _obs("O02", 1.0),
            _obs("O03", 0.0),
            _obs("O04", 0.0),
            _obs("O05", 0.0),
            _obs("O06", 0.0),
            _obs("O07", 0.1),
            _obs("O08", 0.0),
            _obs("O10", 1.0),
            _obs("O11", 0.1),
        ]
        result = run_ach(obs)
        assert result.h1 <= 0.96, f"Cap should hold: H1={result.h1:.0%}"
