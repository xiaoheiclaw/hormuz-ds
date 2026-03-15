"""M2: T distribution estimator — PRD §3.3.

T1: lognormal mixture weighted by ACH posterior.
T2: stock-flow mine clearance model.
T_total = T1 + deployment_gap + T2.
"""

from __future__ import annotations

import numpy as np

from hormuz.core.types import ACHPosterior, Parameters

# ── T1 lognormal parameters ──────────────────────────────────────────
# H1 (depletion): median ~17 days, wider tails for political surprises
_T1_H1_MU = np.log(17)
_T1_H1_SIGMA = 0.6

# H2 (preserved): median ~42 days, wider tails for escalation scenarios
_T1_H2_MU = np.log(42)
_T1_H2_SIGMA = 0.55

# H3 (external resupply): median ~90 days, capability sustained by foreign supply
# Attack phase ends only via political resolution or military defeat, not depletion
_T1_H3_MU = np.log(90)
_T1_H3_SIGMA = 0.5

# ── Event jump days ───────────────────────────────────────────────────
_EVENT_JUMPS = {
    "E2": 14,   # Minesweeper attack
    "E3": 7,    # Mine strike
    "C2": 21,   # Re-mining cleared lanes
}


def estimate_t1(posterior: ACHPosterior, n: int = 10000, seed: int | None = None) -> np.ndarray:
    """Sample T1 from mixture of lognormals weighted by ACH posterior.

    2-way (H3 suspended): H1/H2 mixture.
    3-way (H3 active): H1/H2/H3 mixture — H3 component has longer median (~90d).
    """
    rng = np.random.default_rng(seed)
    samples = np.empty(n)

    h3_weight = posterior.h3 if posterior.h3 is not None else 0.0
    total = posterior.h1 + posterior.h2 + h3_weight
    if total == 0:
        total = 1.0

    w_h1 = posterior.h1 / total
    w_h2 = posterior.h2 / total
    # w_h3 = h3_weight / total (remainder)

    # Draw component assignment per sample
    u = rng.random(n)
    mask_h1 = u < w_h1
    mask_h2 = (u >= w_h1) & (u < w_h1 + w_h2)
    mask_h3 = ~mask_h1 & ~mask_h2

    samples[mask_h1] = rng.lognormal(_T1_H1_MU, _T1_H1_SIGMA, mask_h1.sum())
    samples[mask_h2] = rng.lognormal(_T1_H2_MU, _T1_H2_SIGMA, mask_h2.sum())
    if mask_h3.any():
        samples[mask_h3] = rng.lognormal(_T1_H3_MU, _T1_H3_SIGMA, mask_h3.sum())

    return samples


def estimate_t2(
    params: Parameters,
    events: dict[str, bool],
    mine_signals: dict[str, float] | None = None,
    n: int = 10000,
    seed: int | None = None,
) -> np.ndarray:
    """Sample T2: mine clearance time via stock-flow model.

    mines_in_water range adjusted by observed mine-related signals:
      O03 (attack coordination) high → more mines likely, shift range up
      O10 (transit volume) low → mines blocking shipping, shift range up
      E3 (mine strike) → confirmed mines present, raise floor
      C2 (re-mining cleared lane) → active replenishment, raise ceiling

    Base range from params.mines_in_water_range (default 20-100).
    sweep_time = mines / (ships × rate_per_ship).
    Add event jumps for E2/E3/C2.
    """
    rng = np.random.default_rng(seed)
    lo, hi = params.mines_in_water_range

    # Adjust range based on observed data
    if mine_signals:
        # O03 high (>0.6): coordinated attacks suggest active mining capability
        o03 = mine_signals.get("O03", 0.5)
        if o03 > 0.6:
            lo = lo + int((o03 - 0.6) * 50)   # up to +20 at o03=1.0
            hi = hi + int((o03 - 0.6) * 75)   # up to +30 at o03=1.0

        # O10 low (<0.2): transit near-zero = heavy mining probable
        o10 = mine_signals.get("O10", 0.5)
        if o10 < 0.2:
            lo = lo + int((0.2 - o10) * 100)  # up to +20 at o10=0
            hi = hi + int((0.2 - o10) * 150)  # up to +30 at o10=0

        # E3 confirmed: mines are definitely present, raise floor
        if events.get("E3"):
            lo = max(lo, 40)

        # C2 confirmed: active replenishment, raise both
        if events.get("C2"):
            lo = max(lo, 50)
            hi = max(hi, 150)

    mines = rng.uniform(lo, hi, n)

    # Sweep rate: ~0.5 mines/day/ship, uncertain (0.3-0.8 range)
    rate_per_ship = rng.uniform(0.3, 0.8, n)
    sweep_days = mines / (params.sweep_ships * rate_per_ship)

    # Mine type penalty: mixed types take ~20% longer
    sweep_days *= 1.2

    # Add noise
    sweep_days += rng.normal(0, 2, n)
    sweep_days = np.maximum(sweep_days, 7)  # minimum 1 week

    # Event jumps
    for event_id, jump_days in _EVENT_JUMPS.items():
        if events.get(event_id):
            sweep_days += jump_days

    return sweep_days


def estimate_t_total(
    posterior: ACHPosterior,
    params: Parameters,
    events: dict[str, bool],
    mine_signals: dict[str, float] | None = None,
    n: int = 10000,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """T_total = T1 + deployment_gap(7-14 days) + T2, with tail regime jumps.

    ~8% of samples get a regime jump (surprise ceasefire or major escalation)
    to ensure path A and C tails are adequately represented.

    Returns (t1_samples, t2_samples, t_total_samples).
    """
    rng = np.random.default_rng(seed)

    # Use sub-seeds for reproducibility
    t1 = estimate_t1(posterior, n=n, seed=rng.integers(0, 2**31))
    t2 = estimate_t2(params, events=events, mine_signals=mine_signals, n=n, seed=rng.integers(0, 2**31))
    deployment_gap = rng.uniform(7, 14, n)

    t_total = t1 + deployment_gap + t2

    # Regime jumps: ~8% of samples get override to tail scenarios
    # Probability of short (ceasefire) vs long (escalation) scales with ACH
    jump_mask = rng.random(n) < 0.08
    n_jumps = jump_mask.sum()
    if n_jumps > 0:
        # H1 → ceasefire; H2+H3 → escalation (H3 = sustained resupply = escalation-favoring)
        h3_w = posterior.h3 if posterior.h3 is not None else 0.0
        p_short = posterior.h1 / (posterior.h1 + posterior.h2 + h3_w)
        short_mask = rng.random(n_jumps) < p_short
        ceasefire_t = rng.uniform(14, 30, n_jumps)
        escalation_t = rng.uniform(150, 365, n_jumps)
        new_t_total = np.where(short_mask, ceasefire_t, escalation_t)
        t_total[jump_mask] = new_t_total

        # Sync T1/T2 with jumped T_total to maintain consistency
        # Ceasefire: political end before clearance → T1 = new total, T2 = 0
        # Escalation: attack phase extends → scale T1, keep T2
        jump_indices = np.where(jump_mask)[0]
        for j, idx in enumerate(jump_indices):
            if short_mask[j]:
                # Ceasefire: conflict ends before sweep phase
                t1[idx] = new_t_total[j] - deployment_gap[idx]
                t2[idx] = 0.0
            else:
                # Escalation: longer attack, sweep unchanged
                t1[idx] = new_t_total[j] - deployment_gap[idx] - t2[idx]
            t1[idx] = max(1.0, t1[idx])

    return t1, t2, t_total


def compute_percentiles(samples: np.ndarray) -> dict[str, float]:
    """Compute {p10, p25, p50, p75, p90} from samples."""
    return {
        f"p{p}": float(np.percentile(samples, p))
        for p in [10, 25, 50, 75, 90]
    }
