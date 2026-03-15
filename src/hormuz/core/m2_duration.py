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
    posterior: ACHPosterior | None = None,
    n: int = 10000,
    seed: int | None = None,
) -> np.ndarray:
    """Sample T2: mine clearance time via stock-flow model.

    mines_in_water conditioned on ACH:
      H1 (exhaustion): 15-50 — mining capability degraded
      H2 (preserved):  40-120 — sustained mining, higher stock
      H3 (resupply):   50-150 — external supply of advanced mines
    Fallback: params.mines_in_water_range (20-100) if no posterior.

    sweep_time = mines / (ships × rate_per_ship).
    Add event jumps for E2/E3/C2.
    """
    rng = np.random.default_rng(seed)

    # ACH-conditioned mine counts
    if posterior is not None:
        h3_w = posterior.h3 if posterior.h3 is not None else 0.0
        total_w = posterior.h1 + posterior.h2 + h3_w
        if total_w == 0:
            total_w = 1.0
        w_h1 = posterior.h1 / total_w
        w_h2 = posterior.h2 / total_w

        u = rng.random(n)
        mask_h1 = u < w_h1
        mask_h2 = (u >= w_h1) & (u < w_h1 + w_h2)
        mask_h3 = ~mask_h1 & ~mask_h2

        mines = np.empty(n)
        mines[mask_h1] = rng.uniform(15, 50, mask_h1.sum())
        mines[mask_h2] = rng.uniform(40, 120, mask_h2.sum())
        if mask_h3.any():
            mines[mask_h3] = rng.uniform(50, 150, mask_h3.sum())
    else:
        lo, hi = params.mines_in_water_range
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
    t2 = estimate_t2(params, events=events, posterior=posterior, n=n, seed=rng.integers(0, 2**31))
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
