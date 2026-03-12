"""M2: T distribution estimator — PRD §3.3.

T1: lognormal mixture weighted by ACH posterior.
T2: stock-flow mine clearance model.
T_total = T1 + deployment_gap + T2.
"""

from __future__ import annotations

import numpy as np

from hormuz.core.types import ACHPosterior, Parameters

# ── T1 lognormal parameters ──────────────────────────────────────────
# H1 (depletion): median ~17 days (2.5 weeks)
_T1_H1_MU = np.log(17)
_T1_H1_SIGMA = 0.4

# H2 (preserved): median ~42 days (6 weeks)
_T1_H2_MU = np.log(42)
_T1_H2_SIGMA = 0.35

# ── Event jump days ───────────────────────────────────────────────────
_EVENT_JUMPS = {
    "E2": 14,   # Minesweeper attack
    "E3": 7,    # Mine strike
    "C2": 21,   # Re-mining cleared lanes
}


def estimate_t1(posterior: ACHPosterior, n: int = 10000, seed: int | None = None) -> np.ndarray:
    """Sample T1 from mixture of two lognormals weighted by ACH posterior."""
    rng = np.random.default_rng(seed)

    # Mixture: draw from H1 or H2 component per sample
    w_h1 = posterior.h1 / (posterior.h1 + posterior.h2)
    mask_h1 = rng.random(n) < w_h1

    samples = np.empty(n)
    n_h1 = mask_h1.sum()
    samples[mask_h1] = rng.lognormal(_T1_H1_MU, _T1_H1_SIGMA, n_h1)
    samples[~mask_h1] = rng.lognormal(_T1_H2_MU, _T1_H2_SIGMA, n - n_h1)

    return samples


def estimate_t2(
    params: Parameters,
    events: dict[str, bool],
    n: int = 10000,
    seed: int | None = None,
) -> np.ndarray:
    """Sample T2: mine clearance time via stock-flow model.

    mines_in_water ~ Uniform(range), sweep_time = mines / (ships × rate_per_ship).
    Add event jumps for E2/E3/C2.
    """
    rng = np.random.default_rng(seed)
    lo, hi = params.mines_in_water_range

    # Sample mines in water
    mines = rng.uniform(lo, hi, n)

    # Base sweep rate: ~0.5 mines/day/ship (realistic for mixed-type in contested waters)
    rate_per_ship = 0.5
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
    """T_total = T1 + deployment_gap(7-14 days) + T2.

    Returns (t1_samples, t2_samples, t_total_samples).
    """
    rng = np.random.default_rng(seed)

    # Use sub-seeds for reproducibility
    t1 = estimate_t1(posterior, n=n, seed=rng.integers(0, 2**31))
    t2 = estimate_t2(params, events=events, n=n, seed=rng.integers(0, 2**31))
    deployment_gap = rng.uniform(7, 14, n)

    t_total = t1 + deployment_gap + t2

    return t1, t2, t_total


def compute_percentiles(samples: np.ndarray) -> dict[str, float]:
    """Compute {p10, p25, p50, p75, p90} from samples."""
    return {
        f"p{p}": float(np.percentile(samples, p))
        for p in [10, 25, 50, 75, 90]
    }
