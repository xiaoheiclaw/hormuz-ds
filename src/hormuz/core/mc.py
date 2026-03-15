"""Monte Carlo simulation engine — PRD §7.

Samples T1, T2, Buffer trajectory, integrates TotalGap per sample.
Classifies by T boundaries: A < 35, B = 35-120, C > 120 days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hormuz.core.types import ACHPosterior, Parameters
from hormuz.core.m2_duration import estimate_t_total, compute_percentiles
from hormuz.core.m3_buffer import compute_buffer_trajectory
from hormuz.core.m4_gap import integrate_total_gap


@dataclass
class MCResult:
    """Monte Carlo simulation output."""
    t1_samples: np.ndarray          # shape (N,) — T1 phase
    t2_samples: np.ndarray          # shape (N,) — T2 phase
    t_samples: np.ndarray           # shape (N,) — T_total
    total_gap_samples: np.ndarray   # shape (N,)
    t1_percentiles: dict[str, float]
    t2_percentiles: dict[str, float]
    t_percentiles: dict[str, float]
    path_frequencies: dict[str, float]    # {"A": frac, "B": frac, "C": frac}
    path_total_gap_means: dict[str, float]  # per-path average TotalGap
    buffer_trajectory: list[tuple[int, float]]  # shared buffer trajectory


# ── Path classification ───────────────────────────────────────────────

_PATH_BOUNDARIES = (35, 120)


def classify_paths(
    t_samples: np.ndarray,
    boundaries: tuple[int, int] = _PATH_BOUNDARIES,
) -> dict[str, int]:
    """Count samples per path: A < b1, B = b1-b2, C > b2."""
    b1, b2 = boundaries
    return {
        "A": int(np.sum(t_samples < b1)),
        "B": int(np.sum((t_samples >= b1) & (t_samples <= b2))),
        "C": int(np.sum(t_samples > b2)),
    }


# ── Main simulation ──────────────────────────────────────────────────

def run_monte_carlo(
    posterior: ACHPosterior,
    params: Parameters,
    events: dict[str, bool],
    n: int = 10000,
    seed: int | None = None,
    spr_trigger_day: int | None = None,
    gross_gap_mbd: float | None = None,
) -> MCResult:
    """Run N-sample Monte Carlo simulation.

    Per sample: T1 + deployment_gap(7-14) + T2 = T_total.
    Then compute buffer trajectory and integrate TotalGap for that T.
    """
    rng = np.random.default_rng(seed)

    # Sample T1, T2, T_total via M2 (includes regime jumps)
    mc_seed = int(rng.integers(0, 2**31))
    t1, t2, t_total = estimate_t_total(posterior, params, events, n=n, seed=mc_seed)

    # Pre-compute buffer trajectory — at least 181 days for Path C gap integration
    max_t = max(181, int(np.ceil(np.max(t_total))) + 1)
    buffer_traj = compute_buffer_trajectory(
        max_day=max_t, params=params, spr_trigger_day=spr_trigger_day,
    )

    # Integrate TotalGap per sample (use explicit gross_gap if provided)
    gross_gap = gross_gap_mbd if gross_gap_mbd is not None else params.gross_gap_mbd
    total_gap_samples = np.array([
        integrate_total_gap(gross_gap, buffer_traj, t_end=max(1, int(round(t))))
        for t in t_total
    ])

    # Path classification
    counts = classify_paths(t_total)
    total_n = sum(counts.values())
    path_freq = {k: v / total_n for k, v in counts.items()}

    # Per-path mean TotalGap
    path_means: dict[str, float] = {}
    b1, b2 = _PATH_BOUNDARIES
    masks = {
        "A": t_total < b1,
        "B": (t_total >= b1) & (t_total <= b2),
        "C": t_total > b2,
    }
    for path, mask in masks.items():
        if mask.any():
            path_means[path] = float(np.mean(total_gap_samples[mask]))
        else:
            path_means[path] = 0.0

    return MCResult(
        t1_samples=t1,
        t2_samples=t2,
        t_samples=t_total,
        total_gap_samples=total_gap_samples,
        t1_percentiles=compute_percentiles(t1),
        t2_percentiles=compute_percentiles(t2),
        t_percentiles=compute_percentiles(t_total),
        path_frequencies=path_freq,
        path_total_gap_means=path_means,
        buffer_trajectory=buffer_traj,
    )
