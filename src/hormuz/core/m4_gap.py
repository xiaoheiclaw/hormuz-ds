"""M4: Gap integrator — PRD §5.

GrossGap = C01 × S11 (exposed_supply × effective_disruption)
NetGap(t) = GrossGap - Buffer(t)
TotalGap = ∫₀ᵀ NetGap(t) dt  (trapezoidal, mbd·days)
"""

from __future__ import annotations

from hormuz.core.types import Constants, Parameters, StateVector
from hormuz.core.m3_buffer import compute_buffer_trajectory


def compute_gross_gap(constants: Constants, state: StateVector) -> float:
    """GrossGap = exposed_supply × effective_disruption."""
    return constants.exposed_supply_mbd * state.effective_disruption


def compute_net_gap(gross_gap: float, buffer: float) -> float:
    """NetGap = GrossGap - Buffer, floored at 0."""
    return max(0.0, gross_gap - buffer)


def compute_net_gap_trajectory(
    gross_gap: float,
    buffer_trajectory: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """NetGap at each point in the buffer trajectory."""
    return [(d, compute_net_gap(gross_gap, buf)) for d, buf in buffer_trajectory]


def integrate_total_gap(
    gross_gap: float,
    buffer_trajectory: list[tuple[int, float]],
    t_end: int,
) -> float:
    """∫₀ᵀ NetGap(t) dt via trapezoidal rule over buffer trajectory points.

    buffer_trajectory must cover [0, t_end] with daily resolution.
    """
    total = 0.0
    for i in range(1, min(len(buffer_trajectory), t_end + 1)):
        d0, buf0 = buffer_trajectory[i - 1]
        d1, buf1 = buffer_trajectory[i]
        ng0 = compute_net_gap(gross_gap, buf0)
        ng1 = compute_net_gap(gross_gap, buf1)
        total += (ng0 + ng1) / 2.0 * (d1 - d0)
    return total


def compute_path_total_gaps(
    gross_gap: float,
    params: Parameters,
) -> dict[str, float]:
    """Compute TotalGap for standard paths A/B/C.

    Path boundaries: A < 35 days, B = 35-120, C > 120.
    Uses representative durations: A=28, B=84, C=180.
    """
    # Generate buffer trajectory up to max path duration
    traj = compute_buffer_trajectory(max_day=180, params=params)

    return {
        "A": integrate_total_gap(gross_gap, traj, t_end=28),
        "B": integrate_total_gap(gross_gap, traj, t_end=84),
        "C": integrate_total_gap(gross_gap, traj, t_end=180),
    }
