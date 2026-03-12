"""M3: Buffer ramp function — PRD §4.

Buffer(t) = pipeline(t) + spr(t) + cape(t)
Three independent sub-systems with different activation delays and ramp profiles.
"""

from __future__ import annotations

from hormuz.core.types import Parameters


def _ramp(t: float, start: float, duration: float, end_val: float) -> float:
    """Linear ramp from 0 to end_val over [start, start+duration], clamped."""
    if t <= start:
        return 0.0
    if t >= start + duration:
        return end_val
    return end_val * (t - start) / duration


def pipeline_buffer(
    day: int,
    params: Parameters,
    pipeline_override: float | None = None,
) -> float:
    """Pipeline bypass: ADCOP (D3, quick) + Saudi East-West (D5-14, slower).

    ADCOP: 0.5 mbd, starts day 3, ramps over 2 days.
    Saudi pipeline: up to pipeline_max_mbd - 0.5, starts day 5, ramps over ~10 days.
    """
    max_mbd = pipeline_override if pipeline_override is not None else params.pipeline_max_mbd

    # ADCOP: small, fast
    adcop = _ramp(day, start=3, duration=2, end_val=0.5)

    # Saudi East-West pipeline: larger, slower
    saudi_max = max_mbd - 0.5  # remaining after ADCOP
    ramp_days = params.pipeline_ramp_weeks * 7  # ~17.5 days
    saudi = _ramp(day, start=5, duration=ramp_days, end_val=saudi_max)

    return adcop + saudi


def spr_buffer(
    day: int,
    params: Parameters,
    spr_trigger_day: int | None = None,
    spr_override: float | None = None,
) -> float:
    """SPR release: hard delay then linear ramp.

    13-day delay (spr_pump_min_days) after trigger, then ramp to spr_rate_mean over ~7 days.
    """
    if spr_trigger_day is None:
        return 0.0

    delay = params.spr_pump_min_days
    arrival_day = spr_trigger_day + delay
    rate = spr_override if spr_override is not None else params.spr_rate_mean_mbd

    return _ramp(day, start=arrival_day, duration=7, end_val=rate)


def cape_buffer(day: int) -> float:
    """Cape of Good Hope reroute: first arrivals D14, gradual to ~2.5 mbd by D35."""
    return _ramp(day, start=14, duration=21, end_val=2.5)


def compute_buffer(
    day: int,
    params: Parameters,
    spr_trigger_day: int | None = None,
    pipeline_override: float | None = None,
    spr_override: float | None = None,
) -> float:
    """Total buffer = pipeline + SPR + cape reroute."""
    return (
        pipeline_buffer(day, params, pipeline_override)
        + spr_buffer(day, params, spr_trigger_day, spr_override)
        + cape_buffer(day)
    )


def compute_buffer_trajectory(
    max_day: int,
    params: Parameters,
    spr_trigger_day: int | None = None,
    pipeline_override: float | None = None,
    spr_override: float | None = None,
) -> list[tuple[int, float]]:
    """Buffer(t) for t in [0, max_day]."""
    return [
        (d, compute_buffer(d, params, spr_trigger_day, pipeline_override, spr_override))
        for d in range(max_day + 1)
    ]
