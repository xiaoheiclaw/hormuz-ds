"""MC Model Phase 1: Analytical approximation.

Each path defines a Brent price trajectory over time.
The oil price equation:
    price(week) = base_price + supply_gap(week) * PRICE_SENSITIVITY

Where:
    supply_gap(week) = disrupted_flow * disruption_pct - buffer(week)
    disrupted_flow = 17.0 mbd (normal Hormuz flow)

Path-specific parameters control disruption_pct and recovery timing.
"""

from __future__ import annotations

import math

from hormuz.models import MCParams, MCResult


class MCModel:
    """Phase 1: Analytical approximation using three parameterised paths."""

    # Constants
    BASE_PRICE = 75.0       # Pre-crisis Brent baseline ($/bbl)
    NORMAL_FLOW = 17.0      # Normal Hormuz flow (mbd)
    PRICE_SENSITIVITY = 8.0 # $/barrel per mbd of supply gap

    # Simulation horizon
    _HORIZON_WEEKS = 12

    def run(self, params: MCParams) -> MCResult:
        """Run Phase 1 analytical approximation.

        Calculates price trajectories for three paths (A/B/C), then
        blends them using path weights.
        """
        d_low, d_high = params.disruption_range

        # Path A: fast de-escalation
        price_a = self._path_price(
            weeks=self._HORIZON_WEEKS,
            decay_mean=params.irgc_decay_mean * 0.6,
            convoy_start=params.convoy_start_mean * 0.6,
            disruption_low=d_low,
            disruption_high=d_high,
            pipeline_max=params.pipeline_max,
            pipeline_ramp_weeks=params.pipeline_ramp_weeks,
            spr_rate=params.spr_rate_mean,
            spr_delay_weeks=params.spr_delay_weeks,
            surplus=params.surplus_buffer,
        )

        # Path B: gradual attrition
        price_b = self._path_price(
            weeks=self._HORIZON_WEEKS,
            decay_mean=params.irgc_decay_mean,
            convoy_start=params.convoy_start_mean,
            disruption_low=d_low,
            disruption_high=d_high,
            pipeline_max=params.pipeline_max,
            pipeline_ramp_weeks=params.pipeline_ramp_weeks,
            spr_rate=params.spr_rate_mean,
            spr_delay_weeks=params.spr_delay_weeks,
            surplus=params.surplus_buffer,
        )

        # Path C: prolonged standoff
        price_c = self._path_price(
            weeks=self._HORIZON_WEEKS,
            decay_mean=params.irgc_decay_mean * 1.5,
            convoy_start=params.convoy_start_mean * 2.0,
            disruption_low=d_low,
            disruption_high=d_high,
            pipeline_max=params.pipeline_max,
            pipeline_ramp_weeks=params.pipeline_ramp_weeks,
            spr_rate=params.spr_rate_mean,
            spr_delay_weeks=params.spr_delay_weeks,
            surplus=params.surplus_buffer,
        )

        w = params.path_weights
        price_mean = w.a * price_a + w.b * price_b + w.c * price_c

        # Quantile approximation from 3 paths
        prices_sorted = sorted([price_a, price_b, price_c])
        price_p10 = prices_sorted[0]   # best case
        price_p50 = prices_sorted[1]   # median (typically path B)
        price_p90 = prices_sorted[2]   # worst case

        return MCResult(
            timestamp=params.timestamp,
            price_mean=round(price_mean, 2),
            price_p10=round(price_p10, 2),
            price_p50=round(price_p50, 2),
            price_p90=round(price_p90, 2),
            path_a_price=round(price_a, 2),
            path_b_price=round(price_b, 2),
            path_c_price=round(price_c, 2),
        )

    def _path_price(
        self,
        weeks: int,
        decay_mean: float,
        convoy_start: float,
        disruption_low: float,
        disruption_high: float,
        pipeline_max: float,
        pipeline_ramp_weeks: float,
        spr_rate: float,
        spr_delay_weeks: float,
        surplus: float,
    ) -> float:
        """Calculate average Brent price over time horizon for one path.

        Week by week:
        1. IRGC capability = exp(-week / decay_mean)
        2. disruption_pct = d_low + (d_high - d_low) * capability
        3. disrupted = NORMAL_FLOW * disruption_pct
        4. buffer = pipeline + spr + reroute + surplus
        5. gap = max(0, disrupted - buffer)
        6. price = BASE_PRICE + gap * PRICE_SENSITIVITY

        Returns average price over weeks 1..weeks.
        """
        total_price = 0.0
        for w in range(1, weeks + 1):
            # 1. IRGC capability decays exponentially
            capability = math.exp(-w / decay_mean)

            # 2. Disruption percentage (high when capable, low when degraded)
            disruption_pct = disruption_low + (disruption_high - disruption_low) * capability

            # 3. Disrupted flow (mbd)
            disrupted = self.NORMAL_FLOW * disruption_pct

            # 4. Buffer ramp components
            # Pipeline: linear ramp to max
            if pipeline_ramp_weeks > 0:
                pipeline = min(pipeline_max, pipeline_max * w / pipeline_ramp_weeks)
            else:
                pipeline = pipeline_max

            # SPR: available after delay
            spr_val = spr_rate if w >= spr_delay_weeks else 0.0

            # Reroute: gradual from ~W2, caps at 2.0 mbd
            reroute = min(2.0, 0.5 * max(0.0, w - 1.5))

            # Total buffer
            buffer = pipeline + spr_val + reroute + surplus

            # Convoy effect: before convoy starts, buffer is less effective
            # (convoys enable safe transit, reducing effective disruption)
            if w < convoy_start:
                # Before convoy, only surplus and partial pipeline help
                buffer = surplus + pipeline * 0.5

            # 5. Net supply gap
            gap = max(0.0, disrupted - buffer)

            # 6. Price
            price = self.BASE_PRICE + gap * self.PRICE_SENSITIVITY
            total_price += price

        return total_price / weeks
