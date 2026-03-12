"""MC Model Phase 1: Analytical approximation (v5.4).

v5.4 model: step-function disruption, not exponential decay.
    gross_gap = 16 mbd (constant during T, step to 0 at T-end)
    net_gap(t) = gross_gap - buffer(t)
    buffer(t): D1-14 ~1.5 mbd, D14+ ~7 mbd (piecewise ramp)
    price(day) = base_price + net_gap(day) * PRICE_SENSITIVITY

Path-specific T duration determines total cumulative gap (mbd·days).
IRGC decay_mean only affects T1 length, not disruption rate.
"""

from __future__ import annotations

import math

from hormuz.models import MCParams, MCResult


class MCModel:
    """Phase 1: v5.4 analytical approximation using piecewise net gap."""

    BASE_PRICE = 75.0       # Pre-crisis Brent baseline ($/bbl)
    GROSS_GAP = 16.0        # 20.1 mbd × 80% effective disruption (step function)
    PRICE_SENSITIVITY = 8.0 # $/barrel per mbd of supply gap

    # Buffer ramp breakpoints (mbd)
    BUFFER_D1_14 = 1.5      # D1-14: pipeline switch + initial commercial drawdown
    BUFFER_D14_PLUS = 7.0   # D14+: SPR + pipeline at capacity + reroute
    BUFFER_PATH_C_CRASH = 2.0  # Fujairah hit: buffer collapses

    _HORIZON_WEEKS = 26     # Extended horizon for path C visibility

    def run(self, params: MCParams) -> MCResult:
        """Run Phase 1 v5.4 approximation.

        Each path has a characteristic T (disruption duration in weeks).
        During T, gross gap = 16 mbd; net gap = 16 - buffer(t).
        After T, net gap drops to 0 (step function, not gradual).
        """
        # Path A: ~4 weeks (fast Q1 + low mine density + diplomatic focal point)
        t_a_weeks = params.irgc_decay_mean * 0.6 + params.convoy_start_mean * 0.3
        price_a = self._path_price(t_weeks=min(t_a_weeks, 4.0), path_c_crash=False)

        # Path B: ~12 weeks (Q1 ~6w decay + Q2 ~5w sweep, serial)
        t_b_weeks = params.irgc_decay_mean + params.convoy_start_mean
        price_b = self._path_price(t_weeks=t_b_weeks, path_c_crash=False)

        # Path C: >26 weeks (mines persist + escalation, buffer crash)
        t_c_weeks = self._HORIZON_WEEKS
        price_c = self._path_price(t_weeks=t_c_weeks, path_c_crash=True)

        w = params.path_weights
        price_mean = w.a * price_a + w.b * price_b + w.c * price_c

        prices_sorted = sorted([price_a, price_b, price_c])

        return MCResult(
            timestamp=params.timestamp,
            price_mean=round(price_mean, 2),
            price_p10=round(prices_sorted[0], 2),
            price_p50=round(prices_sorted[1], 2),
            price_p90=round(prices_sorted[2], 2),
            path_a_price=round(price_a, 2),
            path_b_price=round(price_b, 2),
            path_c_price=round(price_c, 2),
        )

    def _path_price(self, t_weeks: float, path_c_crash: bool) -> float:
        """Calculate average Brent price for one path over the horizon.

        v5.4 piecewise:
        - During T (day <= t_days): net_gap = GROSS_GAP - buffer(day)
        - After T: net_gap = 0 (step function recovery)
        - Path C crash: buffer drops to BUFFER_PATH_C_CRASH after D14
        """
        t_days = t_weeks * 7
        total_price = 0.0
        horizon_days = self._HORIZON_WEEKS * 7

        for d in range(1, horizon_days + 1):
            if d <= t_days:
                # During disruption: net gap = gross gap - buffer
                if d <= 14:
                    buffer = self.BUFFER_D1_14
                elif path_c_crash:
                    buffer = self.BUFFER_PATH_C_CRASH
                else:
                    buffer = self.BUFFER_D14_PLUS
                net_gap = max(0.0, self.GROSS_GAP - buffer)
            else:
                # After T-end: step recovery
                net_gap = 0.0

            price = self.BASE_PRICE + net_gap * self.PRICE_SENSITIVITY
            total_price += price

        return total_price / horizon_days
