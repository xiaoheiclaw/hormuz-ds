"""Physical layer: state equations for Q1/Q2/Q3 and regime→parameter mapping.

v5.4: Disruption rate is a step function (~80%) during T, not driven by
IRGC capability decay. Decay only affects T1 length (when Q2 can start).
Insurance withdrawal is actuarial reflection of physical threats, not causal.
"""
import math

from hormuz.models import RegimeType


class PhysicalLayer:
    """Physical model for Hormuz strait crisis scenarios.

    v5.4 model:
    - Q1: IRGC capability decay (only affects T1 length, not disruption rate)
    - Q2: Mine density from U[20,100] + sweep rate → T2 duration
    - Q3: Buffer ramp-up (pipeline + SPR + reroute + surplus)
    - Gross gap = 16 mbd constant during T = T1 + T2
    """

    def __init__(self, params_config: dict):
        """Load from parameters.yaml["physical"]."""
        q1 = params_config["q1"]
        q2 = params_config["q2"]
        q3 = params_config["q3"]

        # v5.4 gross gap constants
        self.normal_flow = params_config.get("normal_flow_mbd", 20.1)
        self.effective_disruption_rate = params_config.get("effective_disruption_rate", 0.80)
        self.gross_gap = params_config.get("gross_gap_mbd", 16.0)

        # Q1 defaults
        self.irgc_decay_mean_default = q1["irgc_decay_mean_weeks"]
        self.attack_threshold = q1["attack_threshold_per_day"]
        self.h3_suspended = q1.get("h3_suspended", True)

        # Q2 defaults
        self.mine_initial_stock = q2["mine_initial_stock"]
        self.convoy_start_mean_default = q2["convoy_start_mean_weeks"]
        # v5.4: uniform distribution, not point estimate
        self.sea_mines_range = q2.get("sea_mines_range", [20, 100])
        self.sweep_ships_available = q2["sweep_ships_available"]

        # Q3 defaults
        self.pipeline_max = q3["pipeline_max_mbd"]
        self.pipeline_ramp_weeks = q3["pipeline_ramp_weeks"]
        self.spr_rate_mean = q3["spr_rate_mean_mbd"]
        self.spr_delay_weeks = q3["spr_delay_weeks"]
        self.surplus_buffer = q3["surplus_buffer_mbd"]

    def update_params(self, q1_regime: RegimeType, q2_regime: RegimeType) -> dict:
        """Map regime judgments to MC parameter adjustments.

        v5.4: decay_mean only affects T1 length; disruption rate is constant 80%.
        convoy_start_mean still driven by Q2 regime. No more disruption_range.
        """
        # Q1 regime → decay mean (affects T1 duration only)
        decay_map = {
            RegimeType.wide: self.irgc_decay_mean_default,
            RegimeType.lean_h1: 4.0,
            RegimeType.lean_h2: self.irgc_decay_mean_default,
            RegimeType.confirmed_h3: 15.0,
        }
        irgc_decay_mean = decay_map[q1_regime]

        # Q2 regime → convoy start mean
        convoy_map = {
            RegimeType.wide: self.convoy_start_mean_default,
            RegimeType.lean_h1: self.convoy_start_mean_default - 1.0,
            RegimeType.lean_h2: self.convoy_start_mean_default + 1.5,
            RegimeType.confirmed_h3: self.convoy_start_mean_default,
        }
        convoy_start_mean = convoy_map[q2_regime]

        return {
            "irgc_decay_mean": irgc_decay_mean,
            "convoy_start_mean": convoy_start_mean,
            "pipeline_max": self.pipeline_max,
            "pipeline_ramp_weeks": self.pipeline_ramp_weeks,
            "spr_rate_mean": self.spr_rate_mean,
            "spr_delay_weeks": self.spr_delay_weeks,
            "surplus_buffer": self.surplus_buffer,
        }

    def q1_capability(self, week: float, decay_mean: float) -> float:
        """IRGC capability at week t: exp(-t / decay_mean).

        Returns fraction of initial capability (1.0 at week 0).
        """
        return math.exp(-week / decay_mean)

    def q2_net_flow(
        self, deploy_rate: float, sweep_rate: float, q1_attack_freq: float
    ) -> float:
        """Net mine flow: deploy_rate - effective_sweep_rate.

        If q1_attack_freq > attack_threshold, sweep_rate = 0 (Q1 gate).
        Positive = mines accumulating, Negative = mines being cleared.
        """
        effective_sweep = 0.0 if q1_attack_freq > self.attack_threshold else sweep_rate
        return deploy_rate - effective_sweep

    def q3_buffer(self, day: int) -> float:
        """Total buffer available at day D (mbd).

        Piecewise ramp:
        - Pipeline: min(pipeline_max, pipeline_max * day / (ramp_weeks * 7))
        - SPR: 0 if day < spr_delay_weeks*7, else spr_rate_mean
        - Reroute: ~0.5 mbd/week from W2, cap 2.0 mbd
        - Surplus: always available from D1
        Total capped at 10 mbd.
        """
        ramp_days = self.pipeline_ramp_weeks * 7

        # Pipeline ramp
        if ramp_days > 0 and day > 0:
            pipeline = min(self.pipeline_max, self.pipeline_max * day / ramp_days)
        else:
            pipeline = 0.0

        # SPR with delay
        spr_delay_days = self.spr_delay_weeks * 7
        spr = self.spr_rate_mean if day >= spr_delay_days else 0.0

        # Reroute: gradual from W2 (day 14), ~0.5 mbd/week, cap 2.0
        reroute_start = 14
        reroute_rate_per_day = 0.5 / 7  # 0.5 mbd per week
        reroute_cap = 2.0
        if day >= reroute_start:
            reroute = min(reroute_cap, reroute_rate_per_day * (day - reroute_start))
        else:
            reroute = 0.0

        # Surplus: always available (even day 0)
        surplus = self.surplus_buffer

        total = pipeline + spr + reroute + surplus
        return min(total, 10.0)
