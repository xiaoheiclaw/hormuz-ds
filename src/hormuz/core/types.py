"""PRD §2 variable taxonomy + §7 SystemOutput — all Pydantic models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, model_validator


# ── C: Constants (frozen, never change at runtime) ────────────────────

class Constants(BaseModel, frozen=True):
    """C01-C05: physical constants from PRD §2.1."""
    exposed_supply_mbd: float = 20.1       # C01: normal Strait flow
    strait_width_km: float = 9.0           # C02: navigable channel width
    sweep_area_description: str = "total area requiring mine clearance"  # C03
    mine_type_mix: list[str] = ["contact", "magnetic", "acoustic"]      # C04
    single_ship_sweep_ceiling: str = "max area one MCM vessel clears/day"  # C05


# ── P: Parameters (tunable, updated via calibration or override) ──────

class Parameters(BaseModel):
    """P01-P10: tunable parameters from PRD §2.2."""
    gross_gap_mbd: float = 16.0                     # P01: 20.1 × 0.80
    mines_in_water_range: tuple[int, int] = (20, 100)  # P02: Uniform draw
    sweep_ships: int = 4                             # P03: post-Avenger decom
    pipeline_max_mbd: float = 4.0                    # P04
    pipeline_ramp_weeks: float = 2.5                 # P05
    spr_rate_mean_mbd: float = 2.0                   # P06: aligned with config
    spr_pump_min_days: int = 13                      # P07: ~2.5 weeks hard delay
    h3_suspended: bool = True                        # P08
    h3_prior: float = 0.10                           # P09
    effective_disruption_rate: float = 0.80           # P10
    regime_jump_rate: float = 0.08                    # tail scenario injection rate

    def override(self, **kwargs: object) -> Parameters:
        """Return new instance with overridden fields."""
        return self.model_copy(update=kwargs)


# ── S: State vector (runtime mutable) ────────────────────────────────

class StateVector(BaseModel):
    """S01-S11: runtime state variables."""
    disruption_rate: float = 0.80       # S01
    mines_in_water: int = 50            # S02
    mines_cleared: int = 0              # S03
    buffer_mbd: float = 0.0             # S04
    transit_count: int = 0              # S05
    attack_freq_per_week: float = 0.0   # S06
    brent_price: float = 0.0            # S07
    ovx: float = 0.0                    # S08
    ap_pct: float = 0.0                 # S09
    freight_ws: float = 0.0             # S10
    effective_disruption: float = 0.80  # S11


# ── O: Observation (external data point) ──────────────────────────────

class Observation(BaseModel):
    """Single observation from external source."""
    id: str
    timestamp: datetime
    value: float
    source: str
    noise_note: Optional[str] = None


# ── D: Control (actor decision) ───────────────────────────────────────

class Control(BaseModel):
    """D01-D05: actor decisions / control variables."""
    id: str
    actor: str
    triggered: bool = False
    trigger_time: Optional[datetime] = None
    effect: Optional[str] = None


# ── Calibration reference ─────────────────────────────────────────────

class CalibrationRef(BaseModel):
    """Historical calibration baseline."""
    name: str
    year: int
    description: str
    relevance: str


# ── ACH Posterior ─────────────────────────────────────────────────────

class ACHPosterior(BaseModel):
    """Posterior probabilities from M1 ACH engine."""
    h1: float
    h2: float
    h3: Optional[float] = None

    @property
    def dominant(self) -> str:
        """Return dominant hypothesis if P > 0.7, else 'inconclusive'."""
        if self.h1 >= 0.7:
            return "H1"
        if self.h2 >= 0.7:
            return "H2"
        if self.h3 is not None and self.h3 >= 0.7:
            return "H3"
        return "inconclusive"


# ── Path weights ──────────────────────────────────────────────────────

class PathWeights(BaseModel):
    """A/B/C path probabilities — always normalized, clipped [0.05, 0.85]."""
    a: float = 0.30
    b: float = 0.50
    c: float = 0.20

    def normalized(self) -> PathWeights:
        """Return new instance: sum=1.0, each clipped to [0.05, 0.85]."""
        total = self.a + self.b + self.c
        if total == 0:
            return PathWeights(a=1 / 3, b=1 / 3, c=1 / 3)
        vals = [self.a / total, self.b / total, self.c / total]
        # Clip then redistribute residual to unclamped values
        for _ in range(20):
            clamped = [max(0.05, min(0.85, v)) for v in vals]
            residual = 1.0 - sum(clamped)
            if abs(residual) < 1e-12:
                break
            free = [i for i in range(3) if 0.05 < clamped[i] < 0.85]
            if not free:
                # All at boundaries — distribute residual to boundary values
                # proportionally, relaxing clip to maintain sum=1.0
                if residual > 0:
                    # Need to increase: give to floor values (they have room to grow)
                    floor_ids = [i for i in range(3) if clamped[i] <= 0.05]
                    targets = floor_ids if floor_ids else list(range(3))
                else:
                    # Need to decrease: take from ceiling values
                    ceil_ids = [i for i in range(3) if clamped[i] >= 0.85]
                    targets = ceil_ids if ceil_ids else list(range(3))
                share = residual / len(targets)
                clamped = [clamped[i] + (share if i in targets else 0) for i in range(3)]
                break
            share = residual / len(free)
            vals = [clamped[i] + (share if i in free else 0) for i in range(3)]
        else:
            clamped = vals
        return PathWeights(a=clamped[0], b=clamped[1], c=clamped[2])


# ── System output (PRD §7.1) ─────────────────────────────────────────

class SystemOutput(BaseModel):
    """Complete engine output — one per pipeline run."""
    timestamp: datetime
    ach_posterior: ACHPosterior
    t1_percentiles: dict[str, float]
    t2_percentiles: dict[str, float]
    t_total_percentiles: dict[str, float]
    t_weighted_mean: float = 0.0                # path-weighted expected T
    buffer_trajectory: list[tuple[int, float]]
    gross_gap_mbd: float
    net_gap_trajectories: dict[str, list[tuple[int, float]]]
    path_probabilities: PathWeights
    path_total_gaps: dict[str, float]
    expected_total_gap: float
    consistency_flags: list[str]
    confidence_level: str = "normal"  # "burn_in" (<3d) / "low" (3-7d) / "normal" (>7d)
