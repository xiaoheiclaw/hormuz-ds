"""YAML variable loader — pure functions mapping config files to typed models."""

from __future__ import annotations

from pathlib import Path

import yaml

from hormuz.core.types import CalibrationRef, Constants, Parameters


def load_constants(path: Path) -> Constants:
    """Parse constants.yaml into Constants model."""
    data = yaml.safe_load(path.read_text())
    phys = data["physical"]
    return Constants(
        exposed_supply_mbd=20.1,  # hardcoded per PRD
        strait_width_km=phys["q1"]["C2"]["value_km"],
        mine_type_mix=phys["q2"]["C4"]["types"],
    )


def load_parameters(path: Path) -> Parameters:
    """Parse parameters.yaml into Parameters model."""
    data = yaml.safe_load(path.read_text())
    phys = data["physical"]
    q2 = phys["q2"]
    q3 = phys["q3"]
    return Parameters(
        gross_gap_mbd=phys["gross_gap_mbd"],
        mines_in_water_range=tuple(q2["sea_mines_range"]),
        sweep_ships=q2["sweep_ships_available"],
        pipeline_max_mbd=q3["pipeline_max_mbd"],
        pipeline_ramp_weeks=q3["pipeline_ramp_weeks"],
        spr_rate_mean_mbd=q3["spr_rate_mean_mbd"],
        spr_pump_min_days=int(q3["spr_delay_weeks"] * 7 / 1.346),  # ~13 days
        h3_suspended=phys["q1"]["h3_suspended"],
        h3_prior=phys["q1"]["h3_prior"],
        effective_disruption_rate=phys["effective_disruption_rate"],
    )


def load_calibration_refs(path: Path) -> list[CalibrationRef]:
    """Parse calibration.references from constants.yaml."""
    data = yaml.safe_load(path.read_text())
    return [
        CalibrationRef(**ref)
        for ref in data["calibration"]["references"]
    ]
