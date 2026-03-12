from pathlib import Path
import pytest

CONFIGS = Path(__file__).parents[2] / "configs"


def test_load_constants():
    from hormuz.core.variables import load_constants
    c = load_constants(CONFIGS / "constants.yaml")
    assert c.strait_width_km == 9.0
    assert len(c.mine_type_mix) == 3


def test_load_parameters():
    from hormuz.core.variables import load_parameters
    p = load_parameters(CONFIGS / "parameters.yaml")
    assert p.gross_gap_mbd == 16.0
    assert p.mines_in_water_range == (20, 100)
    assert p.spr_pump_min_days == 13


def test_load_calibration_refs():
    from hormuz.core.variables import load_calibration_refs
    refs = load_calibration_refs(CONFIGS / "constants.yaml")
    assert len(refs) == 3
    assert refs[0].year == 1988
