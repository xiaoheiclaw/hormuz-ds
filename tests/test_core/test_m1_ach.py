import pytest
from datetime import datetime
from hormuz.core.types import Observation, ACHPosterior, Parameters


def make_obs(obs_id: str, value: float) -> Observation:
    return Observation(id=obs_id, timestamp=datetime(2026, 3, 12), value=value, source="test")


def test_prior_with_h3_suspended():
    from hormuz.core.m1_ach import compute_prior
    prior = compute_prior(h3_suspended=True, h3_prior=0.10)
    assert abs(prior["H1"] - 0.475) < 1e-6
    assert abs(prior["H2"] - 0.475) < 1e-6
    assert prior.get("H3") is None


def test_prior_with_h3_active():
    from hormuz.core.m1_ach import compute_prior
    prior = compute_prior(h3_suspended=False, h3_prior=0.10)
    assert abs(prior["H1"] - 0.45) < 1e-6
    assert abs(prior["H3"] - 0.10) < 1e-6


def test_likelihood_ratio_basic():
    """O04 ammo substitution: high value = depletion (H1)"""
    from hormuz.core.m1_ach import get_likelihood_ratio
    lr = get_likelihood_ratio("O04", value=0.9, context={})
    assert lr["H1"] > lr["H2"]


def test_t1a_t1b_unbinding():
    """GPS spoofing + attack frequency co-occurrence"""
    from hormuz.core.m1_ach import get_likelihood_ratio
    # T1a: GPS up + frequency up -> offensive H2
    lr_a = get_likelihood_ratio("O05", value=0.8, context={"O01_trend": "rising"})
    assert lr_a["H2"] == 5.0
    # T1b: GPS up + frequency down -> defensive H2
    lr_b = get_likelihood_ratio("O05", value=0.8, context={"O01_trend": "falling"})
    assert lr_b["H2"] == 2.0
    # GPS down -> H1
    lr_c = get_likelihood_ratio("O05", value=0.2, context={})
    assert lr_c["H1"] == 3.0


def test_bayesian_update_single():
    from hormuz.core.m1_ach import bayesian_update
    prior = {"H1": 0.475, "H2": 0.475}
    lr = {"H1": 5.0, "H2": 1.0}
    posterior = bayesian_update(prior, lr)
    assert posterior["H1"] > 0.8  # strong H1 evidence


def test_bayesian_update_multiple():
    """Multiple H1 evidence should push posterior strongly to H1"""
    from hormuz.core.m1_ach import run_ach
    obs = [
        make_obs("O02", 0.9),  # attack freq decline (H1)
        make_obs("O03", 0.9),  # coordination degrading (H1)
        make_obs("O04", 0.9),  # ammo substitution high (H1)
    ]
    result = run_ach(obs, h3_suspended=True, h3_prior=0.10)
    assert isinstance(result, ACHPosterior)
    assert result.h1 > 0.7
    assert result.dominant == "H1"


def test_decay_rate_mapping():
    from hormuz.core.m1_ach import map_to_decay_rate
    # H1 dominant -> high decay
    assert map_to_decay_rate(ACHPosterior(h1=0.8, h2=0.2, h3=None)) > 0.05
    # H2 dominant -> low decay
    assert map_to_decay_rate(ACHPosterior(h1=0.2, h2=0.8, h3=None)) < 0.04
