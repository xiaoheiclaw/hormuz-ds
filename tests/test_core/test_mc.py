import numpy as np
import pytest
from hormuz.core.types import ACHPosterior, Parameters, PathWeights


def test_mc_basic():
    from hormuz.core.mc import run_monte_carlo, MCResult
    posterior = ACHPosterior(h1=0.5, h2=0.5, h3=None)
    params = Parameters()
    result = run_monte_carlo(posterior, params, events={}, n=100, seed=42)
    assert isinstance(result, MCResult)
    assert len(result.t_samples) == 100
    assert len(result.total_gap_samples) == 100


def test_mc_path_classification():
    """T<35->A, 35-120->B, >120->C"""
    from hormuz.core.mc import classify_paths
    t_samples = np.array([20, 30, 50, 80, 130, 200])
    counts = classify_paths(t_samples)
    assert counts["A"] == 2
    assert counts["B"] == 2
    assert counts["C"] == 2


def test_mc_percentiles():
    from hormuz.core.mc import run_monte_carlo
    posterior = ACHPosterior(h1=0.5, h2=0.5, h3=None)
    params = Parameters()
    result = run_monte_carlo(posterior, params, events={}, n=500, seed=42)
    assert "p10" in result.t_percentiles
    assert "p90" in result.t_percentiles
    assert result.t_percentiles["p10"] < result.t_percentiles["p90"]


def test_mc_h1_dominant_shorter():
    """H1 dominant should produce shorter T distribution"""
    from hormuz.core.mc import run_monte_carlo
    params = Parameters()
    r_h1 = run_monte_carlo(ACHPosterior(h1=0.8, h2=0.2, h3=None), params, {}, n=500, seed=42)
    r_h2 = run_monte_carlo(ACHPosterior(h1=0.2, h2=0.8, h3=None), params, {}, n=500, seed=42)
    assert np.median(r_h1.t_samples) < np.median(r_h2.t_samples)


def test_mc_result_has_path_gaps():
    from hormuz.core.mc import run_monte_carlo
    posterior = ACHPosterior(h1=0.5, h2=0.5, h3=None)
    params = Parameters()
    result = run_monte_carlo(posterior, params, events={}, n=100, seed=42)
    assert "A" in result.path_total_gap_means
    assert "B" in result.path_total_gap_means
    assert "C" in result.path_total_gap_means


def test_mc_reproducible():
    from hormuz.core.mc import run_monte_carlo
    posterior = ACHPosterior(h1=0.5, h2=0.5, h3=None)
    params = Parameters()
    r1 = run_monte_carlo(posterior, params, {}, n=100, seed=42)
    r2 = run_monte_carlo(posterior, params, {}, n=100, seed=42)
    np.testing.assert_array_equal(r1.t_samples, r2.t_samples)
