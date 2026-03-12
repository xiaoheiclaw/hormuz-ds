import numpy as np
import pytest
from hormuz.core.types import ACHPosterior, Parameters


def test_t1_h1_dominant():
    """H1 dominant -> T1 median ~2-3 weeks"""
    from hormuz.core.m2_duration import estimate_t1
    posterior = ACHPosterior(h1=0.8, h2=0.2, h3=None)
    samples = estimate_t1(posterior, n=1000, seed=42)
    median = np.median(samples)
    assert 10 <= median <= 25  # 2-3 weeks in days


def test_t1_h2_dominant():
    """H2 dominant -> T1 median ~5-7 weeks"""
    from hormuz.core.m2_duration import estimate_t1
    posterior = ACHPosterior(h1=0.2, h2=0.8, h3=None)
    samples = estimate_t1(posterior, n=1000, seed=42)
    median = np.median(samples)
    assert 30 <= median <= 55  # 5-7 weeks in days


def test_t2_basic():
    """T2 with default params, no events"""
    from hormuz.core.m2_duration import estimate_t2
    params = Parameters()
    samples = estimate_t2(params, events={}, n=1000, seed=42)
    median = np.median(samples)
    assert 20 <= median <= 60  # ~5 weeks median


def test_t2_event_e3():
    """E3 (mine strike) adds 7 days"""
    from hormuz.core.m2_duration import estimate_t2
    params = Parameters()
    s_no_event = estimate_t2(params, events={}, n=1000, seed=42)
    s_e3 = estimate_t2(params, events={"E3": True}, n=1000, seed=42)
    assert np.median(s_e3) > np.median(s_no_event) + 5


def test_t2_event_c2():
    """C2 (re-mining cleared lanes) adds 21 days"""
    from hormuz.core.m2_duration import estimate_t2
    params = Parameters()
    s_c2 = estimate_t2(params, events={"C2": True}, n=1000, seed=42)
    s_no = estimate_t2(params, events={}, n=1000, seed=42)
    assert np.median(s_c2) > np.median(s_no) + 15


def test_t_total_convolution():
    """T = T1 + deployment_gap + T2"""
    from hormuz.core.m2_duration import estimate_t_total
    posterior = ACHPosterior(h1=0.5, h2=0.5, h3=None)
    params = Parameters()
    t1, t2, t_total = estimate_t_total(posterior, params, events={}, n=1000, seed=42)
    assert len(t_total) == 1000
    # T_total should be > T1 + 7 (min deployment gap)
    assert np.min(t_total) >= np.min(t1) + 7


def test_percentiles():
    from hormuz.core.m2_duration import compute_percentiles
    samples = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    pct = compute_percentiles(samples)
    assert "p10" in pct and "p50" in pct and "p90" in pct
