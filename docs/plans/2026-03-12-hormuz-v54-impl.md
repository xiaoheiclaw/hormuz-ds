# Hormuz Decision OS v5.4 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean rewrite of the Hormuz crisis decision system based on v5.4 PRD — M1-M5 Bayesian engine, MC simulation (N=10000), data ingestion, CLI, and HTML dashboard with parameter documentation.

**Architecture:** Three-layer separation — `core/` (pure compute, zero IO) + `infra/` (DB, network, LLM) + `app/` (pipeline, CLI, reporting). Core modules map 1:1 to PRD M1-M5. All existing code under `src/hormuz/` is deleted and rewritten.

**Tech Stack:** Python 3.12+, Pydantic 2, numpy, scipy, SQLite, Jinja2, matplotlib, yfinance, httpx, Click

**Design Doc:** `docs/plans/2026-03-12-hormuz-v54-reimpl-design.md`
**PRD Reference:** `docs/ref-v54-backend-prd.md`

---

## Task 0: Scaffold — Clean slate and project structure

**Files:**
- Delete: `src/hormuz/` (entire existing package)
- Create: new package structure under `src/hormuz/`
- Modify: `pyproject.toml` (add numpy, scipy)
- Modify: `.structure.yml` (update directory whitelist)

**Step 1: Delete existing source and tests**

```bash
rm -rf src/hormuz tests
```

**Step 2: Create new directory structure**

```bash
mkdir -p src/hormuz/core src/hormuz/infra src/hormuz/app
mkdir -p tests/test_core tests/test_infra tests/test_app
touch src/hormuz/__init__.py src/hormuz/core/__init__.py src/hormuz/infra/__init__.py src/hormuz/app/__init__.py
touch tests/__init__.py tests/test_core/__init__.py tests/test_infra/__init__.py tests/test_app/__init__.py
touch tests/conftest.py
```

**Step 3: Update pyproject.toml**

Add numpy and scipy to dependencies. Update CLI entry point to `hormuz.app.cli:cli`. Update hatch packages path (still `src/hormuz`).

```toml
dependencies = [
    "pydantic>=2.0",
    "numpy>=1.26",
    "scipy>=1.12",
    "jinja2>=3.1",
    "matplotlib>=3.8",
    "yfinance>=0.2",
    "httpx>=0.27",
    "click>=8.1",
    "pyyaml>=6.0",
    "python-dotenv>=1.2.2",
]

[project.scripts]
hormuz = "hormuz.app.cli:cli"
```

**Step 4: Update .structure.yml**

Replace `src/hormuz/engine/` and `src/hormuz/llm/` entries with:

```yaml
directories:
  src/hormuz/core/:
    description: "Pure computation — zero IO, zero side effects"
    allow: ["*.py"]
  src/hormuz/infra/:
    description: "IO layer — DB, network, LLM"
    allow: ["*.py"]
  src/hormuz/app/:
    description: "Product layer — pipeline, CLI, reporting"
    allow: ["*.py"]
  # ... keep configs/, tests/, templates/, data/, reports/, docs/
```

Update `tests/` to allow subdirectories.

**Step 5: Install dependencies**

```bash
cd ~/Projects/hormuz-ds && uv sync
```

**Step 6: Verify clean state**

```bash
python -c "import hormuz; print('ok')"
pytest tests/ --co -q  # should collect 0 tests
```

**Step 7: Commit**

```bash
git add -A && git commit -m "chore: clean slate — delete old code, scaffold v5.4 structure"
```

---

## Task 1: Data contracts — core/types.py

**Files:**
- Create: `src/hormuz/core/types.py`
- Test: `tests/test_core/test_types.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_types.py
import pytest
from datetime import datetime

def test_constants_immutable():
    from hormuz.core.types import Constants
    c = Constants()
    assert c.exposed_supply_mbd == 20.1
    assert c.strait_width_km == 9.0

def test_parameters_defaults():
    from hormuz.core.types import Parameters
    p = Parameters()
    assert p.mines_in_water_range == (20, 100)
    assert p.spr_pump_min_days == 13
    assert p.pipeline_max_mbd == 4.0

def test_observation():
    from hormuz.core.types import Observation
    o = Observation(id="O01", timestamp=datetime(2026, 3, 12), value=3.5, source="CENTCOM")
    assert o.id == "O01"

def test_state_vector_defaults():
    from hormuz.core.types import StateVector
    sv = StateVector()
    assert sv.disruption_rate == 0.80
    assert sv.buffer_mbd == 0.0

def test_control():
    from hormuz.core.types import Control
    c = Control(id="D01", actor="US_NAVY", triggered=False)
    assert not c.triggered

def test_ach_posterior_validation():
    from hormuz.core.types import ACHPosterior
    p = ACHPosterior(h1=0.6, h2=0.4, h3=None)
    assert p.dominant == "inconclusive"
    p2 = ACHPosterior(h1=0.75, h2=0.25, h3=None)
    assert p2.dominant == "H1"

def test_path_weights_normalize():
    from hormuz.core.types import PathWeights
    pw = PathWeights(a=0.50, b=0.40, c=0.30)
    pw = pw.normalized()
    assert abs(pw.a + pw.b + pw.c - 1.0) < 1e-9

def test_path_weights_clip():
    from hormuz.core.types import PathWeights
    pw = PathWeights(a=0.95, b=0.04, c=0.01)
    pw = pw.normalized()
    assert pw.a <= 0.85
    assert pw.c >= 0.05

def test_system_output():
    from hormuz.core.types import SystemOutput, ACHPosterior, PathWeights
    so = SystemOutput(
        timestamp=datetime(2026, 3, 12),
        ach_posterior=ACHPosterior(h1=0.5, h2=0.5, h3=None),
        t1_percentiles={"p50": 21},
        t2_percentiles={"p50": 35},
        t_total_percentiles={"p50": 63},
        buffer_trajectory=[(0, 0.0), (14, 1.5), (30, 7.0)],
        gross_gap_mbd=16.0,
        net_gap_trajectories={"A": [(0, 16.0), (14, 14.5)]},
        path_probabilities=PathWeights(),
        path_total_gaps={"A": 270.0, "B": 833.0, "C": 2500.0},
        expected_total_gap=700.0,
        consistency_flags=[],
    )
    assert so.gross_gap_mbd == 16.0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_core/test_types.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'hormuz.core.types'`

**Step 3: Write implementation**

`src/hormuz/core/types.py` — All Pydantic models:

- `Constants` (frozen=True): C01-C05 with defaults from PRD §2.1
- `Parameters`: P01-P10 with defaults from PRD §2.2, mutable via `.override()` returning new instance
- `Observation`: id, timestamp, value, source, noise_note
- `StateVector`: S01-S11, all floats with sensible defaults (disruption_rate=0.80, buffer_mbd=0.0)
- `Control`: D01-D05, id, actor, triggered, trigger_time, effect
- `CalibrationRef`: name, year, description, relevance
- `ACHPosterior`: h1, h2, h3 (Optional), computed `dominant` property (>0.7 threshold)
- `PathWeights`: a, b, c with `normalized()` method (sum=1.0, clip [0.05, 0.85])
- `SystemOutput`: PRD §7.1 complete dataclass

**Step 4: Run tests**

```bash
pytest tests/test_core/test_types.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hormuz/core/types.py tests/test_core/test_types.py
git commit -m "feat(core): add data contracts — types.py with PRD §2 variable models"
```

---

## Task 2: Variable loader — core/variables.py

**Files:**
- Create: `src/hormuz/core/variables.py`
- Test: `tests/test_core/test_variables.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_variables.py
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_core/test_variables.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/hormuz/core/variables.py` — YAML parsing functions:

- `load_constants(path) -> Constants`: parse `physical.q1.C2.value_km`, `physical.q2.C4.types`, etc. into Constants model
- `load_parameters(path) -> Parameters`: parse all P01-P10 fields into Parameters model
- `load_calibration_refs(path) -> list[CalibrationRef]`: parse `calibration.references` list

Pure functions, no state. Uses `yaml.safe_load()`.

**Step 4: Run tests**

```bash
pytest tests/test_core/test_variables.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hormuz/core/variables.py tests/test_core/test_variables.py
git commit -m "feat(core): add YAML variable loader — constants + parameters + calibration refs"
```

---

## Task 3: M1 ACH Bayesian engine — core/m1_ach.py

**Files:**
- Create: `src/hormuz/core/m1_ach.py`
- Test: `tests/test_core/test_m1_ach.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_m1_ach.py
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_core/test_m1_ach.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/hormuz/core/m1_ach.py`:

- `compute_prior(h3_suspended, h3_prior) -> dict`: PRD §3.2 prior logic
- `get_likelihood_ratio(obs_id, value, context) -> dict`: Likelihood table from PRD §3.2, LR ∈ {0.2, 0.5, 1.0, 2.0, 5.0}. T1a/T1b unbinding per `context["O01_trend"]`
- `bayesian_update(prior, lr) -> dict`: Single-step Bayes update + normalize
- `run_ach(observations, h3_suspended, h3_prior) -> ACHPosterior`: Iterate over observations, apply sequential Bayes updates, return posterior
- `map_to_decay_rate(posterior) -> float`: Linear interpolation in [0.02, 0.08] based on P(H1)

All pure functions. The likelihood table is a dict mapping `(obs_id, direction)` to LR values, defined as module-level constant.

**Step 4: Run tests**

```bash
pytest tests/test_core/test_m1_ach.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hormuz/core/m1_ach.py tests/test_core/test_m1_ach.py
git commit -m "feat(core): add M1 ACH Bayesian inference engine"
```

---

## Task 4: M2 T distribution estimator — core/m2_duration.py

**Files:**
- Create: `src/hormuz/core/m2_duration.py`
- Test: `tests/test_core/test_m2_duration.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_m2_duration.py
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_core/test_m2_duration.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/hormuz/core/m2_duration.py`:

- `estimate_t1(posterior, n, seed) -> np.ndarray`: Sample T1 from mixture of two lognormals. P(H1) weight → lognormal(mu_h1, sigma_h1), P(H2) weight → lognormal(mu_h2, sigma_h2). Parameters: H1 median ~17 days (2.5w), H2 median ~42 days (6w).
- `estimate_t2(params, events, n, seed) -> np.ndarray`: Sample mines_in_water ~ Uniform(20,100). Compute sweep time = mines / (ships × rate). Add mine_type_penalty. Apply event jumps: E3→+7, C2→+21, E2→+14.
- `estimate_t_total(posterior, params, events, n, seed) -> tuple[ndarray, ndarray, ndarray]`: T1 + Uniform(7,14) + T2 per sample.
- `compute_percentiles(samples) -> dict`: {p10, p25, p50, p75, p90}

Uses `numpy.random.Generator` with explicit seed for reproducibility.

**Step 4: Run tests**

```bash
pytest tests/test_core/test_m2_duration.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hormuz/core/m2_duration.py tests/test_core/test_m2_duration.py
git commit -m "feat(core): add M2 T distribution estimator — lognormal T1 + MC T2"
```

---

## Task 5: M3 Buffer ramp function — core/m3_buffer.py

**Files:**
- Create: `src/hormuz/core/m3_buffer.py`
- Test: `tests/test_core/test_m3_buffer.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_m3_buffer.py
import pytest
from hormuz.core.types import Parameters, Control

def test_buffer_day_0():
    from hormuz.core.m3_buffer import compute_buffer
    params = Parameters()
    assert compute_buffer(day=0, params=params) == pytest.approx(0.0, abs=0.1)

def test_buffer_day_2():
    """Before ADCOP kicks in"""
    from hormuz.core.m3_buffer import compute_buffer
    params = Parameters()
    assert compute_buffer(day=2, params=params) == pytest.approx(0.0, abs=0.1)

def test_buffer_day_10():
    """During pipeline ramp"""
    from hormuz.core.m3_buffer import compute_buffer
    params = Parameters()
    b = compute_buffer(day=10, params=params)
    assert 0.5 < b < 3.0  # partial ramp

def test_buffer_day_30():
    """Steady state"""
    from hormuz.core.m3_buffer import compute_buffer
    params = Parameters()
    b = compute_buffer(day=30, params=params)
    assert 5.0 < b < 9.0  # ~7 mbd steady state

def test_pipeline_component():
    from hormuz.core.m3_buffer import pipeline_buffer
    params = Parameters()
    assert pipeline_buffer(day=2, params=params) == pytest.approx(0.0, abs=0.01)
    assert pipeline_buffer(day=4, params=params) > 0  # ADCOP starting
    p14 = pipeline_buffer(day=14, params=params)
    assert p14 > 2.0  # ADCOP + Saudi pipeline near steady

def test_spr_component_no_trigger():
    from hormuz.core.m3_buffer import spr_buffer
    params = Parameters()
    assert spr_buffer(day=20, params=params, spr_trigger_day=None) == 0.0

def test_spr_component_with_trigger():
    from hormuz.core.m3_buffer import spr_buffer
    params = Parameters()
    # Triggered day 1, 13 day delay, so arrives ~day 14
    assert spr_buffer(day=10, params=params, spr_trigger_day=1) == 0.0
    assert spr_buffer(day=20, params=params, spr_trigger_day=1) > 0.5

def test_cape_component():
    from hormuz.core.m3_buffer import cape_buffer
    assert cape_buffer(day=10) == pytest.approx(0.0, abs=0.01)
    assert cape_buffer(day=15) > 0  # first arrivals
    assert cape_buffer(day=60) > 1.0  # steady

def test_buffer_trajectory():
    """Generate full trajectory"""
    from hormuz.core.m3_buffer import compute_buffer_trajectory
    params = Parameters()
    traj = compute_buffer_trajectory(max_day=90, params=params)
    assert len(traj) == 91  # day 0..90
    assert traj[0][1] == pytest.approx(0.0, abs=0.1)
    assert traj[-1][1] > 5.0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_core/test_m3_buffer.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/hormuz/core/m3_buffer.py`:

- `pipeline_buffer(day, params, pipeline_override=None) -> float`: ADCOP ramp (D3-5) + Saudi pipeline ramp (D5-14) + steady state. O11 校验 via `pipeline_override`.
- `spr_buffer(day, params, spr_trigger_day=None, spr_override=None) -> float`: 13-day hard delay + linear ramp to spr_rate_mean. O13 校验 via `spr_override`.
- `cape_buffer(day) -> float`: D14 first arrival + D21 gradual increase to ~1.5 mbd.
- `compute_buffer(day, params, spr_trigger_day=None, pipeline_override=None, spr_override=None) -> float`: Sum of three components.
- `compute_buffer_trajectory(max_day, params, ...) -> list[tuple[int, float]]`: Buffer(t) for t in [0, max_day].

Helper: `ramp(start_val, end_val, t, duration) -> float` — linear interpolation clamped.

All pure functions.

**Step 4: Run tests**

```bash
pytest tests/test_core/test_m3_buffer.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hormuz/core/m3_buffer.py tests/test_core/test_m3_buffer.py
git commit -m "feat(core): add M3 Buffer ramp function — pipeline + SPR + cape reroute"
```

---

## Task 6: M4 Gap integrator — core/m4_gap.py

**Files:**
- Create: `src/hormuz/core/m4_gap.py`
- Test: `tests/test_core/test_m4_gap.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_m4_gap.py
import pytest
from hormuz.core.types import Constants, StateVector

def test_gross_gap():
    from hormuz.core.m4_gap import compute_gross_gap
    c = Constants()
    sv = StateVector()
    assert compute_gross_gap(c, sv) == pytest.approx(16.08, abs=0.1)

def test_net_gap_day_0():
    from hormuz.core.m4_gap import compute_net_gap
    assert compute_net_gap(gross_gap=16.0, buffer=0.0) == 16.0

def test_net_gap_day_30():
    from hormuz.core.m4_gap import compute_net_gap
    assert compute_net_gap(gross_gap=16.0, buffer=7.0) == 9.0

def test_total_gap_path_a():
    """Path A: T~28 days -> ~270 mbd·days"""
    from hormuz.core.m4_gap import integrate_total_gap
    buffer_traj = [(d, 0.0 if d < 3 else 1.5 if d < 14 else 7.0) for d in range(29)]
    tg = integrate_total_gap(gross_gap=16.0, buffer_trajectory=buffer_traj, t_end=28)
    assert 200 < tg < 350

def test_total_gap_path_b():
    """Path B: T~84 days -> ~833 mbd·days"""
    from hormuz.core.m4_gap import integrate_total_gap
    buffer_traj = [(d, 0.0 if d < 3 else 1.5 if d < 14 else 7.0) for d in range(85)]
    tg = integrate_total_gap(gross_gap=16.0, buffer_trajectory=buffer_traj, t_end=84)
    assert 700 < tg < 950

def test_net_gap_trajectory():
    from hormuz.core.m4_gap import compute_net_gap_trajectory
    buffer_traj = [(0, 0.0), (14, 1.5), (30, 7.0)]
    traj = compute_net_gap_trajectory(gross_gap=16.0, buffer_trajectory=buffer_traj)
    assert traj[0][1] == pytest.approx(16.0)
    assert traj[1][1] == pytest.approx(14.5)
    assert traj[2][1] == pytest.approx(9.0)

def test_path_total_gaps():
    """Compute TotalGap for all three paths"""
    from hormuz.core.m4_gap import compute_path_total_gaps
    from hormuz.core.types import Parameters
    params = Parameters()
    gaps = compute_path_total_gaps(gross_gap=16.0, params=params)
    assert "A" in gaps and "B" in gaps and "C" in gaps
    assert gaps["A"] < gaps["B"] < gaps["C"]
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_core/test_m4_gap.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/hormuz/core/m4_gap.py`:

- `compute_gross_gap(constants, state) -> float`: C05 × S11
- `compute_net_gap(gross_gap, buffer) -> float`: gross_gap - buffer
- `compute_net_gap_trajectory(gross_gap, buffer_trajectory) -> list[tuple[int, float]]`: NetGap at each day
- `integrate_total_gap(gross_gap, buffer_trajectory, t_end) -> float`: Σ segments [days × NetGap]. Uses trapezoidal or piecewise constant integration.
- `compute_path_total_gaps(gross_gap, params) -> dict[str, float]`: Shortcut for three standard paths using PRD buffer ramp breakpoints.

**Step 4: Run tests**

```bash
pytest tests/test_core/test_m4_gap.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hormuz/core/m4_gap.py tests/test_core/test_m4_gap.py
git commit -m "feat(core): add M4 gap integrator — net gap trajectory + total gap per path"
```

---

## Task 7: M5 Game theory path adjuster — core/m5_game.py

**Files:**
- Create: `src/hormuz/core/m5_game.py`
- Test: `tests/test_core/test_m5_game.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_m5_game.py
import pytest
from hormuz.core.types import PathWeights

def test_no_signals():
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    result = adjust_path_weights(base, active_signals=[])
    assert result.a == pytest.approx(0.30)

def test_mediation_signal():
    """D03 mediation -> A+=0.15, C-=0.10, B-=0.05"""
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    result = adjust_path_weights(base, active_signals=["mediation"])
    assert result.a > base.a
    assert result.c < base.c

def test_escalation_signal():
    """E1 target spillover -> C+=0.15, A-=0.10, B-=0.05"""
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    result = adjust_path_weights(base, active_signals=["escalation"])
    assert result.c > base.c
    assert result.a < base.a

def test_multiple_signals():
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    result = adjust_path_weights(base, active_signals=["mediation", "commitment_softening"])
    assert result.a > base.a + 0.1

def test_always_normalized():
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    for signals in [["mediation"], ["escalation"], ["mediation", "escalation"]]:
        result = adjust_path_weights(base, active_signals=signals)
        assert abs(result.a + result.b + result.c - 1.0) < 1e-9

def test_clip_bounds():
    """Extreme signals should still clip to [0.05, 0.85]"""
    from hormuz.core.m5_game import adjust_path_weights
    base = PathWeights()
    result = adjust_path_weights(base, active_signals=["mediation", "commitment_softening", "commitment_lock"])
    assert result.a <= 0.85
    assert result.c >= 0.05
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_core/test_m5_game.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/hormuz/core/m5_game.py`:

- Signal delta table (module-level dict):
  - `"mediation"` → {a: +0.15, b: -0.05, c: -0.10}
  - `"commitment_softening"` → {a: +0.10, b: -0.10, c: 0}
  - `"commitment_lock"` → {a: +0.05, b: -0.05, c: 0}
  - `"escalation"` → {a: -0.10, b: -0.05, c: +0.15}
- `adjust_path_weights(base, active_signals) -> PathWeights`: Apply deltas sequentially, normalize + clip after each.

**Step 4: Run tests**

```bash
pytest tests/test_core/test_m5_game.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hormuz/core/m5_game.py tests/test_core/test_m5_game.py
git commit -m "feat(core): add M5 game theory path weight adjuster"
```

---

## Task 8: Monte Carlo simulation — core/mc.py

**Files:**
- Create: `src/hormuz/core/mc.py`
- Test: `tests/test_core/test_mc.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_mc.py
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_core/test_mc.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/hormuz/core/mc.py`:

```python
@dataclass
class MCResult:
    t_samples: np.ndarray           # shape (N,)
    total_gap_samples: np.ndarray   # shape (N,)
    t_percentiles: dict             # {p10, p25, p50, p75, p90}
    path_frequencies: dict          # {"A": float, "B": float, "C": float}
    path_total_gap_means: dict      # per-path average TotalGap
```

- `run_monte_carlo(posterior, params, events, n=10000, seed=None) -> MCResult`:
  1. Create `numpy.random.Generator(seed)`
  2. For each of N rounds: sample 6 parameters, call `estimate_t1`, `estimate_t2`, `compute_buffer_trajectory`, `integrate_total_gap`
  3. Classify by T boundaries (35/120)
  4. Compute percentiles and path statistics
- `classify_paths(t_samples, boundaries=(35, 120)) -> dict`: Count samples per path

Uses vectorized numpy where possible. For full N=10000, optionally parallelize with `concurrent.futures.ProcessPoolExecutor` (chunk samples across workers).

**Step 4: Run tests**

```bash
pytest tests/test_core/test_mc.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hormuz/core/mc.py tests/test_core/test_mc.py
git commit -m "feat(core): add Monte Carlo simulation engine — N=10000, 6-param sampling"
```

---

## Task 9: SQLite storage — infra/db.py

**Files:**
- Create: `src/hormuz/infra/db.py`
- Test: `tests/test_infra/test_db.py`

**Step 1: Write the failing test**

```python
# tests/test_infra/test_db.py
import pytest
import sqlite3
from pathlib import Path
from datetime import datetime

@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"

def test_init_db(db_path):
    from hormuz.infra.db import init_db
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "observations" in tables
    assert "ach_evidence" in tables
    assert "state_snapshots" in tables
    assert "controls" in tables
    assert "mc_runs" in tables
    assert "system_outputs" in tables
    assert "position_signals" in tables
    assert "parameters_override" in tables
    conn.close()

def test_insert_observation(db_path):
    from hormuz.infra.db import init_db, insert_observation, get_observations
    from hormuz.core.types import Observation
    init_db(db_path)
    obs = Observation(id="O01", timestamp=datetime(2026, 3, 12), value=3.5, source="CENTCOM")
    insert_observation(db_path, obs)
    results = get_observations(db_path, since=datetime(2026, 3, 11))
    assert len(results) == 1
    assert results[0].id == "O01"

def test_insert_control(db_path):
    from hormuz.infra.db import init_db, insert_control, get_controls
    from hormuz.core.types import Control
    init_db(db_path)
    ctrl = Control(id="D02", actor="WHITE_HOUSE", triggered=True, trigger_time=datetime(2026, 3, 12))
    insert_control(db_path, ctrl)
    results = get_controls(db_path)
    assert len(results) == 1
    assert results[0].triggered

def test_save_system_output(db_path):
    from hormuz.infra.db import init_db, save_system_output, get_latest_output
    from hormuz.core.types import SystemOutput, ACHPosterior, PathWeights
    init_db(db_path)
    so = SystemOutput(
        timestamp=datetime(2026, 3, 12),
        ach_posterior=ACHPosterior(h1=0.5, h2=0.5, h3=None),
        t1_percentiles={"p50": 21}, t2_percentiles={"p50": 35},
        t_total_percentiles={"p50": 63},
        buffer_trajectory=[(0, 0.0)], gross_gap_mbd=16.0,
        net_gap_trajectories={}, path_probabilities=PathWeights(),
        path_total_gaps={"A": 270.0, "B": 833.0, "C": 2500.0},
        expected_total_gap=700.0, consistency_flags=[],
    )
    save_system_output(db_path, so)
    latest = get_latest_output(db_path)
    assert latest is not None
    assert latest.gross_gap_mbd == 16.0

def test_save_parameter_override(db_path):
    from hormuz.infra.db import init_db, save_parameter_override, get_parameter_overrides
    init_db(db_path)
    save_parameter_override(db_path, param="mines_in_water_range", old_value="(20, 100)", new_value="(30, 80)")
    overrides = get_parameter_overrides(db_path)
    assert len(overrides) == 1
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_infra/test_db.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/hormuz/infra/db.py`:

- `init_db(path)`: CREATE TABLE IF NOT EXISTS for all 8 tables
- `insert_observation(path, obs)`, `get_observations(path, since) -> list[Observation]`
- `insert_control(path, ctrl)`, `get_controls(path) -> list[Control]`
- `save_system_output(path, so)`, `get_latest_output(path) -> SystemOutput | None`
- `save_mc_run(path, result)`, `save_state_snapshot(path, state)`
- `save_parameter_override(path, param, old_value, new_value)`, `get_parameter_overrides(path)`
- `save_position_signal(path, signal)`, `get_pending_signals(path)`

All functions take `path: Path` as first argument, open/close connection per call. SystemOutput and MCResult serialized as JSON blobs.

**Step 4: Run tests**

```bash
pytest tests/test_infra/test_db.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hormuz/infra/db.py tests/test_infra/test_db.py
git commit -m "feat(infra): add SQLite storage — 8 tables, CRUD for all variable types"
```

---

## Task 10: Data ingestion — infra/ingester.py

**Files:**
- Create: `src/hormuz/infra/ingester.py`
- Test: `tests/test_infra/test_ingester.py`

**Step 1: Write the failing test**

```python
# tests/test_infra/test_ingester.py
import pytest
from unittest.mock import patch, AsyncMock
from datetime import datetime

def test_parse_readwise_articles():
    from hormuz.infra.ingester import parse_readwise_articles
    raw = [
        {"id": 1, "title": "CENTCOM report", "summary": "3 attacks today", "published_date": "2026-03-12",
         "source": "CENTCOM", "url": "https://example.com/1"},
    ]
    articles = parse_readwise_articles(raw)
    assert len(articles) == 1
    assert articles[0]["source"] == "CENTCOM"

def test_market_data_to_observations():
    from hormuz.infra.ingester import market_data_to_observations
    market = {"brent": 95.5, "ovx": 42.0, "brent_term_structure": [95.5, 93.0, 90.0]}
    obs_list = market_data_to_observations(market, timestamp=datetime(2026, 3, 12))
    ids = [o.id for o in obs_list]
    assert any("O07" in i or "O09" in i or "O10" in i for i in ids)

@pytest.mark.asyncio
async def test_fetch_market_data():
    from hormuz.infra.ingester import fetch_market_data
    with patch("hormuz.infra.ingester.yf") as mock_yf:
        mock_ticker = mock_yf.Ticker.return_value
        mock_ticker.info = {"regularMarketPrice": 95.5}
        result = await fetch_market_data(proxy=None)
        assert "brent" in result
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_infra/test_ingester.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/hormuz/infra/ingester.py`:

- `fetch_readwise_articles(token, tag, proxy, timeout) -> list[dict]`: httpx GET to Readwise API, filter by tag
- `parse_readwise_articles(raw) -> list[dict]`: Normalize article fields
- `fetch_market_data(proxy) -> dict`: yfinance for Brent (BZ=F), OVX (^OVX)
- `market_data_to_observations(market, timestamp) -> list[Observation]`: Map market fields to O07/O09/O10

**Step 4: Run tests**

```bash
pytest tests/test_infra/test_ingester.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hormuz/infra/ingester.py tests/test_infra/test_ingester.py
git commit -m "feat(infra): add data ingestion — Readwise + yfinance"
```

---

## Task 11: LLM backend + analyzer — infra/llm.py + infra/analyzer.py

**Files:**
- Create: `src/hormuz/infra/llm.py`
- Create: `src/hormuz/infra/analyzer.py`
- Test: `tests/test_infra/test_analyzer.py`

**Step 1: Write the failing test**

```python
# tests/test_infra/test_analyzer.py
import pytest
from unittest.mock import AsyncMock
from datetime import datetime

@pytest.mark.asyncio
async def test_extract_observations():
    from hormuz.infra.analyzer import extract_observations
    mock_llm = AsyncMock()
    mock_llm.extract.return_value = {
        "observations": [
            {"id": "O01", "value": 3.0, "confidence": "high", "direction": "H1"},
            {"id": "O04", "value": 0.8, "confidence": "high", "direction": "H1"},
        ]
    }
    articles = [{"title": "CENTCOM update", "summary": "Attack frequency declining", "source": "CENTCOM"}]
    obs = await extract_observations(articles, llm=mock_llm)
    assert len(obs) == 2
    assert obs[0].id == "O01"

def test_llm_factory():
    from hormuz.infra.llm import create_llm_backend
    backend = create_llm_backend(backend_type="claude_api", model="claude-sonnet-4-6", api_key="test")
    assert hasattr(backend, "extract")
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_infra/test_analyzer.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/hormuz/infra/llm.py`:

```python
class LLMBackend(Protocol):
    async def extract(self, text: str, prompt: str) -> dict: ...

class ClaudeAPIBackend:
    def __init__(self, model, api_key, base_url=None, proxy=None): ...
    async def extract(self, text, prompt) -> dict: # httpx POST to Claude API

class OpenClawBackend:
    def __init__(self, endpoint): ...
    async def extract(self, text, prompt) -> dict: # httpx POST to OpenClaw

def create_llm_backend(backend_type, **kwargs) -> LLMBackend: # factory
```

`src/hormuz/infra/analyzer.py`:

- `extract_observations(articles, llm) -> list[Observation]`: Build structured prompt asking LLM to extract O01-O13 observations from article text. Parse JSON response into Observation list.
- Prompt template includes observation ID definitions and expected output format.

**Step 4: Run tests**

```bash
pytest tests/test_infra/test_analyzer.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hormuz/infra/llm.py src/hormuz/infra/analyzer.py tests/test_infra/test_analyzer.py
git commit -m "feat(infra): add LLM backend + observation analyzer"
```

---

## Task 12: Grabo tripwire signals — app/signals.py

**Files:**
- Create: `src/hormuz/app/signals.py`
- Test: `tests/test_app/test_signals.py`

**Step 1: Write the failing test**

```python
# tests/test_app/test_signals.py
import pytest
from datetime import datetime, timedelta
from hormuz.core.types import Observation

def make_obs(obs_id, value, ts=None):
    return Observation(id=obs_id, timestamp=ts or datetime(2026, 3, 12), value=value, source="test")

def test_no_signals():
    from hormuz.app.signals import scan_signals
    obs = [make_obs("O01", 3.0)]
    result = scan_signals(obs, signal_state={})
    assert len(result.triggered) == 0

def test_t1a_triggered():
    """GPS up + attack freq rising -> T1a"""
    from hormuz.app.signals import scan_signals
    obs = [
        make_obs("O05", 0.8),  # GPS spoofing high
        make_obs("O01", 5.0),  # attack freq high
    ]
    result = scan_signals(obs, signal_state={}, o01_trend="rising")
    assert "T1a" in result.triggered

def test_t1b_triggered():
    """GPS up + attack freq falling -> T1b"""
    from hormuz.app.signals import scan_signals
    obs = [make_obs("O05", 0.8), make_obs("O01", 1.5)]
    result = scan_signals(obs, signal_state={}, o01_trend="falling")
    assert "T1b" in result.triggered

def test_48h_revert():
    """T1a should revert after 48h"""
    from hormuz.app.signals import check_reverts
    state = {"T1a": {"triggered_at": datetime(2026, 3, 10, 0, 0)}}
    now = datetime(2026, 3, 12, 1, 0)  # 49h later
    reverted = check_reverts(state, now=now)
    assert "T1a" in reverted

def test_e3_persistent():
    """E3 mine strike is persistent (no revert)"""
    from hormuz.app.signals import scan_signals
    result = scan_signals([], signal_state={}, events={"E3": True})
    assert "E3" in result.triggered

def test_signal_result_has_position_actions():
    from hormuz.app.signals import scan_signals
    result = scan_signals([], signal_state={}, events={"E1": True})
    assert len(result.position_actions) > 0  # E1 -> vol×2 + recession 5%
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_app/test_signals.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/hormuz/app/signals.py`:

```python
@dataclass
class SignalResult:
    triggered: list[str]         # signal IDs that fired
    reverted: list[str]          # signals that auto-reverted
    position_actions: list[dict] # immediate position adjustments
    events: dict                 # event flags for engine

def scan_signals(observations, signal_state, o01_trend=None, events=None) -> SignalResult:
    # T1a/T1b: check O05 + O01 co-occurrence
    # T2: multi-region activation
    # T3: mining boats
    # E1-E4: event-based triggers
    # C1-C2: confirmations
    # Map triggers to position actions per PRD

def check_reverts(signal_state, now) -> list[str]:
    # Check 48h expiry for T1a/T1b, T2, T3, E4
    # E1-E3, C1-C2 are persistent
```

**Step 4: Run tests**

```bash
pytest tests/test_app/test_signals.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hormuz/app/signals.py tests/test_app/test_signals.py
git commit -m "feat(app): add Grabo tripwire signal system — T1-T3 + E1-E4 + 48h revert"
```

---

## Task 13: Position rules engine — app/positions.py

**Files:**
- Create: `src/hormuz/app/positions.py`
- Test: `tests/test_app/test_positions.py`

**Step 1: Write the failing test**

```python
# tests/test_app/test_positions.py
import pytest
from hormuz.core.types import PathWeights, SystemOutput, ACHPosterior
from datetime import datetime

def make_output(**kwargs):
    defaults = dict(
        timestamp=datetime(2026, 3, 12),
        ach_posterior=ACHPosterior(h1=0.5, h2=0.5, h3=None),
        t1_percentiles={"p50": 21}, t2_percentiles={"p50": 35},
        t_total_percentiles={"p50": 63},
        buffer_trajectory=[], gross_gap_mbd=16.0,
        net_gap_trajectories={}, path_probabilities=PathWeights(),
        path_total_gaps={"A": 270, "B": 833, "C": 2500},
        expected_total_gap=700, consistency_flags=[],
    )
    defaults.update(kwargs)
    return SystemOutput(**defaults)

def test_base_positions():
    from hormuz.app.positions import evaluate_positions
    so = make_output()
    signals = []
    result = evaluate_positions(so, brent_price=95.0, signals=signals)
    assert result.energy_pct == 15
    assert result.vol_pct == 3
    assert result.recession_pct == 2

def test_t_end_exit():
    """Transit up 3 days + AP < 1% -> unwind"""
    from hormuz.app.positions import evaluate_positions
    so = make_output()
    result = evaluate_positions(so, brent_price=85.0, signals=[], t_end_confirmed=True)
    assert result.energy_pct < 15  # reducing
    assert result.vol_pct == 0     # closed

def test_demand_destruction():
    """Brent > 150 -> clear energy, double recession"""
    from hormuz.app.positions import evaluate_positions
    so = make_output()
    result = evaluate_positions(so, brent_price=155.0, signals=[])
    assert result.energy_pct == 0
    assert result.recession_pct == 4

def test_system_failure():
    """Brent < 80 for 3 days -> force close all"""
    from hormuz.app.positions import evaluate_positions
    so = make_output()
    result = evaluate_positions(so, brent_price=78.0, signals=[], brent_below_80_days=3)
    assert result.energy_pct == 0
    assert result.vol_pct == 0

def test_tripwire_override():
    """T1a signal -> vol×2"""
    from hormuz.app.positions import evaluate_positions
    so = make_output()
    result = evaluate_positions(so, brent_price=95.0, signals=[{"action": "vol_double"}])
    assert result.vol_pct == 6
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_app/test_positions.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/hormuz/app/positions.py`:

```python
@dataclass
class PositionResult:
    energy_pct: int
    vol_pct: int
    recession_pct: int
    actions: list[str]  # human-readable action descriptions
    executed: bool = False  # human confirmation boundary

def evaluate_positions(system_output, brent_price, signals,
                       t_end_confirmed=False, brent_below_80_days=0) -> PositionResult:
    # 1. Start with base positions (15/3/2)
    # 2. Apply tripwire overrides from signals
    # 3. Check exit rules: T end / $150 / $80
    # 4. Return PositionResult with action descriptions
```

**Step 4: Run tests**

```bash
pytest tests/test_app/test_positions.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hormuz/app/positions.py tests/test_app/test_positions.py
git commit -m "feat(app): add position rules engine — base positions + exit rules + tripwire overrides"
```

---

## Task 14: Pipeline orchestrator — app/pipeline.py

**Files:**
- Create: `src/hormuz/app/pipeline.py`
- Test: `tests/test_app/test_pipeline.py`

**Step 1: Write the failing test**

```python
# tests/test_app/test_pipeline.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
from datetime import datetime

@pytest.fixture
def db_path(tmp_path):
    from hormuz.infra.db import init_db
    p = tmp_path / "test.db"
    init_db(p)
    return p

@pytest.mark.asyncio
async def test_pipeline_full_run(db_path, tmp_path):
    from hormuz.app.pipeline import run_pipeline
    config = {
        "db": {"path": str(db_path)},
        "configs_dir": str(Path(__file__).parents[2] / "configs"),
        "readwise": {"token": "fake", "tag": "test", "proxy": None, "timeout": 10},
        "llm": {"backend": "claude_api", "claude_api": {"model": "test", "api_key": "fake"}},
        "conflict": {"start_date": "2026-03-01"},
        "output_dir": str(tmp_path),
    }
    with patch("hormuz.app.pipeline.fetch_readwise_articles", new_callable=AsyncMock, return_value=[]), \
         patch("hormuz.app.pipeline.fetch_market_data", new_callable=AsyncMock, return_value={"brent": 95.0}), \
         patch("hormuz.app.pipeline.extract_observations", new_callable=AsyncMock, return_value=[]):
        result = await run_pipeline(config)
    assert result["steps_completed"] >= 5
    assert "system_output" in result

def test_engine_run_pure():
    """Engine run with canned data, no IO"""
    from hormuz.app.pipeline import engine_run
    from hormuz.core.types import Parameters, ACHPosterior
    from hormuz.core.variables import load_constants
    constants = load_constants(Path(__file__).parents[2] / "configs" / "constants.yaml")
    params = Parameters()
    observations = []
    controls = []
    so = engine_run(constants, params, observations, controls, events={}, mc_n=100, seed=42)
    assert so.gross_gap_mbd == pytest.approx(16.0, abs=0.2)
    assert so.path_probabilities.a + so.path_probabilities.b + so.path_probabilities.c == pytest.approx(1.0)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_app/test_pipeline.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/hormuz/app/pipeline.py`:

- `engine_run(constants, params, observations, controls, events, mc_n, seed) -> SystemOutput`: Pure compute chain M1→M2→M3→M4→M5→MC. No IO. Returns SystemOutput.
- `run_pipeline(config) -> dict`: Full 7-step orchestrator:
  1. `ingester.fetch_readwise_articles()` + `ingester.fetch_market_data()`
  2. `analyzer.extract_observations()`
  3. `signals.scan_signals()` (穿透语义, before ACH)
  4. `engine_run()` (M1→M5→MC)
  5. `positions.evaluate_positions()`
  6. `reporter.render()` (Task 16)
  7. `db.snapshot()` (persist all)

Each step wrapped in try/except, errors collected, pipeline continues.

**Step 4: Run tests**

```bash
pytest tests/test_app/test_pipeline.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hormuz/app/pipeline.py tests/test_app/test_pipeline.py
git commit -m "feat(app): add pipeline orchestrator — 7-step engine with signal-first semantics"
```

---

## Task 15: CLI — app/cli.py

**Files:**
- Create: `src/hormuz/app/cli.py`
- Test: `tests/test_app/test_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_app/test_cli.py
import pytest
from click.testing import CliRunner
from unittest.mock import patch, AsyncMock

def test_cli_help():
    from hormuz.app.cli import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "status" in result.output

def test_cli_init_db(tmp_path):
    from hormuz.app.cli import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["init-db", "--db-path", str(tmp_path / "test.db")])
    assert result.exit_code == 0
    assert (tmp_path / "test.db").exists()

def test_cli_status_no_data(tmp_path):
    from hormuz.app.cli import cli
    from hormuz.infra.db import init_db
    db = tmp_path / "test.db"
    init_db(db)
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--db-path", str(db)])
    assert result.exit_code == 0
    assert "No data" in result.output or "no runs" in result.output.lower()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_app/test_cli.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/hormuz/app/cli.py`:

```python
@click.group()
def cli(): ...

@cli.command()
def run(): # Load config, run_pipeline()

@cli.command()
def status(): # get_latest_output(), print summary

@cli.command()
def record(): # Interactive observation/control entry

@cli.command()
def mc(): # Standalone MC rerun

@cli.command()
def report(): # Generate weekly report

@cli.command()
def override(): # Parameter override with DB logging

@cli.command()
def init_db(): # Initialize SQLite

@cli.command()
def validate(): # Run consistency checks
```

Config loading: read `configs/config.yaml`, expand env vars, merge with CLI flags.

**Step 4: Run tests**

```bash
pytest tests/test_app/test_cli.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hormuz/app/cli.py tests/test_app/test_cli.py
git commit -m "feat(app): add Click CLI — run/status/record/mc/report/override/init-db/validate"
```

---

## Task 16: HTML reporter — app/reporter.py + templates/status.html.jinja

**Files:**
- Create: `src/hormuz/app/reporter.py`
- Recreate: `templates/status.html.jinja`
- Test: `tests/test_app/test_reporter.py`

**Step 1: Write the failing test**

```python
# tests/test_app/test_reporter.py
import pytest
from pathlib import Path
from datetime import datetime
from hormuz.core.types import SystemOutput, ACHPosterior, PathWeights, Parameters
from hormuz.core.mc import MCResult
import numpy as np

def make_system_output():
    return SystemOutput(
        timestamp=datetime(2026, 3, 12),
        ach_posterior=ACHPosterior(h1=0.6, h2=0.4, h3=None),
        t1_percentiles={"p10": 14, "p25": 17, "p50": 21, "p75": 28, "p90": 35},
        t2_percentiles={"p10": 20, "p25": 28, "p50": 35, "p75": 42, "p90": 56},
        t_total_percentiles={"p10": 42, "p25": 52, "p50": 63, "p75": 77, "p90": 98},
        buffer_trajectory=[(d, min(d * 0.5, 7.0)) for d in range(91)],
        gross_gap_mbd=16.0,
        net_gap_trajectories={"A": [(0, 16.0), (28, 5.0)], "B": [(0, 16.0), (84, 9.0)]},
        path_probabilities=PathWeights(a=0.30, b=0.50, c=0.20),
        path_total_gaps={"A": 270.0, "B": 833.0, "C": 2500.0},
        expected_total_gap=700.0,
        consistency_flags=["AP declining but S06 still high"],
    )

def test_render_status(tmp_path):
    from hormuz.app.reporter import render_status
    so = make_system_output()
    params = Parameters()
    mc_result = MCResult(
        t_samples=np.random.default_rng(42).normal(63, 20, 100),
        total_gap_samples=np.random.default_rng(42).normal(700, 200, 100),
        t_percentiles={"p10": 42, "p50": 63, "p90": 98},
        path_frequencies={"A": 0.25, "B": 0.55, "C": 0.20},
        path_total_gap_means={"A": 280.0, "B": 850.0, "C": 2600.0},
    )
    output_path = tmp_path / "status.html"
    render_status(so, mc_result, params, output_path=output_path,
                  conflict_start="2026-03-01", brent_price=95.0)
    assert output_path.exists()
    html = output_path.read_text()
    assert "16.0" in html  # gross gap
    assert "270" in html   # path A gap
    assert "参数" in html   # parameter section in Chinese

def test_render_has_all_sections(tmp_path):
    from hormuz.app.reporter import render_status
    so = make_system_output()
    params = Parameters()
    mc_result = MCResult(
        t_samples=np.ones(10), total_gap_samples=np.ones(10),
        t_percentiles={"p10": 1, "p50": 1, "p90": 1},
        path_frequencies={"A": 0.3, "B": 0.5, "C": 0.2},
        path_total_gap_means={"A": 1, "B": 1, "C": 1},
    )
    out = tmp_path / "status.html"
    render_status(so, mc_result, params, output_path=out,
                  conflict_start="2026-03-01", brent_price=95.0)
    html = out.read_text()
    # Check all 9 sections present
    for section in ["状态总览", "核心公式", "物理层", "博弈层", "路径",
                    "MC", "仓位", "参数", "校验"]:
        assert section in html, f"Missing section: {section}"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_app/test_reporter.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`src/hormuz/app/reporter.py`:

- `render_status(system_output, mc_result, params, output_path, conflict_start, brent_price, overrides=None)`: Load Jinja2 template, inject all data, generate matplotlib charts as base64, write HTML.
- `render_weekly_report(output_path, ...)`: Same data + archival metadata.
- `generate_mc_chart(mc_result) -> str`: T histogram + TotalGap histogram as base64 PNG.
- `generate_buffer_chart(buffer_trajectory) -> str`: Buffer ramp line chart as base64 PNG.

`templates/status.html.jinja`: 9 sections matching design doc:

1. 状态总览 — conflict day count, path weights bar, Brent price, transit count
2. 核心公式 — formula bar (inspired by ref-v54-dashboard.html)
3. 物理层 — ACH heatmap (H1/H2 posterior bars), Q2 mine stock, Q3 buffer ramp chart
4. 博弈层 — Schelling 3 diagnostics + signal table + delta display
5. 路径卡片 — A/B/C probability + TotalGap + segment breakdown
6. MC 分布 — T histogram + TotalGap histogram (base64 img)
7. 仓位 — energy/vol/recession percentages + pending actions
8. 参数设置说明 — Constants table (C01-C05 with values + source), Parameters table (P01-P10 with current value, distribution, last override), MC config (N, distributions), path boundaries (35/120)
9. 校验层 — consistency_flags list with warning styling

Dark theme styling inspired by `docs/ref-v54-dashboard.html`.

**Step 4: Run tests**

```bash
pytest tests/test_app/test_reporter.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add src/hormuz/app/reporter.py templates/status.html.jinja tests/test_app/test_reporter.py
git commit -m "feat(app): add HTML reporter — 9-section dashboard with parameter docs"
```

---

## Task 17: Integration test + CLAUDE.md update + final commit

**Files:**
- Create: `tests/test_app/test_integration.py`
- Modify: `CLAUDE.md`
- Modify: `configs/config.yaml` (update CLI entry point)

**Step 1: Write integration test**

```python
# tests/test_app/test_integration.py
"""Full pipeline integration test with fixture data, no external IO."""
import pytest
from pathlib import Path
from datetime import datetime

@pytest.fixture
def setup(tmp_path):
    from hormuz.infra.db import init_db
    db = tmp_path / "test.db"
    init_db(db)
    return {"db_path": db, "output_dir": tmp_path, "configs_dir": Path(__file__).parents[2] / "configs"}

def test_engine_run_produces_valid_output(setup):
    from hormuz.app.pipeline import engine_run
    from hormuz.core.variables import load_constants, load_parameters
    from hormuz.core.types import SystemOutput
    constants = load_constants(setup["configs_dir"] / "constants.yaml")
    params = load_parameters(setup["configs_dir"] / "parameters.yaml")
    so = engine_run(constants, params, observations=[], controls=[], events={}, mc_n=100, seed=42)
    assert isinstance(so, SystemOutput)
    assert so.gross_gap_mbd > 15
    assert so.path_probabilities.a + so.path_probabilities.b + so.path_probabilities.c == pytest.approx(1.0)
    assert so.path_total_gaps["A"] < so.path_total_gaps["B"] < so.path_total_gaps["C"]

def test_full_roundtrip(setup):
    """Engine run -> save to DB -> load from DB -> render HTML"""
    from hormuz.app.pipeline import engine_run
    from hormuz.app.reporter import render_status
    from hormuz.infra.db import save_system_output, get_latest_output
    from hormuz.core.variables import load_constants, load_parameters
    from hormuz.core.mc import run_monte_carlo
    from hormuz.core.types import ACHPosterior

    constants = load_constants(setup["configs_dir"] / "constants.yaml")
    params = load_parameters(setup["configs_dir"] / "parameters.yaml")
    so = engine_run(constants, params, [], [], {}, mc_n=100, seed=42)

    # Save + load
    save_system_output(setup["db_path"], so)
    loaded = get_latest_output(setup["db_path"])
    assert loaded is not None
    assert loaded.gross_gap_mbd == so.gross_gap_mbd

    # Render
    mc_result = run_monte_carlo(
        ACHPosterior(h1=0.5, h2=0.5, h3=None), params, {}, n=100, seed=42
    )
    out_html = setup["output_dir"] / "status.html"
    render_status(so, mc_result, params, output_path=out_html,
                  conflict_start="2026-03-01", brent_price=95.0)
    assert out_html.exists()
    assert len(out_html.read_text()) > 1000

def test_all_core_tests_pass():
    """Meta: ensure pytest collects and passes all core tests"""
    import subprocess
    result = subprocess.run(
        ["pytest", "tests/test_core/", "-v", "--tb=short"],
        capture_output=True, text=True, cwd=str(Path(__file__).parents[2])
    )
    assert result.returncode == 0
```

**Step 2: Run integration test**

```bash
pytest tests/test_app/test_integration.py -v
```
Expected: all PASS

**Step 3: Update CLAUDE.md**

Rewrite to reflect v5.4 architecture:

```markdown
# Hormuz Decision OS v5.4

## WHAT
霍尔木兹海峡危机投资决策操作系统。状态空间模型 + 贝叶斯推断引擎，
输出 T 分布、净缺口轨迹、三路径总缺口与概率权重。

## WHY
三个物理问题（主动威胁衰减/水雷演化/缓冲到位）决定油价路径，
需要系统化跟踪 13 类观测、贝叶斯更新假设、MC 模拟不确定性。

## HOW
- Python 3.12+, numpy, scipy, SQLite, Jinja2, matplotlib
- 三层架构: core/(纯计算) + infra/(IO) + app/(产品)
- core/ 映射 PRD M1-M5: ACH → T分布 → Buffer → 缺口积分 → 博弈调节
- MC N=10000 联合采样 6 个不确定参数
- 绊线穿透语义: signals.scan() 在 ACH 前执行

## 架构
- core/types.py: PRD §2 六类变量 + §7 SystemOutput
- core/m1_ach.py: 贝叶斯推断 (LR ∈ {0.2, 0.5, 1.0, 2.0, 5.0})
- core/m2_duration.py: T₁对数正态 + T₂ stock-flow + 卷积合成
- core/m3_buffer.py: pipeline(t) + spr(t) + cape(t) 三子缓冲
- core/m4_gap.py: 分段积分 ∫₀ᵀ [16 - Buffer(t)] dt
- core/m5_game.py: Schelling 信号 delta + 归一化 + clip
- core/mc.py: N=10000 蒙特卡洛，路径分类 35/120 天

## 关键约定
- 物理 > 制度 > 博弈（因果优先级硬编码）
- core/ 零 IO 零副作用，纯函数可独立测试
- PathWeights 始终归一化，clip [0.05, 0.85]
- position_signals.executed 是人机边界
- H3（外部补给）挂起，先验 ±5% 分配给 H1/H2
```

**Step 4: Run full test suite**

```bash
pytest tests/ -v
```
Expected: all PASS

**Step 5: Delete stale data and reset DB**

```bash
rm -f data/hormuz.db
cd ~/Projects/hormuz-ds && python -c "from hormuz.infra.db import init_db; from pathlib import Path; init_db(Path('data/hormuz.db'))"
```

**Step 6: Final commit**

```bash
git add -A && git commit -m "feat: complete v5.4 reimplementation — integration tests + updated CLAUDE.md"
```

---

## Summary

| Task | Module | Key Deliverable |
|---|---|---|
| 0 | Scaffold | Clean slate, new dir structure, deps |
| 1 | core/types.py | Pydantic data contracts (6 variable types + SystemOutput) |
| 2 | core/variables.py | YAML loader for Constants + Parameters |
| 3 | core/m1_ach.py | ACH Bayesian engine (LR table + T1a/T1b unbinding) |
| 4 | core/m2_duration.py | T distribution (lognormal T1 + MC T2 + convolution) |
| 5 | core/m3_buffer.py | Buffer ramp (pipeline + SPR + cape, 3 sub-buffers) |
| 6 | core/m4_gap.py | Gap integrator (segmented ∫ NetGap dt) |
| 7 | core/m5_game.py | Game theory path adjuster (signal deltas + normalize) |
| 8 | core/mc.py | Monte Carlo N=10000 (6-param sampling, path classification) |
| 9 | infra/db.py | SQLite 8 tables + CRUD |
| 10 | infra/ingester.py | Readwise + yfinance data ingestion |
| 11 | infra/llm.py + analyzer.py | LLM backend + observation extraction |
| 12 | app/signals.py | Grabo tripwires (T1-T3, E1-E4, 48h revert) |
| 13 | app/positions.py | Position rules (base + exit + tripwire override) |
| 14 | app/pipeline.py | 7-step orchestrator |
| 15 | app/cli.py | Click CLI (8 commands) |
| 16 | app/reporter.py | HTML dashboard (9 sections + parameter docs) |
| 17 | Integration | Full roundtrip test + CLAUDE.md update |
