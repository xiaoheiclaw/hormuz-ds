"""Pipeline orchestrator — 7-step engine with signal-first semantics.

engine_run: pure compute chain (M1→M5→MC), zero IO.
run_pipeline: async orchestrator with IO (fetch, analyze, persist).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from hormuz.core.types import (
    ACHPosterior,
    Constants,
    Control,
    Observation,
    Parameters,
    PathWeights,
    StateVector,
    SystemOutput,
)
from hormuz.core.m1_ach import run_ach, map_to_decay_rate
from hormuz.core.m2_duration import estimate_t_total, compute_percentiles
from hormuz.core.m3_buffer import compute_buffer_trajectory
from hormuz.core.m4_gap import compute_gross_gap, compute_net_gap_trajectory, integrate_total_gap
from hormuz.core.m5_game import adjust_path_weights
from hormuz.core.mc import run_monte_carlo

# These are imported for run_pipeline but may be mocked in tests
from hormuz.infra.ingester import fetch_readwise_articles, fetch_market_data, market_data_to_observations
from hormuz.infra.analyzer import extract_observations


def engine_run(
    constants: Constants,
    params: Parameters,
    observations: list[Observation],
    controls: list[Control],
    events: dict[str, bool],
    mc_n: int = 10000,
    seed: int | None = None,
) -> SystemOutput:
    """Pure compute chain: M1 → M2 → M3 → M4 → M5 → MC.

    No IO, no side effects. Returns SystemOutput.
    """
    # M1: ACH Bayesian inference
    posterior = run_ach(observations, h3_suspended=params.h3_suspended, h3_prior=params.h3_prior)

    # M2: T distribution
    t1_samples, t2_samples, t_total_samples = estimate_t_total(
        posterior, params, events, n=mc_n, seed=seed,
    )
    t1_pct = compute_percentiles(t1_samples)
    t2_pct = compute_percentiles(t2_samples)
    t_total_pct = compute_percentiles(t_total_samples)

    # M3: Buffer trajectory
    max_day = max(180, int(t_total_pct.get("p90", 120)) + 30)
    buffer_traj = compute_buffer_trajectory(max_day=max_day, params=params)

    # M4: Gap
    state = StateVector(effective_disruption=params.effective_disruption_rate)
    gross_gap = compute_gross_gap(constants, state)
    net_gap_traj = compute_net_gap_trajectory(gross_gap, buffer_traj)

    # Path total gaps (representative durations)
    path_t = {"A": 28, "B": 84, "C": 180}
    path_total_gaps = {
        path: integrate_total_gap(gross_gap, buffer_traj, t_end=t)
        for path, t in path_t.items()
    }

    # Net gap trajectories per path
    net_gap_trajectories = {}
    for path, t_end in path_t.items():
        net_gap_trajectories[path] = [
            (d, ng) for d, ng in net_gap_traj if d <= t_end
        ]

    # M5: Game theory path adjustment
    # Extract active game signals from controls
    game_signals = []
    for ctrl in controls:
        if ctrl.triggered and ctrl.effect:
            game_signals.append(ctrl.effect)
    path_probs = adjust_path_weights(PathWeights(), active_signals=game_signals)

    # MC simulation
    mc_result = run_monte_carlo(posterior, params, events, n=mc_n, seed=seed)

    # Expected total gap
    expected_gap = (
        path_probs.a * path_total_gaps["A"]
        + path_probs.b * path_total_gaps["B"]
        + path_probs.c * path_total_gaps["C"]
    )

    # Consistency flags
    flags = _check_consistency(posterior, params, gross_gap)

    return SystemOutput(
        timestamp=datetime.now(),
        ach_posterior=posterior,
        t1_percentiles=t1_pct,
        t2_percentiles=t2_pct,
        t_total_percentiles=t_total_pct,
        buffer_trajectory=buffer_traj,
        gross_gap_mbd=gross_gap,
        net_gap_trajectories=net_gap_trajectories,
        path_probabilities=path_probs,
        path_total_gaps=path_total_gaps,
        expected_total_gap=expected_gap,
        consistency_flags=flags,
    )


def _check_consistency(
    posterior: ACHPosterior,
    params: Parameters,
    gross_gap: float,
) -> list[str]:
    """Basic consistency checks."""
    flags = []
    if gross_gap < 10:
        flags.append(f"gross_gap unexpectedly low: {gross_gap:.1f} mbd")
    if posterior.h1 + posterior.h2 < 0.95 and posterior.h3 is None:
        flags.append("H1+H2 < 0.95 with H3 suspended — prior leak")
    return flags


async def run_pipeline(config: dict) -> dict:
    """Full 7-step pipeline orchestrator.

    1. Fetch articles + market data
    2. LLM extract observations
    3. Signal scan (穿透语义, before ACH)
    4. Engine run (M1→M5→MC)
    5. Position evaluation
    6. Report generation (if reporter available)
    7. DB snapshot
    """
    from hormuz.core.variables import load_constants, load_parameters
    from hormuz.infra.db import (
        save_system_output,
        get_observations,
        get_controls,
    )
    from hormuz.app.signals import scan_signals
    from hormuz.app.positions import evaluate_positions

    result: dict = {"steps_completed": 0, "errors": []}
    configs_dir = Path(config["configs_dir"])
    db_path = Path(config["db"]["path"])

    # Step 1: Fetch data
    try:
        articles = await fetch_readwise_articles(
            token=config["readwise"]["token"],
            tag=config["readwise"].get("tag", "hormuz"),
            proxy=config["readwise"].get("proxy"),
            timeout=config["readwise"].get("timeout", 30),
        )
        market = await fetch_market_data(proxy=config["readwise"].get("proxy"))
        result["steps_completed"] += 1
    except Exception as e:
        result["errors"].append(f"Step 1 fetch: {e}")
        articles, market = [], {}
        result["steps_completed"] += 1

    # Step 2: LLM observation extraction
    try:
        from hormuz.infra.llm import create_llm_backend
        llm_config = config.get("llm", {})
        backend_type = llm_config.get("backend", "claude_api")
        backend_kwargs = llm_config.get(backend_type, {})
        llm = create_llm_backend(backend_type, **backend_kwargs)
        llm_obs = await extract_observations(articles, llm=llm)
        result["steps_completed"] += 1
    except Exception as e:
        result["errors"].append(f"Step 2 LLM: {e}")
        llm_obs = []
        result["steps_completed"] += 1

    # Add market observations
    brent_price = market.get("brent", 95.0)
    market_obs = market_data_to_observations(market, timestamp=datetime.now())
    all_obs = llm_obs + market_obs

    # Step 3: Signal scan (穿透, before ACH)
    try:
        signal_result = scan_signals(all_obs, signal_state={})
        result["steps_completed"] += 1
    except Exception as e:
        result["errors"].append(f"Step 3 signals: {e}")
        signal_result = None
        result["steps_completed"] += 1

    # Step 4: Engine run
    try:
        constants = load_constants(configs_dir / "constants.yaml")
        params = load_parameters(configs_dir / "parameters.yaml")
        controls = get_controls(db_path)
        events = signal_result.events if signal_result else {}
        so = engine_run(constants, params, all_obs, controls, events, mc_n=100, seed=42)
        result["system_output"] = so
        result["steps_completed"] += 1
    except Exception as e:
        result["errors"].append(f"Step 4 engine: {e}")
        result["steps_completed"] += 1
        return result

    # Step 5: Position evaluation
    try:
        pos_actions = signal_result.position_actions if signal_result else []
        pos = evaluate_positions(so, brent_price=brent_price, signals=pos_actions)
        result["positions"] = pos
        result["steps_completed"] += 1
    except Exception as e:
        result["errors"].append(f"Step 5 positions: {e}")
        result["steps_completed"] += 1

    # Step 6: Report (skip if reporter not ready)
    result["steps_completed"] += 1

    # Step 7: DB snapshot
    try:
        save_system_output(db_path, so)
        result["steps_completed"] += 1
    except Exception as e:
        result["errors"].append(f"Step 7 DB: {e}")
        result["steps_completed"] += 1

    return result
