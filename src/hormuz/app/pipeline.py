"""Pipeline orchestrator — 6-step engine.

engine_run: pure compute chain (M1→M5→MC), zero IO.
run_pipeline: async orchestrator with IO (fetch, analyze, persist, report).
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
from hormuz.core.mc import MCResult, run_monte_carlo

# These are imported for run_pipeline but may be mocked in tests
from hormuz.infra.ingester import fetch_readwise_articles, fetch_market_data, fetch_spr_release, fetch_bunker_spread, get_calibration_data, bwet_to_vlcc_obs, parse_readwise_articles
from hormuz.infra.analyzer import extract_observations


def engine_run(
    constants: Constants,
    params: Parameters,
    observations: list[Observation],
    controls: list[Control],
    events: dict[str, bool],
    mc_n: int = 10000,
    seed: int | None = None,
    o01_trend: str = "stable",
    schelling_signals: list | None = None,
) -> tuple[SystemOutput, MCResult]:
    """Pure compute chain: M1 → M2 → M3 → M4 → M5 → MC.

    No IO, no side effects. Returns (SystemOutput, MCResult).
    """
    # M1: ACH Bayesian inference
    posterior = run_ach(
        observations, h3_suspended=params.h3_suspended,
        h3_prior=params.h3_prior, o01_trend=o01_trend,
    )

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

    # MC simulation (before M5, because MC path frequencies become base weights)
    mc_result = run_monte_carlo(posterior, params, events, n=mc_n, seed=seed)

    # M5: Game theory path adjustment
    # Base weights from MC physical simulation, then Schelling signals adjust
    # Sources: 1) controls table (manual → medium evidence), 2) LLM-extracted signals
    from hormuz.core.m5_game import SignalEvidence
    game_signals: list[SignalEvidence] = []
    for ctrl in controls:
        if ctrl.triggered and ctrl.effect:
            game_signals.append(SignalEvidence(key=ctrl.effect, evidence=0.5))
    if schelling_signals:
        game_signals.extend(schelling_signals)
    mc_base = PathWeights(
        a=mc_result.path_frequencies["A"],
        b=mc_result.path_frequencies["B"],
        c=mc_result.path_frequencies["C"],
    )
    path_probs = adjust_path_weights(mc_base, signals=game_signals)

    # Expected total gap
    expected_gap = (
        path_probs.a * path_total_gaps["A"]
        + path_probs.b * path_total_gaps["B"]
        + path_probs.c * path_total_gaps["C"]
    )

    # Consistency flags
    flags = _check_consistency(posterior, params, gross_gap)

    so = SystemOutput(
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
    return so, mc_result


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
    """Full 6-step pipeline orchestrator.

    1. Fetch articles + market data
    2. LLM extract observations + Schelling signals
    3. Engine run (M1→M5→MC)
    4. Position evaluation
    5. DB snapshot
    6. Report generation
    """
    from hormuz.core.variables import load_constants, load_parameters
    from hormuz.infra.db import (
        save_system_output,
        get_observations,
        get_controls,
        insert_observations,
        compute_o02_from_history,
        compute_o01_trend,
        compute_confidence_level,
    )
    from hormuz.app.positions import evaluate_positions

    result: dict = {"steps_completed": 0, "errors": []}
    configs_dir = Path(config["configs_dir"])
    db_path = Path(config["db"]["path"])

    # Step 1: Fetch data
    try:
        rw = config["readwise"]
        sources = set(rw["sources"]) if "sources" in rw else None
        articles = await fetch_readwise_articles(
            token=rw["token"],
            sources=sources,
            proxy=rw.get("proxy"),
            timeout=rw.get("timeout", 30),
        )
        market = await fetch_market_data(proxy=config["readwise"].get("proxy"))
        result["steps_completed"] += 1
    except Exception as e:
        result["errors"].append(f"Step 1 fetch: {e}")
        articles, market = [], {}
        result["steps_completed"] += 1

    # Step 2: LLM observation + Schelling signal extraction
    llm_signals: list[str] = []
    try:
        from hormuz.infra.llm import create_llm_backend
        llm_config = config.get("llm", {})
        backend_type = llm_config.get("backend", "claude_api")
        backend_kwargs = llm_config.get(backend_type, {})
        llm = create_llm_backend(backend_type, **backend_kwargs)
        parsed = parse_readwise_articles(articles)[:30]
        extraction = await extract_observations(parsed, llm=llm, batch_size=5)
        llm_obs = extraction.observations
        llm_signals = extraction.signals
        result["steps_completed"] += 1
    except Exception as e:
        result["errors"].append(f"Step 2 LLM: {e}")
        llm_obs = []
        result["steps_completed"] += 1

    # Calibration data (Brent/OVX — not O-series, used for consistency checks)
    calib = get_calibration_data(market)
    brent_price = calib.get("brent_price", 95.0)
    all_obs = llm_obs  # O-series comes only from LLM extraction now

    # BWET → O09 (VLCC freight proxy) — yfinance, free
    vlcc_obs = bwet_to_vlcc_obs(market, timestamp=datetime.now())
    if vlcc_obs:
        all_obs = [o for o in all_obs if o.id != "O09"] + [vlcc_obs]

    # Ship & Bunker → O12 (Fujairah-Singapore spread) — web scrape, free
    try:
        bunker_obs = await fetch_bunker_spread(
            proxy=config["readwise"].get("proxy"),
        )
        if bunker_obs:
            all_obs = [o for o in all_obs if o.id != "O12"] + [bunker_obs]
    except Exception as e:
        result["errors"].append(f"Bunker spread fetch: {e}")

    # EIA SPR data (O13) — public API, if key configured
    eia_key = config.get("eia", {}).get("api_key")
    if eia_key:
        try:
            spr_obs = await fetch_spr_release(
                eia_key, proxy=config["readwise"].get("proxy"),
            )
            if spr_obs:
                all_obs = [o for o in all_obs if o.id != "O13"] + [spr_obs]
        except Exception as e:
            result["errors"].append(f"EIA SPR fetch: {e}")

    # Persist observations to DB
    if all_obs:
        try:
            insert_observations(db_path, all_obs)
        except Exception as e:
            result["errors"].append(f"DB insert obs: {e}")

    # Compute O02 from historical O01 (replaces LLM's O02)
    try:
        computed_o02 = compute_o02_from_history(db_path)
        if computed_o02 is not None:
            all_obs = [o for o in all_obs if o.id != "O02"] + [computed_o02]
    except Exception:
        pass  # Fall back to LLM's O02 if any

    # Compute O01 trend from DB history (for T1a/T1b and signal scan)
    o01_trend = compute_o01_trend(db_path)

    # A6/O14 check: H3 unfreeze if unknown weapon type detected
    o14 = next((o for o in all_obs if o.id == "O14" and o.value > 0.5), None)

    # Step 3: Engine run
    try:
        constants = load_constants(configs_dir / "constants.yaml")
        params = load_parameters(configs_dir / "parameters.yaml")
        # H3 unfreeze: if O14 detected with high confidence, switch to 3-way ACH
        if o14 and o14.source.endswith(":high"):
            params = params.override(h3_suspended=False)
            result["h3_unfrozen"] = True
        controls = get_controls(db_path)
        confidence = compute_confidence_level(db_path)
        so, mc_result = engine_run(
            constants, params, all_obs, controls, events={},
            mc_n=config.get("mc", {}).get("n", 10000),
            seed=config.get("mc", {}).get("seed", 42),
            o01_trend=o01_trend,
            schelling_signals=llm_signals,
        )
        result["schelling_signals"] = llm_signals
        so.confidence_level = confidence
        if confidence == "burn_in":
            so.consistency_flags.append("BURN-IN: <3 days history, output unreliable")
        result["system_output"] = so
        result["steps_completed"] += 1
    except Exception as e:
        result["errors"].append(f"Step 3 engine: {e}")
        result["steps_completed"] += 1
        return result

    # Step 4: Position evaluation
    try:
        pos = evaluate_positions(so, brent_price=brent_price)
        result["positions"] = pos
        result["steps_completed"] += 1
    except Exception as e:
        result["errors"].append(f"Step 4 positions: {e}")
        result["steps_completed"] += 1

    # Step 5: DB snapshot
    try:
        save_system_output(db_path, so)
        result["steps_completed"] += 1
    except Exception as e:
        result["errors"].append(f"Step 5 DB: {e}")
        result["steps_completed"] += 1

    # Step 6: Report generation
    try:
        from hormuz.app.reporter import render_status
        output_html = Path(config.get("report_output", "data/status.html"))
        render_status(
            system_output=so,
            mc_result=mc_result,
            params=params,
            output_path=output_html,
            conflict_start=config.get("conflict_start", "2026-03-01"),
            brent_price=brent_price,
            position_result=result.get("positions"),
            game_signals=llm_signals,
            mc_n=config.get("mc", {}).get("n", 10000),
        )
        # Also sync to docs/ for GitHub Pages
        docs_html = Path(config.get("docs_dir", "docs")) / "index.html"
        if docs_html.parent.exists():
            import shutil
            shutil.copy2(output_html, docs_html)
        result["steps_completed"] += 1
    except Exception as e:
        result["errors"].append(f"Step 6 report: {e}")
        result["steps_completed"] += 1

    return result
