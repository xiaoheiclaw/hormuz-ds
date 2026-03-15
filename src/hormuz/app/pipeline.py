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
from hormuz.core.m1_ach import run_ach
from hormuz.core.m4_gap import compute_gross_gap, compute_net_gap_trajectory, integrate_total_gap
from hormuz.core.m5_game import adjust_path_weights
from hormuz.core.mc import MCResult, run_monte_carlo

# These are imported for run_pipeline but may be mocked in tests
from hormuz.infra.ingester import fetch_readwise_articles, fetch_market_data, fetch_spr_release, fetch_bunker_spread, get_calibration_data, bwet_to_vlcc_obs, parse_readwise_articles
from hormuz.infra.analyzer import extract_observations
from hormuz.infra.db import insert_articles, get_article_ids, insert_article_observations


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
    prior_log_odds: float | None = None,
    prior_h3_suspended: bool | None = None,
    prior_h3_posterior: float | None = None,
) -> tuple[SystemOutput, MCResult, float]:
    """Pure compute chain: M1 → MC (M2+M3+M4 inside) → M5.

    No IO, no side effects. Returns (SystemOutput, MCResult, ach_log_odds).
    MC is the single source of truth for T samples and buffer trajectory.
    ach_log_odds is the accumulated log-odds for persistence across runs.
    """
    # M1: ACH Bayesian inference (physical/market evidence only)
    posterior, ach_log_odds = run_ach(
        observations, h3_suspended=params.h3_suspended,
        h3_prior=params.h3_prior, o01_trend=o01_trend,
        prior_log_odds=prior_log_odds,
        prior_h3_suspended=prior_h3_suspended,
        prior_h3_posterior=prior_h3_posterior,
    )

    # Infer SPR trigger day from O13 observations
    spr_trigger_day: int | None = None
    o13_val = next((o.value for o in observations if o.id == "O13" and o.value > 0), None)
    if o13_val is not None:
        spr_trigger_day = 1  # conservative: ordered day 1, physical flow after pump delay

    # M4: Gross gap (computed once, shared with MC)
    state = StateVector(effective_disruption=params.effective_disruption_rate)
    gross_gap = compute_gross_gap(constants, state)

    # Extract mine-related signals from observations for T2 conditioning
    mine_signals: dict[str, float] = {}
    for o in observations:
        if o.id in ("O03", "O10"):
            mine_signals[o.id] = o.value

    # MC simulation: single source for T1/T2/T_total samples + buffer trajectory
    mc_result = run_monte_carlo(
        posterior, params, events, mine_signals=mine_signals,
        n=mc_n, seed=seed,
        spr_trigger_day=spr_trigger_day,
        gross_gap_mbd=gross_gap,
    )

    # Use MC's buffer trajectory for all gap calculations (consistency)
    buffer_traj = mc_result.buffer_trajectory
    net_gap_traj = compute_net_gap_trajectory(gross_gap, buffer_traj)

    # Path total gaps and trajectories — use MC actual per-path T means
    path_t = {
        p: max(1, int(round(mc_result.path_t_means.get(p, fallback))))
        for p, fallback in [("A", 28), ("B", 84), ("C", 180)]
    }
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
    # Schelling signals live HERE (M5), not in ACH. M5 has the mature game-theory
    # framework (credibility, focal convergence). ACH handles physical/market evidence only.
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

    # Expected total gap — use MC physical weights × MC sample means (consistent pair)
    # M5-adjusted weights shift path probabilities but cannot change the physical gap
    # within each path, so mixing M5 weights with MC means is a category error.
    mc_means = mc_result.path_total_gap_means
    expected_gap = (
        mc_base.a * mc_means["A"]
        + mc_base.b * mc_means["B"]
        + mc_base.c * mc_means["C"]
    )

    # Consistency flags
    flags = _check_consistency(posterior, params, gross_gap)

    so = SystemOutput(
        timestamp=datetime.now(),
        ach_posterior=posterior,
        t1_percentiles=mc_result.t1_percentiles,
        t2_percentiles=mc_result.t2_percentiles,
        t_total_percentiles=mc_result.t_percentiles,
        t_weighted_mean=mc_result.t_weighted_mean,
        buffer_trajectory=buffer_traj,
        gross_gap_mbd=gross_gap,
        net_gap_trajectories=net_gap_trajectories,
        path_probabilities=path_probs,
        path_total_gaps=path_total_gaps,
        expected_total_gap=expected_gap,
        consistency_flags=flags,
    )
    return so, mc_result, ach_log_odds


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

    # Compute conflict day for LLM context
    conflict_start = config.get("conflict", {}).get("start_date", "2026-03-01")
    try:
        conflict_day = (datetime.now() - datetime.strptime(conflict_start, "%Y-%m-%d")).days
    except (ValueError, TypeError):
        conflict_day = None

    # Fetch previous observation values as baseline for LLM
    previous_obs: dict[str, float] | None = None
    try:
        from datetime import timedelta
        recent = get_observations(db_path, since=datetime.now() - timedelta(hours=12))
        if recent:
            # Keep latest value per obs ID
            prev: dict[str, float] = {}
            for o in sorted(recent, key=lambda x: x.timestamp):
                prev[o.id] = o.value
            if prev:
                previous_obs = prev
    except Exception:
        pass

    # Step 2: LLM observation + Schelling signal extraction
    from hormuz.core.m5_game import SignalEvidence as _SE
    llm_signals: list[_SE] = []
    try:
        from hormuz.infra.llm import create_llm_backend
        llm_config = config.get("llm", {})
        backend_type = llm_config.get("backend", "claude_api")
        backend_kwargs = llm_config.get(backend_type, {})
        llm = create_llm_backend(backend_type, **backend_kwargs)
        parsed = parse_readwise_articles(articles)

        # Dedup: skip articles already processed in DB
        candidate_ids = {a["id"] for a in parsed if a.get("id")}
        existing_ids = get_article_ids(db_path, candidate_ids) if candidate_ids else set()
        new_parsed = [a for a in parsed if a.get("id") not in existing_ids]

        if new_parsed:
            extraction = await extract_observations(
                new_parsed, llm=llm, batch_size=5,
                conflict_day=conflict_day,
                previous_obs=previous_obs,
            )
            llm_obs = extraction.observations
            llm_signals = extraction.signals
            result["parameter_updates"] = extraction.parameter_updates

            # Store articles and provenance
            batch_run = datetime.now().strftime("%Y%m%d-%H%M%S")
            insert_articles(db_path, new_parsed)
            if extraction.provenance:
                insert_article_observations(db_path, extraction.provenance, batch_run=batch_run)
        else:
            llm_obs = []

        result["articles_total"] = len(parsed)
        result["articles_new"] = len(new_parsed)
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

    # A6/O14 check: H3 unfreeze if external resupply evidence detected
    # O14 >= 0.7 = strong evidence (satellite/photo or confirmed), triggers H3 unfreeze
    o14 = next((o for o in all_obs if o.id == "O14" and o.value >= 0.7), None)

    # Extract events from observations for M2 duration model.
    # NOTE: O06 and O10 also feed ACH (M1). This is intentional dual use:
    # - In ACH: O06/O10 assess capability persistence (H1 vs H2) → affects T1
    # - Here: O06 low → E2 (sweep fleet attacked, +14d T2); O10 low → E3 (mines confirmed, +7d T2)
    # The two effects operate on different phases (T1 vs T2) and are physically independent.
    events: dict[str, bool] = {}
    # E2 (minesweeper attack): O06 sharp drop indicates coastal nodes destroyed
    o06_val = next((o.value for o in all_obs if o.id == "O06"), None)
    if o06_val is not None and o06_val < 0.4:
        events["E2"] = True
    # E3 (mine strike on vessel): O10 near-zero + O01 high = active mine threat
    o10_val = next((o.value for o in all_obs if o.id == "O10"), None)
    o01_val = next((o.value for o in all_obs if o.id == "O01"), None)
    if o10_val is not None and o10_val < 0.1 and o01_val is not None and o01_val > 0.6:
        events["E3"] = True

    # Step 3: Engine run
    try:
        constants = load_constants(configs_dir / "constants.yaml")
        params = load_parameters(configs_dir / "parameters.yaml")
        # Apply LLM-extracted parameter updates (e.g., sweep_ships from MCM deployment news)
        _PARAM_MAP = {"sweep_ships": "sweep_ships"}
        for pu in result.get("parameter_updates", []):
            field_name = _PARAM_MAP.get(pu.get("param"))
            if field_name and hasattr(params, field_name):
                try:
                    params = params.override(**{field_name: int(float(pu["value"]))})
                except (TypeError, ValueError, KeyError):
                    result.setdefault("errors", []).append(
                        f"Invalid param update: {pu}"
                    )
        # H3 unfreeze: O14 >= 0.7 = strong external resupply evidence → 3-way ACH
        if o14:
            params = params.override(h3_suspended=False)
            result["h3_unfrozen"] = True
        controls = get_controls(db_path)
        confidence = compute_confidence_level(db_path)
        # Load persisted ACH state for cross-run accumulation
        from hormuz.infra.db import get_ach_state, save_ach_state, ACHState
        prev_ach = get_ach_state(db_path)
        so, mc_result, ach_log_odds = engine_run(
            constants, params, all_obs, controls, events=events,
            mc_n=config.get("mc", {}).get("n", 10000),
            seed=config.get("mc", {}).get("seed", 42),
            o01_trend=o01_trend,
            schelling_signals=llm_signals,
            prior_log_odds=prev_ach.log_odds if prev_ach else None,
            prior_h3_suspended=prev_ach.h3_suspended if prev_ach else None,
            prior_h3_posterior=prev_ach.h3_posterior if prev_ach else None,
        )
        # Persist updated ACH state
        save_ach_state(db_path, ACHState(
            log_odds=ach_log_odds,
            h3_suspended=params.h3_suspended,
            h3_posterior=so.ach_posterior.h3,
        ))
        # Build full signal list for reporter (LLM + control-derived)
        from hormuz.core.m5_game import SignalEvidence
        all_signals = list(llm_signals) if llm_signals else []
        for ctrl in controls:
            if ctrl.triggered and ctrl.effect:
                all_signals.append(SignalEvidence(key=ctrl.effect, evidence=0.5))
        result["schelling_signals"] = all_signals
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
        # Count consecutive days Brent closed below $80 (stop-loss rule)
        brent_below_80_days = 0
        try:
            import yfinance as yf
            brent_hist = yf.Ticker("BZ=F").history(period="7d")
            if not brent_hist.empty:
                for close in reversed(brent_hist["Close"].tolist()):
                    if close < 80:
                        brent_below_80_days += 1
                    else:
                        break
        except Exception:
            pass  # fail open — don't block position eval for market data issues
        pos = evaluate_positions(
            so, brent_price=brent_price,
            brent_below_80_days=brent_below_80_days,
        )
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
            conflict_start=config.get("conflict", {}).get("start_date", "2026-03-01"),
            brent_price=brent_price,
            position_result=result.get("positions"),
            game_signals=result.get("schelling_signals", []),
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
