"""Click CLI — 8 commands for the Hormuz Decision System."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import click
import yaml


def _project_root() -> Path:
    """Resolve project root (contains pyproject.toml)."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _load_config(config_path: Path | None = None) -> dict:
    """Load config from YAML file, inject resolved paths."""
    root = _project_root()
    if config_path is None:
        config_path = root / "configs" / "config.yaml"
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text()) or {}
    else:
        cfg = {}
    # Resolve configs_dir and db.path relative to project root
    cfg["configs_dir"] = str(root / cfg.get("configs_dir", "configs"))
    if "db" in cfg:
        cfg["db"]["path"] = str(root / cfg["db"]["path"])
    return cfg


@click.group()
def cli():
    """Hormuz Decision System v5.4 — 霍尔木兹海峡危机投资决策操作系统"""
    pass


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--mc-n", default=10000, help="Monte Carlo sample count")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
def run(config_path, mc_n, seed):
    """Run full pipeline — fetch data, analyze, compute, report."""
    from hormuz.app.pipeline import run_pipeline

    config = _load_config(config_path)
    if mc_n != 10000:
        config.setdefault("mc", {})["n"] = mc_n
    if seed is not None:
        config.setdefault("mc", {})["seed"] = seed

    result = asyncio.run(run_pipeline(config))

    steps = result.get("steps_completed", 0)
    errors = result.get("errors", [])
    click.echo(f"Pipeline completed: {steps}/7 steps")
    if errors:
        for e in errors:
            click.echo(f"  ⚠ {e}", err=True)
    if "system_output" in result:
        so = result["system_output"]
        if so.confidence_level != "normal":
            click.echo(f"  Confidence: {so.confidence_level.upper()}")
        click.echo(f"  GrossGap: {so.gross_gap_mbd:.1f} mbd")
        click.echo(f"  ACH: H1={so.ach_posterior.h1:.2f} H2={so.ach_posterior.h2:.2f}")
        click.echo(f"  Paths: A={so.path_probabilities.a:.0%} B={so.path_probabilities.b:.0%} C={so.path_probabilities.c:.0%}")


@cli.command()
@click.option("--db-path", type=click.Path(path_type=Path), default="data/hormuz.db")
def status(db_path):
    """Show latest system output summary."""
    from hormuz.infra.db import get_latest_output

    so = get_latest_output(db_path)
    if so is None:
        click.echo("No data — run `hormuz run` first or `hormuz init-db`")
        return

    click.echo(f"Timestamp: {so.timestamp}")
    click.echo(f"GrossGap: {so.gross_gap_mbd:.1f} mbd")
    click.echo(f"ACH: H1={so.ach_posterior.h1:.2f} H2={so.ach_posterior.h2:.2f} → {so.ach_posterior.dominant}")
    click.echo(f"T total p50: {so.t_total_percentiles.get('p50', '?')} days")
    click.echo(f"Paths: A={so.path_probabilities.a:.0%} B={so.path_probabilities.b:.0%} C={so.path_probabilities.c:.0%}")
    click.echo(f"Expected TotalGap: {so.expected_total_gap:.0f} mbd·days")
    if so.consistency_flags:
        click.echo("Flags:")
        for f in so.consistency_flags:
            click.echo(f"  ⚠ {f}")


@cli.command("init-db")
@click.option("--db-path", type=click.Path(path_type=Path), default="data/hormuz.db")
def init_db_cmd(db_path):
    """Initialize SQLite database."""
    from hormuz.infra.db import init_db

    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(db_path)
    click.echo(f"Database initialized: {db_path}")


@cli.command()
@click.option("--db-path", type=click.Path(path_type=Path), default="data/hormuz.db")
@click.argument("obs_id")
@click.argument("value", type=float)
@click.option("--source", default="manual")
def record(db_path, obs_id, value, source):
    """Record a manual observation."""
    from datetime import datetime
    from hormuz.core.types import Observation
    from hormuz.infra.db import insert_observation

    obs = Observation(id=obs_id, timestamp=datetime.now(), value=value, source=source)
    insert_observation(db_path, obs)
    click.echo(f"Recorded: {obs_id}={value} from {source}")


@cli.command()
@click.option("--db-path", type=click.Path(path_type=Path), default="data/hormuz.db")
@click.option("--n", default=10000, help="Sample count")
@click.option("--seed", default=42, type=int)
def mc(db_path, n, seed):
    """Run standalone Monte Carlo simulation."""
    from hormuz.core.types import ACHPosterior, Parameters
    from hormuz.core.mc import run_monte_carlo
    import numpy as np

    params = Parameters()
    posterior = ACHPosterior(h1=0.5, h2=0.5, h3=None)
    result = run_monte_carlo(posterior, params, {}, n=n, seed=seed)

    click.echo(f"MC N={n}, seed={seed}")
    click.echo(f"  T p10={result.t_percentiles['p10']:.0f} p50={result.t_percentiles['p50']:.0f} p90={result.t_percentiles['p90']:.0f}")
    click.echo(f"  Paths: A={result.path_frequencies['A']:.0%} B={result.path_frequencies['B']:.0%} C={result.path_frequencies['C']:.0%}")


@cli.command()
@click.option("--output", type=click.Path(path_type=Path), default="data/status.html")
def report(output):
    """Generate HTML status report."""
    click.echo(f"Report generation: use `hormuz run` for full pipeline (includes report).")


@cli.command()
@click.option("--db-path", type=click.Path(path_type=Path), default="data/hormuz.db")
@click.argument("param")
@click.argument("new_value")
def override(db_path, param, new_value):
    """Override a parameter with DB logging."""
    from hormuz.infra.db import save_parameter_override

    save_parameter_override(db_path, param=param, old_value="(see DB)", new_value=new_value)
    click.echo(f"Parameter override logged: {param} → {new_value}")


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--days", default=7, help="Number of days to backfill")
@click.option("--batch-size", default=5, help="Articles per LLM batch")
def backfill(config_path, days, batch_size):
    """Backfill observations from historical Readwise articles.

    Fetches articles from the past N days, groups by date,
    runs LLM extraction per day, and inserts into DB with correct timestamps.
    This builds the historical baseline needed for O01 trend, O02 computation, etc.
    """
    asyncio.run(_backfill(config_path, days, batch_size))


async def _backfill(config_path, days, batch_size):
    from datetime import timedelta
    from collections import defaultdict
    from hormuz.infra.ingester import fetch_readwise_articles, parse_readwise_articles
    from hormuz.infra.analyzer import extract_observations
    from hormuz.infra.llm import create_llm_backend
    from hormuz.infra.db import init_db, insert_observations

    config = _load_config(config_path)
    db_path = Path(config["db"]["path"])
    init_db(db_path)

    rw = config["readwise"]
    sources = set(rw["sources"]) if "sources" in rw else None
    cutoff = datetime.now() - timedelta(days=days)

    click.echo(f"Fetching articles from past {days} days...")
    articles = await fetch_readwise_articles(
        token=rw["token"],
        sources=sources,
        proxy=rw.get("proxy"),
        timeout=rw.get("timeout", 30),
        limit=200,
        updated_after=cutoff.strftime("%Y-%m-%dT00:00:00"),
    )

    # Filter to articles within date range and group by date
    parsed = parse_readwise_articles(articles)
    by_date: dict[str, list[dict]] = defaultdict(list)
    for a in parsed:
        pub = a.get("published_date") or ""
        if pub >= cutoff.strftime("%Y-%m-%d"):
            date_key = pub[:10] if pub else "unknown"
            by_date[date_key].append(a)

    if not by_date:
        click.echo("No articles found in date range.")
        return

    click.echo(f"Found {sum(len(v) for v in by_date.values())} articles across {len(by_date)} days")

    # LLM setup
    llm_config = config.get("llm", {})
    backend_type = llm_config.get("backend", "claude_api")
    backend_kwargs = llm_config.get(backend_type, {})
    llm = create_llm_backend(backend_type, **backend_kwargs)

    total_obs = 0
    for date_str in sorted(by_date.keys()):
        day_articles = by_date[date_str][:30]
        try:
            ts = datetime.fromisoformat(date_str + "T12:00:00")
        except ValueError:
            continue

        if len(day_articles) < 3:
            click.echo(f"  {date_str}: {len(day_articles)} articles — skipped (min 3)")
            continue

        click.echo(f"  {date_str}: {len(day_articles)} articles...", nl=False)
        try:
            extraction = await extract_observations(
                day_articles, llm=llm, timestamp=ts, batch_size=batch_size,
            )
            obs = extraction.observations
            if obs:
                insert_observations(db_path, obs)
                total_obs += len(obs)
                ids = sorted(set(o.id for o in obs))
                sigs = extraction.signals
                sig_str = f" + signals: {', '.join(s.key for s in sigs)}" if sigs else ""
                click.echo(f" {len(obs)} obs ({', '.join(ids)}){sig_str}")
            else:
                click.echo(" 0 obs")
        except Exception as e:
            click.echo(f" error: {e}")

    click.echo(f"Backfill complete: {total_obs} observations inserted across {len(by_date)} days")


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), default=None)
def validate(config_path):
    """Run consistency checks on current state."""
    config = _load_config(config_path)
    db_path = Path(config.get("db", {}).get("path", "data/hormuz.db"))
    issues: list[str] = []
    warnings: list[str] = []

    # 1. DB exists
    if not db_path.exists():
        click.echo(click.style("FAIL", fg="red") + f": database not found at {db_path}")
        raise SystemExit(1)

    from hormuz.infra.db import (
        get_observations, get_history_days, compute_confidence_level,
        get_latest_output,
    )

    # 2. Observation coverage
    all_ids = {f"O{i:02d}" for i in range(1, 15)}
    obs = get_observations(db_path)
    present_ids = {o.id for o in obs}
    missing = sorted(all_ids - present_ids)
    if missing:
        warnings.append(f"missing observations: {', '.join(missing)}")

    # 3. Freshness
    if obs:
        latest_ts = max(o.timestamp for o in obs)
        age_hours = (datetime.now() - latest_ts).total_seconds() / 3600
        if age_hours > 8:
            issues.append(f"stale data: latest observation is {age_hours:.0f}h old")
        elif age_hours > 4:
            warnings.append(f"data aging: latest observation is {age_hours:.1f}h old")

    # 4. History depth & confidence
    days = get_history_days(db_path)
    confidence = compute_confidence_level(db_path)
    if confidence == "burn_in":
        warnings.append(f"BURN-IN: only {days} days of history (<3 needed for reliable output)")
    elif confidence == "low":
        warnings.append(f"low confidence: {days} days of history (need >7 for normal)")

    # 5. Latest system output
    so = get_latest_output(db_path)
    if so is None:
        issues.append("no system output found — run `hormuz run` first")
    else:
        # Consistency flags from engine
        for flag in so.consistency_flags:
            issues.append(f"engine flag: {flag}")
        # Path weights sum
        pw = so.path_probabilities
        total = pw.a + pw.b + pw.c
        if abs(total - 1.0) > 0.02:
            issues.append(f"path weights don't sum to 1: A={pw.a} B={pw.b} C={pw.c} (sum={total:.3f})")
        # T percentile ordering
        p = so.t_total_percentiles
        if p.get("p10", 0) > p.get("p50", 0) or p.get("p50", 0) > p.get("p90", 0):
            issues.append(f"T percentile inversion: p10={p.get('p10')} p50={p.get('p50')} p90={p.get('p90')}")

    # 6. Report
    click.echo(f"DB: {db_path} ({days} days, confidence={confidence})")
    click.echo(f"Observations: {len(obs)} total, {len(present_ids)}/{len(all_ids)} O-series covered")
    if so:
        click.echo(f"Latest output: {so.timestamp.strftime('%Y-%m-%d %H:%M')}")

    if not issues and not warnings:
        click.echo(click.style("OK", fg="green") + " — all checks passed")
    else:
        for w in warnings:
            click.echo(click.style("WARN", fg="yellow") + f": {w}")
        for i in issues:
            click.echo(click.style("ISSUE", fg="red") + f": {i}")
        if issues:
            raise SystemExit(1)
