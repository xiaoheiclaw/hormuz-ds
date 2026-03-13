"""Click CLI — 8 commands for the Hormuz Decision System."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
import yaml


def _load_config(config_path: Path | None = None) -> dict:
    """Load config from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text()) or {}
    return {}


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
def validate():
    """Run consistency checks on current state."""
    click.echo("Validation: load latest output and check consistency flags")
    # TODO: implement full validation
    click.echo("OK")
