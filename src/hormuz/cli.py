"""CLI entry point for hormuz-ds."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

# Project root: two levels up from src/hormuz/cli.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
def cli() -> None:
    """Hormuz 决策辅助系统."""
    load_dotenv()


@cli.command()
@click.option("-v", "--verbose", is_flag=True, help="Debug logging")
@click.option(
    "--root",
    type=click.Path(exists=True, path_type=Path),
    default=_PROJECT_ROOT,
    help="Project root directory",
)
def run(verbose: bool, root: Path) -> None:
    """Run the full 7-step pipeline once."""
    _setup_logging(verbose)
    from hormuz.pipeline import Pipeline

    pipeline = Pipeline(
        config_dir=root / "configs",
        data_dir=root / "data",
        reports_dir=root / "reports",
        template_dir=root / "templates",
    )
    result = asyncio.run(pipeline.run())

    # Summary
    n_steps = len(result["steps_completed"])
    click.echo(f"Pipeline: {n_steps}/7 steps | {result['observations_count']} observations")

    if result.get("engine_result"):
        er = result["engine_result"]
        pw = er.get("path_weights")
        if pw:
            click.echo(f"  Path A={pw.a*100:.0f}%  B={pw.b*100:.0f}%  C={pw.c*100:.0f}%")
        click.echo(f"  Q1={er.get('q1_regime', '?')}  Q2={er.get('q2_regime', '?')}  Week={er.get('crisis_week', '?')}")

    if result["errors"]:
        click.secho(f"  Errors: {list(result['errors'].keys())}", fg="red")
    else:
        click.secho("  All OK", fg="green")


@cli.command()
@click.option(
    "--root",
    type=click.Path(exists=True, path_type=Path),
    default=_PROJECT_ROOT,
)
def status(root: Path) -> None:
    """Generate status.html without running the full pipeline."""
    _setup_logging(False)
    from hormuz.db import HormuzDB
    from hormuz.reporter import Reporter

    db = HormuzDB(root / "data" / "hormuz.db")
    reporter = Reporter(db, root / "templates", root / "data", root / "reports", config_dir=root / "configs")
    out = reporter.update_status(docs_dir=root / "docs")
    click.echo(f"Status written to {out} + docs/index.html")


@cli.command()
@click.option(
    "--root",
    type=click.Path(exists=True, path_type=Path),
    default=_PROJECT_ROOT,
)
def weekly(root: Path) -> None:
    """Force-generate weekly report (normally auto on Wednesdays)."""
    _setup_logging(False)
    from hormuz.db import HormuzDB
    from hormuz.reporter import Reporter

    db = HormuzDB(root / "data" / "hormuz.db")
    reporter = Reporter(db, root / "templates", root / "data", root / "reports", config_dir=root / "configs")
    out = reporter.archive_weekly(force=True)
    if out:
        click.echo(f"Weekly report written to {out}")


if __name__ == "__main__":
    cli()
