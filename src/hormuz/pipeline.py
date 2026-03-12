"""Pipeline orchestrator — 7-step execution flow for the Hormuz decision system."""
from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from hormuz.analyzer import Analyzer
from hormuz.db import HormuzDB
from hormuz.engine.ach import ACHEngine
from hormuz.engine.physical import PhysicalLayer
from hormuz.engine.schelling import SchellingSheet
from hormuz.engine.signals import SignalEngine
from hormuz.ingester import MarketIngester, ReadwiseIngester
from hormuz.llm import get_backend
from hormuz.models import Observation, PathWeights
from hormuz.reporter import Reporter

logger = logging.getLogger(__name__)

_ENV_RE = re.compile(r"\$\{(\w+)\}")


def _load_yaml_with_env(path: Path) -> dict:
    """Load YAML and expand ${VAR} references from environment."""
    text = path.read_text()
    text = _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), text)
    return yaml.safe_load(text)


class Pipeline:
    """Orchestrates the 7-step Hormuz analysis pipeline."""

    def __init__(
        self,
        config_dir: Path,
        data_dir: Path,
        reports_dir: Path,
        template_dir: Path,
    ) -> None:
        """Load configs and initialize all components."""
        config = _load_yaml_with_env(config_dir / "config.yaml")
        constants = yaml.safe_load((config_dir / "constants.yaml").read_text())
        parameters = yaml.safe_load((config_dir / "parameters.yaml").read_text())

        # Initialize DB
        db_path = data_dir / "hormuz.db"
        self.db = HormuzDB(db_path)

        # Initialize components
        self.readwise = ReadwiseIngester(config["readwise"])
        self.market = MarketIngester()
        self.backend = get_backend(config)
        self.analyzer = Analyzer(self.backend)
        self.signals = SignalEngine(self.db)
        h3_suspended = parameters.get("physical", {}).get("q1", {}).get("h3_suspended", True)
        self.ach = ACHEngine(self.db, constants["ach"], h3_suspended=h3_suspended)
        self.physical = PhysicalLayer(parameters["physical"])
        self.schelling = SchellingSheet(constants["schelling"])
        self.reporter = Reporter(self.db, template_dir, data_dir, reports_dir, config_dir=config_dir)
        self.docs_dir = config_dir.parent / "docs"

        self.config = config
        self.parameters = parameters

    async def run(self) -> dict:
        """Execute the 7-step pipeline.

        Each step is wrapped in try/except — failure of one step
        doesn't crash the pipeline.
        """
        now = datetime.now(UTC)
        steps_completed: list[str] = []
        errors: dict[str, str] = {}
        articles: list[dict] = []
        all_observations: list[Observation] = []

        # Step 1: Auto-revert expired tripwires
        try:
            reverted = self.signals.check_reverts(now)
            steps_completed.append("revert_check")
            if reverted:
                logger.info("Reverted %d expired tripwires", len(reverted))
        except Exception as e:
            logger.exception("Step 1 (revert_check) failed")
            errors["revert_check"] = str(e)

        # Step 2: Ingest
        try:
            articles, market_obs = await self._step_ingest()
            all_observations.extend(market_obs)
            steps_completed.append("ingest")
        except Exception as e:
            logger.exception("Step 2 (ingest) failed")
            errors["ingest"] = str(e)

        # Step 3: Analyze (LLM extraction)
        try:
            llm_obs = await self._step_analyze(articles)
            all_observations.extend(llm_obs)
            steps_completed.append("analyze")
        except Exception as e:
            logger.exception("Step 3 (analyze) failed")
            errors["analyze"] = str(e)

        # Step 4: Signal scan (BEFORE engine — penetration semantics)
        try:
            triggered = self.signals.scan(all_observations)
            steps_completed.append("signals")
            if triggered:
                logger.info("Triggered %d signals: %s", len(triggered),
                           [s.signal_id for s in triggered])
        except Exception as e:
            logger.exception("Step 4 (signals) failed")
            errors["signals"] = str(e)

        # Step 5: Engine update (ACH → physical → schelling → MC → positions)
        engine_result: dict | None = None
        try:
            engine_result = self._step_engine(all_observations)
            steps_completed.append("engine")
        except Exception as e:
            logger.exception("Step 5 (engine) failed")
            errors["engine"] = str(e)

        # Step 6: Report
        try:
            self.reporter.update_status(docs_dir=self.docs_dir)
            if now.weekday() == 2:  # Wednesday
                self.reporter.archive_weekly()
            steps_completed.append("report")
        except Exception as e:
            logger.exception("Step 6 (report) failed")
            errors["report"] = str(e)

        # Step 7: Notify (if tripwires triggered or regime changed)
        try:
            self._step_notify(
                triggered_signals=triggered if "signals" in steps_completed else [],
                engine_result=engine_result,
            )
            steps_completed.append("notify")
        except Exception as e:
            logger.exception("Step 7 (notify) failed")
            errors["notify"] = str(e)

        return {
            "steps_completed": steps_completed,
            "errors": errors,
            "timestamp": now.isoformat(),
            "observations_count": len(all_observations),
            "engine_result": engine_result,
        }

    async def _step_ingest(self) -> tuple[list[dict], list[Observation]]:
        """Step 2: Fetch from Readwise + market APIs.

        Returns (articles, market_observations).
        Market observations are immediately written to DB.
        """
        since = datetime.now(UTC) - timedelta(hours=4)
        articles = await self.readwise.fetch(since=since)
        market_obs = self.market.fetch()

        # Write market observations to DB immediately
        for obs in market_obs:
            self.db.insert_observation(obs)

        return articles, market_obs

    async def _step_analyze(self, articles: list[dict]) -> list[Observation]:
        """Step 3: LLM extraction. Returns observations, writes to DB."""
        observations = await self.analyzer.extract(articles)

        for obs in observations:
            self.db.insert_observation(obs)

        return observations

    def _step_engine(self, new_observations: list[Observation]) -> dict:
        """Step 5: Engine update — ACH → physical → schelling → path weights.

        MC pricing and position signals are Phase 2.
        """
        now = datetime.now(UTC)

        # 1. Evaluate regimes from DB evidence
        q1_regime = self.ach.evaluate_regime("q1", as_of=now)
        q2_regime = self.ach.evaluate_regime("q2", as_of=now)

        # 2. Physical layer: regime -> parameter adjustments
        phys_params = self.physical.update_params(q1_regime, q2_regime)

        # 3. Schelling delta
        current_week = self._current_week()

        since_7d = now - timedelta(days=7)
        schelling_obs = self.db.get_observations_since(since_7d, category="schelling")
        active_schelling: dict[int, bool] = {}
        for obs in schelling_obs:
            try:
                sig_id = int(obs.key)
                active_schelling[sig_id] = True
            except (ValueError, TypeError):
                pass

        delta = self.schelling.compute_delta(active_schelling, current_week=current_week)

        # 4. Path weights: base + schelling delta
        base_weights = PathWeights(
            a=self.parameters["paths"]["a"],
            b=self.parameters["paths"]["b"],
            c=self.parameters["paths"]["c"],
        )
        adjusted_weights = base_weights.apply_delta(
            a_delta=delta["a"], c_delta=delta["c"]
        )

        return {
            "q1_regime": q1_regime,
            "q2_regime": q2_regime,
            "phys_params": phys_params,
            "path_weights": adjusted_weights,
            "schelling_delta": delta,
            "crisis_week": current_week,
        }

    def _step_notify(
        self,
        triggered_signals: list | None = None,
        engine_result: dict | None = None,
    ) -> None:
        """Step 7: Notify if tripwires triggered or regime changed.

        Currently logs only. Future: Telegram push via OpenClaw.
        """
        if triggered_signals:
            for sig in triggered_signals:
                logger.warning(
                    "SIGNAL TRIGGERED: %s — %s", sig.signal_id, sig.action_taken
                )

        if engine_result:
            pw = engine_result.get("path_weights")
            if pw:
                logger.info(
                    "Path weights: A=%.0f%% B=%.0f%% C=%.0f%%",
                    pw.a * 100, pw.b * 100, pw.c * 100,
                )
            logger.info(
                "Regimes: Q1=%s, Q2=%s",
                engine_result.get("q1_regime", "?"),
                engine_result.get("q2_regime", "?"),
            )

    def _current_week(self) -> int:
        """Compute current crisis week from config start_date."""
        start_str = self.config["conflict"]["start_date"]
        start = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=UTC)
        now = datetime.now(UTC)
        delta = now - start
        return max(1, int(delta.days / 7) + 1)
