"""HTML report generator for the Hormuz decision support system."""
from datetime import UTC, datetime, timedelta
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from hormuz.db import HormuzDB
from hormuz.models import SignalStatus

# Conflict start date — used to calculate current week number
_CONFLICT_START = datetime(2025, 6, 1, tzinfo=UTC)

# Signal status → CSS dot color class suffix
_STATUS_COLORS = {
    "inactive": "grey",
    "triggered": "yellow",
    "confirmed": "green",
    "reverted": "dark",
}


def _status_color(status: str) -> str:
    """Map signal status to dot color class suffix."""
    return _STATUS_COLORS.get(status, "grey")


class Reporter:
    """Generates status.html (every 4h) and weekly archive reports."""

    def __init__(
        self,
        db: HormuzDB,
        template_dir: Path,
        output_dir: Path,
        reports_dir: Path,
    ) -> None:
        self.db = db
        self.output_dir = output_dir
        self.reports_dir = reports_dir
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=False,
        )
        self._env.globals["status_color"] = _status_color

    def update_status(self) -> Path:
        """Generate/overwrite status.html with current system state."""
        data = self._gather_status_data()
        data["is_weekly"] = False
        html = self._render("status.html.jinja", data)
        out = self.output_dir / "status.html"
        out.write_text(html, encoding="utf-8")
        return out

    def archive_weekly(self, force: bool = False) -> Path | None:
        """Generate weekly archive report.

        Called on Wednesdays. Generates reports/YYYY-WNN.html.
        Returns path or None if not Wednesday (unless force=True).
        """
        now = datetime.now(UTC)
        if not force and now.weekday() != 2:  # 2 = Wednesday
            return None

        data = self._gather_status_data()
        data["is_weekly"] = True

        # Week label
        iso_cal = now.isocalendar()
        week_label = f"{iso_cal.year}-W{iso_cal.week:02d}"
        data["week_label"] = week_label

        # Weekly extras: ACH evidence added this week
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        q1_ach = self.db.get_ach_evidence("q1", since=week_start)
        q2_ach = self.db.get_ach_evidence("q2", since=week_start)
        data["weekly_ach"] = q1_ach + q2_ach

        html = self._render("status.html.jinja", data)
        out = self.reports_dir / f"{week_label}.html"
        out.write_text(html, encoding="utf-8")
        return out

    def _gather_status_data(self) -> dict:
        """Collect all data needed for the template."""
        now = datetime.now(UTC)

        # Regimes
        q1_regime = self.db.get_latest_regime("q1")
        q2_regime = self.db.get_latest_regime("q2")

        # MC params + path weights
        mc_params = self.db.get_latest_mc_params()
        path_weights = mc_params.path_weights if mc_params else None

        # MC result (latest)
        mc_result = self._get_latest_mc_result()

        # Active signals → build signal_id → status map
        active_signals = self.db.get_active_signals()
        signal_map: dict[str, str] = {}
        for sig in active_signals:
            signal_map[sig.signal_id] = sig.status.value

        # Recent observations (last 7 days)
        since_7d = now - timedelta(days=7)
        recent_observations = self.db.get_observations_since(since_7d)

        # Latest Brent price
        latest_brent = None
        market_obs = self.db.get_observations_since(since_7d, category="market")
        for obs in reversed(market_obs):
            if obs.key == "brent_close":
                latest_brent = obs.value
                break

        # Schelling observations
        schelling_observations = self.db.get_observations_since(since_7d, category="schelling")

        # Unexecuted position signals
        positions = self.db.get_unexecuted_position_signals()

        # Week number
        delta = now - _CONFLICT_START
        week_number = max(1, int(delta.days / 7) + 1)

        return {
            "generated_at": now.strftime("%Y-%m-%d %H:%M UTC"),
            "week_number": week_number,
            "q1_regime": q1_regime,
            "q2_regime": q2_regime,
            "mc_params": mc_params,
            "path_weights": path_weights,
            "mc_result": mc_result,
            "signals": active_signals,
            "signal_map": signal_map,
            "recent_observations": recent_observations,
            "latest_brent": latest_brent,
            "schelling_observations": schelling_observations,
            "positions": positions,
        }

    def _render(self, template_name: str, data: dict) -> str:
        """Render Jinja2 template with data."""
        template = self._env.get_template(template_name)
        return template.render(**data)

    def _get_latest_mc_result(self):
        """Get latest MC result from DB."""
        from hormuz.models import MCResult
        row = self.db._conn.execute(
            "SELECT * FROM mc_results ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        import json
        from hormuz.db import _dt
        output = json.loads(row["output"])
        return MCResult(
            id=row["id"],
            timestamp=_dt(row["timestamp"]),
            params_id=row["params_id"],
            price_mean=output["price_mean"],
            price_p10=output["price_p10"],
            price_p50=output["price_p50"],
            price_p90=output["price_p90"],
            path_a_price=output["path_a_price"],
            path_b_price=output["path_b_price"],
            path_c_price=output["path_c_price"],
            key_dates=output.get("key_dates"),
        )
