"""HTML report generator for the Hormuz decision support system."""
from datetime import UTC, datetime, timedelta
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from hormuz.db import HormuzDB
from hormuz.models import RegimeType, SignalStatus

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
        config_dir: Path | None = None,
    ) -> None:
        self.db = db
        self.output_dir = output_dir
        self.reports_dir = reports_dir
        self.config_dir = config_dir
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=False,
        )
        self._env.globals["status_color"] = _status_color

    def update_status(self, docs_dir: Path | None = None) -> Path:
        """Generate/overwrite status.html with current system state."""
        data = self._gather_status_data()
        data["is_weekly"] = False
        html = self._render("status.html.jinja", data)
        out = self.output_dir / "status.html"
        out.write_text(html, encoding="utf-8")
        # Also write to docs/index.html for GitHub Pages
        if docs_dir:
            pages_out = docs_dir / "index.html"
            pages_out.write_text(html, encoding="utf-8")
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

        # Path weights from latest MC params (if any)
        mc_params = self.db.get_latest_mc_params()
        path_weights = mc_params.path_weights if mc_params else None

        # Physical params: derive from current regimes
        from hormuz.engine.physical import PhysicalLayer
        phys_params = None
        try:
            # Read parameters.yaml to get physical config
            import yaml
            params_path = (self.config_dir or self.output_dir.parent / "configs") / "parameters.yaml"
            if params_path.exists():
                parameters = yaml.safe_load(params_path.read_text())
                physical = PhysicalLayer(parameters["physical"])
                q1_r = q1_regime.regime if q1_regime else RegimeType.wide
                q2_r = q2_regime.regime if q2_regime else RegimeType.wide
                phys_params = physical.update_params(q1_r, q2_r)
        except Exception:
            pass

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

        # Week number
        delta = now - _CONFLICT_START
        week_number = max(1, int(delta.days / 7) + 1)

        return {
            "generated_at": now.strftime("%Y-%m-%d %H:%M UTC"),
            "week_number": week_number,
            "q1_regime": q1_regime,
            "q2_regime": q2_regime,
            "path_weights": path_weights,
            "phys_params": phys_params,
            "signals": active_signals,
            "signal_map": signal_map,
            "recent_observations": recent_observations,
            "latest_brent": latest_brent,
            "schelling_observations": schelling_observations,
        }

    def _render(self, template_name: str, data: dict) -> str:
        """Render Jinja2 template with data."""
        template = self._env.get_template(template_name)
        return template.render(**data)
