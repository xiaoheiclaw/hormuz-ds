"""SQLite storage — 8 tables, CRUD for all variable types.

Each function takes Path as first arg, opens/closes connection per call.
SystemOutput serialized as JSON blob.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from hormuz.core.types import (
    ACHPosterior,
    Control,
    Observation,
    PathWeights,
    SystemOutput,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id TEXT,
    timestamp TEXT,
    value REAL,
    source TEXT,
    noise_note TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ach_evidence (
    obs_id TEXT,
    direction TEXT,
    lr_h1 REAL,
    lr_h2 REAL,
    timestamp TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS state_snapshots (
    timestamp TEXT,
    data_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS controls (
    id TEXT,
    actor TEXT,
    triggered INTEGER,
    trigger_time TEXT,
    effect TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS mc_runs (
    timestamp TEXT,
    n_samples INTEGER,
    seed INTEGER,
    result_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS system_outputs (
    timestamp TEXT,
    data_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS position_signals (
    timestamp TEXT,
    signal_type TEXT,
    action TEXT,
    executed INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS parameters_override (
    param TEXT,
    old_value TEXT,
    new_value TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    title TEXT,
    title_zh TEXT,
    source TEXT,
    url TEXT,
    summary TEXT,
    published_date TEXT,
    fetched_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS article_observations (
    article_id TEXT,
    obs_id TEXT,
    confidence TEXT,
    batch_run TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS article_attribution (
    title TEXT,
    obs_id TEXT,
    delta REAL,
    batch_run TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_ts TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    resolve_by TEXT,
    resolution_criteria TEXT,
    resolved INTEGER DEFAULT 0,
    actual REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_obs_id_ts ON observations(id, timestamp);
CREATE INDEX IF NOT EXISTS idx_sysout_created ON system_outputs(created_at);
"""


def init_db(path: Path) -> None:
    """Create all tables if they don't exist."""
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.close()


# ── Observations ──────────────────────────────────────────────────────

def insert_observation(path: Path, obs: Observation) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO observations (id, timestamp, value, source, noise_note) VALUES (?, ?, ?, ?, ?)",
        (obs.id, obs.timestamp.isoformat(), obs.value, obs.source, obs.noise_note),
    )
    conn.commit()
    conn.close()


def insert_observations(path: Path, obs_list: list[Observation]) -> None:
    """Batch insert observations."""
    conn = sqlite3.connect(path)
    conn.executemany(
        "INSERT INTO observations (id, timestamp, value, source, noise_note) VALUES (?, ?, ?, ?, ?)",
        [(o.id, o.timestamp.isoformat(), o.value, o.source, o.noise_note) for o in obs_list],
    )
    conn.commit()
    conn.close()


def get_observations(path: Path, since: datetime | None = None) -> list[Observation]:
    conn = sqlite3.connect(path)
    if since:
        rows = conn.execute(
            "SELECT id, timestamp, value, source, noise_note FROM observations WHERE timestamp >= ?",
            (since.isoformat(),),
        ).fetchall()
    else:
        rows = conn.execute("SELECT id, timestamp, value, source, noise_note FROM observations").fetchall()
    conn.close()
    return [
        Observation(
            id=r[0],
            timestamp=datetime.fromisoformat(r[1]),
            value=r[2],
            source=r[3],
            noise_note=r[4],
        )
        for r in rows
    ]


def compute_o02_from_history(path: Path, lookback_days: int = 7) -> Observation | None:
    """Compute O02 (attack trend change) from recent O01 history.

    Compares average O01 in recent half vs older half of lookback window.
    Returns None if insufficient data (<2 O01 records).
    """
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(days=lookback_days)
    o01_records = [
        o for o in get_observations(path, since=cutoff)
        if o.id == "O01"
    ]
    if len(o01_records) < 2:
        return None

    o01_records.sort(key=lambda o: o.timestamp)
    mid = len(o01_records) // 2
    older_avg = sum(o.value for o in o01_records[:mid]) / mid
    recent_avg = sum(o.value for o in o01_records[mid:]) / (len(o01_records) - mid)

    # O02: 0=rising, 0.5=stable, 1.0=sharp decline
    # If recent < older → declining → high O02
    if older_avg == 0:
        # Zero baseline → any recent activity = attacks rising (O02 low)
        change = 0.0 if recent_avg > 0 else 0.5
    else:
        ratio = (older_avg - recent_avg) / older_avg  # positive = decline
        change = max(0.0, min(1.0, 0.5 + ratio))  # map [-1,1] → [0,1] centered at 0.5

    return Observation(
        id="O02",
        timestamp=datetime.now(),
        value=round(change, 2),
        source="db:computed",
    )


def compute_o01_rolling(path: Path, window_days: int = 7) -> float | None:
    """Compute O01 7-day rolling average from DB history.

    Returns None if no O01 records in window.
    """
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(days=window_days)
    o01_records = [
        o for o in get_observations(path, since=cutoff)
        if o.id == "O01"
    ]
    if not o01_records:
        return None
    return sum(o.value for o in o01_records) / len(o01_records)


def compute_o01_trend(path: Path, window_days: int = 7) -> str:
    """Determine O01 trend direction from DB history.

    Returns "rising", "falling", or "stable".
    Used for T1a/T1b unbinding (GPS spoofing interpretation).
    """
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(days=window_days)
    o01_records = [
        o for o in get_observations(path, since=cutoff)
        if o.id == "O01"
    ]
    if len(o01_records) < 2:
        return "stable"

    o01_records.sort(key=lambda o: o.timestamp)
    mid = len(o01_records) // 2
    older_avg = sum(o.value for o in o01_records[:mid]) / mid
    recent_avg = sum(o.value for o in o01_records[mid:]) / (len(o01_records) - mid)

    diff = recent_avg - older_avg
    if diff > 0.05:
        return "rising"
    elif diff < -0.05:
        return "falling"
    return "stable"


def get_history_days(path: Path) -> int:
    """Count distinct days with observations in DB."""
    conn = sqlite3.connect(path)
    row = conn.execute(
        "SELECT COUNT(DISTINCT date(timestamp)) FROM observations"
    ).fetchone()
    conn.close()
    return row[0] if row else 0


def compute_confidence_level(path: Path) -> str:
    """Determine system confidence based on DB history depth.

    <3 days: burn_in (output unreliable, don't act on it)
    3-7 days: low (directional but noisy)
    >7 days: normal
    """
    days = get_history_days(path)
    if days < 3:
        return "burn_in"
    elif days <= 7:
        return "low"
    return "normal"


# ── Controls ──────────────────────────────────────────────────────────

def insert_control(path: Path, ctrl: Control) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO controls (id, actor, triggered, trigger_time, effect) VALUES (?, ?, ?, ?, ?)",
        (ctrl.id, ctrl.actor, int(ctrl.triggered),
         ctrl.trigger_time.isoformat() if ctrl.trigger_time else None,
         ctrl.effect),
    )
    conn.commit()
    conn.close()


def get_controls(path: Path) -> list[Control]:
    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT id, actor, triggered, trigger_time, effect FROM controls").fetchall()
    conn.close()
    return [
        Control(
            id=r[0], actor=r[1], triggered=bool(r[2]),
            trigger_time=datetime.fromisoformat(r[3]) if r[3] else None,
            effect=r[4],
        )
        for r in rows
    ]


# ── SystemOutput ──────────────────────────────────────────────────────

def _serialize_system_output(so: SystemOutput) -> str:
    """Serialize SystemOutput to JSON string."""
    data = so.model_dump(mode="json")
    return json.dumps(data, ensure_ascii=False)


def _deserialize_system_output(json_str: str) -> SystemOutput:
    """Deserialize JSON string to SystemOutput."""
    data = json.loads(json_str)
    # Reconstruct nested models
    data["ach_posterior"] = ACHPosterior(**data["ach_posterior"])
    data["path_probabilities"] = PathWeights(**data["path_probabilities"])
    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
    # buffer_trajectory: list of lists -> list of tuples
    data["buffer_trajectory"] = [tuple(x) for x in data["buffer_trajectory"]]
    # net_gap_trajectories: dict of list of lists -> dict of list of tuples
    data["net_gap_trajectories"] = {
        k: [tuple(x) for x in v]
        for k, v in data["net_gap_trajectories"].items()
    }
    return SystemOutput(**data)


def save_system_output(path: Path, so: SystemOutput) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO system_outputs (timestamp, data_json) VALUES (?, ?)",
        (so.timestamp.isoformat(), _serialize_system_output(so)),
    )
    conn.commit()
    conn.close()


def get_latest_output(path: Path) -> SystemOutput | None:
    conn = sqlite3.connect(path)
    row = conn.execute(
        "SELECT data_json FROM system_outputs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return _deserialize_system_output(row[0])


# ── State snapshots ───────────────────────────────────────────────────

def save_state_snapshot(path: Path, state_data: dict) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO state_snapshots (timestamp, data_json) VALUES (?, ?)",
        (datetime.now().isoformat(), json.dumps(state_data, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


# ── MC runs ───────────────────────────────────────────────────────────

def save_mc_run(path: Path, result_json: str, n_samples: int = 0, seed: int = 0) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO mc_runs (timestamp, n_samples, seed, result_json) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(), n_samples, seed, result_json),
    )
    conn.commit()
    conn.close()


# ── Parameter overrides ───────────────────────────────────────────────

def save_parameter_override(path: Path, param: str, old_value: str, new_value: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO parameters_override (param, old_value, new_value) VALUES (?, ?, ?)",
        (param, old_value, new_value),
    )
    conn.commit()
    conn.close()


def get_parameter_overrides(path: Path) -> list[dict]:
    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT param, old_value, new_value, created_at FROM parameters_override").fetchall()
    conn.close()
    return [{"param": r[0], "old_value": r[1], "new_value": r[2], "created_at": r[3]} for r in rows]


# ── Position signals ──────────────────────────────────────────────────

def save_position_signal(path: Path, signal_type: str, action: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO position_signals (timestamp, signal_type, action) VALUES (?, ?, ?)",
        (datetime.now().isoformat(), signal_type, action),
    )
    conn.commit()
    conn.close()


def get_pending_signals(path: Path) -> list[dict]:
    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT signal_type, action, created_at FROM position_signals WHERE executed = 0"
    ).fetchall()
    conn.close()
    return [{"signal_type": r[0], "action": r[1], "created_at": r[2]} for r in rows]


# ── Articles & provenance ────────────────────────────────────────────

def insert_articles(path: Path, articles: list[dict]) -> None:
    """Store articles, skip duplicates by ID (INSERT OR IGNORE)."""
    conn = sqlite3.connect(path)
    conn.executemany(
        "INSERT OR IGNORE INTO articles (id, title, title_zh, source, url, summary, published_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (a["id"], a.get("title", ""), a.get("title_zh", ""),
             a.get("source", ""), a.get("url", ""),
             a.get("summary", "")[:3000], a.get("published_date"))
            for a in articles if a.get("id")
        ],
    )
    conn.commit()
    conn.close()


def get_article_ids(path: Path, candidate_ids: set[str]) -> set[str]:
    """Return subset of candidate_ids that already exist in DB."""
    if not candidate_ids:
        return set()
    conn = sqlite3.connect(path)
    placeholders = ",".join("?" for _ in candidate_ids)
    rows = conn.execute(
        f"SELECT id FROM articles WHERE id IN ({placeholders})",
        list(candidate_ids),
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}


def insert_article_observations(path: Path, mappings: list[dict], batch_run: str = "") -> None:
    """Store article→observation provenance links."""
    conn = sqlite3.connect(path)
    conn.executemany(
        "INSERT INTO article_observations (article_id, obs_id, confidence, batch_run) VALUES (?, ?, ?, ?)",
        [(m["article_id"], m["obs_id"], m.get("confidence", ""), batch_run) for m in mappings],
    )
    conn.commit()
    conn.close()


def get_article_observations(
    path: Path, obs_id: str | None = None, batch_run: str | None = None,
) -> list[dict]:
    """Query article→observation provenance, optionally filtered."""
    conn = sqlite3.connect(path)
    query = "SELECT article_id, obs_id, confidence, batch_run, created_at FROM article_observations WHERE 1=1"
    params: list = []
    if obs_id:
        query += " AND obs_id = ?"
        params.append(obs_id)
    if batch_run:
        query += " AND batch_run = ?"
        params.append(batch_run)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [
        {"article_id": r[0], "obs_id": r[1], "confidence": r[2], "batch_run": r[3], "created_at": r[4]}
        for r in rows
    ]


# ── Calibration predictions ──────────────────────────────────────────

def record_predictions(path: Path, so: "SystemOutput", conflict_day: int) -> None:
    """Record key predictions from a pipeline run for future calibration.

    Tracks: ACH posterior, path probabilities, T expected.
    Resolution criteria defined per metric for later Brier scoring.
    """
    from datetime import timedelta
    run_ts = so.timestamp.isoformat()
    resolve_date = (so.timestamp + timedelta(days=30)).strftime("%Y-%m-%d")

    predictions = [
        ("ach_h2", so.ach_posterior.h2, resolve_date,
         "H2 correct if conflict still active at T expected date"),
        ("path_a", so.path_probabilities.a, resolve_date,
         "Path A correct if conflict resolves within 35 days of start"),
        ("path_b", so.path_probabilities.b, resolve_date,
         "Path B correct if conflict resolves between 35-120 days"),
        ("path_c", so.path_probabilities.c, resolve_date,
         "Path C correct if conflict exceeds 120 days"),
        ("t_expected", so.t_weighted_mean, None,
         "Compare to actual conflict duration when resolved"),
    ]

    conn = sqlite3.connect(path)
    conn.executemany(
        """INSERT INTO predictions (run_ts, metric, value, resolve_by, resolution_criteria)
           VALUES (?, ?, ?, ?, ?)""",
        [(run_ts, m, v, rb, rc) for m, v, rb, rc in predictions],
    )
    conn.commit()
    conn.close()
