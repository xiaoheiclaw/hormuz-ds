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
