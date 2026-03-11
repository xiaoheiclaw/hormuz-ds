"""SQLite database layer for the Hormuz decision support system."""
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from hormuz.models import (
    ACHEvidence,
    MCParams,
    MCResult,
    Observation,
    PathWeights,
    PositionSignal,
    Regime,
    RegimeType,
    Signal,
    SignalStatus,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value REAL NOT NULL,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS ach_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    question TEXT NOT NULL,
    evidence_id INTEGER NOT NULL,
    direction TEXT NOT NULL,
    confidence TEXT NOT NULL,
    notes TEXT,
    source_observation_id INTEGER REFERENCES observations(id)
);

CREATE TABLE IF NOT EXISTS regimes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    question TEXT NOT NULL,
    regime TEXT NOT NULL,
    trigger TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    status TEXT NOT NULL,
    revert_deadline TEXT,
    action_taken TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mc_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    params TEXT NOT NULL,
    path_weights TEXT NOT NULL,
    trigger TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mc_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    params_id INTEGER REFERENCES mc_params(id),
    output TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS position_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    trigger TEXT NOT NULL,
    action TEXT NOT NULL,
    executed INTEGER NOT NULL DEFAULT 0
);
"""


def _ts(dt: datetime) -> str:
    """Datetime to ISO8601 UTC string. Naive datetimes assumed UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _dt(s: str) -> datetime:
    """ISO8601 string to datetime."""
    return datetime.fromisoformat(s)


class HormuzDB:
    """SQLite-backed storage for all Hormuz pipeline data."""

    def __init__(self, path: Path) -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)

    # --- Utility ---

    def list_tables(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return [r["name"] for r in rows]

    # --- Observations ---

    def insert_observation(self, obs: Observation) -> int:
        cur = self._conn.execute(
            "INSERT INTO observations (timestamp, source, category, key, value, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                _ts(obs.timestamp),
                obs.source,
                obs.category,
                obs.key,
                obs.value,
                json.dumps(obs.metadata) if obs.metadata is not None else None,
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_observations_since(
        self, since: datetime, category: str | None = None
    ) -> list[Observation]:
        sql = "SELECT * FROM observations WHERE timestamp >= ?"
        params: list = [_ts(since)]
        if category is not None:
            sql += " AND category = ?"
            params.append(category)
        sql += " ORDER BY timestamp"
        rows = self._conn.execute(sql, params).fetchall()
        return [
            Observation(
                id=r["id"],
                timestamp=_dt(r["timestamp"]),
                source=r["source"],
                category=r["category"],
                key=r["key"],
                value=r["value"],
                metadata=json.loads(r["metadata"]) if r["metadata"] else None,
            )
            for r in rows
        ]

    # --- ACH Evidence ---

    def insert_ach_evidence(self, ev: ACHEvidence) -> int:
        cur = self._conn.execute(
            "INSERT INTO ach_evidence "
            "(timestamp, question, evidence_id, direction, confidence, notes, source_observation_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                _ts(ev.timestamp),
                ev.question,
                ev.evidence_id,
                ev.direction,
                ev.confidence,
                ev.notes,
                ev.source_observation_id,
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_ach_evidence(
        self,
        question: str,
        confidence: str | None = None,
        since: datetime | None = None,
    ) -> list[ACHEvidence]:
        sql = "SELECT * FROM ach_evidence WHERE question = ?"
        params: list = [question]
        if confidence is not None:
            sql += " AND confidence = ?"
            params.append(confidence)
        if since is not None:
            sql += " AND timestamp >= ?"
            params.append(_ts(since))
        sql += " ORDER BY timestamp"
        rows = self._conn.execute(sql, params).fetchall()
        return [
            ACHEvidence(
                id=r["id"],
                timestamp=_dt(r["timestamp"]),
                question=r["question"],
                evidence_id=r["evidence_id"],
                direction=r["direction"],
                confidence=r["confidence"],
                notes=r["notes"],
                source_observation_id=r["source_observation_id"],
            )
            for r in rows
        ]

    # --- Regimes ---

    def insert_regime(self, regime: Regime) -> int:
        cur = self._conn.execute(
            "INSERT INTO regimes (timestamp, question, regime, trigger) VALUES (?, ?, ?, ?)",
            (_ts(regime.timestamp), regime.question, regime.regime, regime.trigger),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_latest_regime(self, question: str) -> Regime | None:
        row = self._conn.execute(
            "SELECT * FROM regimes WHERE question = ? ORDER BY timestamp DESC LIMIT 1",
            (question,),
        ).fetchone()
        if row is None:
            return None
        return Regime(
            id=row["id"],
            timestamp=_dt(row["timestamp"]),
            question=row["question"],
            regime=RegimeType(row["regime"]),
            trigger=row["trigger"],
        )

    # --- Signals ---

    def insert_signal(self, signal: Signal) -> int:
        cur = self._conn.execute(
            "INSERT INTO signals (timestamp, signal_id, status, revert_deadline, action_taken) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                _ts(signal.timestamp),
                signal.signal_id,
                signal.status,
                _ts(signal.revert_deadline) if signal.revert_deadline else None,
                signal.action_taken,
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_active_signals(self) -> list[Signal]:
        rows = self._conn.execute(
            "SELECT * FROM signals WHERE status IN (?, ?) ORDER BY timestamp",
            (SignalStatus.triggered, SignalStatus.confirmed),
        ).fetchall()
        return [
            Signal(
                id=r["id"],
                timestamp=_dt(r["timestamp"]),
                signal_id=r["signal_id"],
                status=SignalStatus(r["status"]),
                revert_deadline=_dt(r["revert_deadline"]) if r["revert_deadline"] else None,
                action_taken=r["action_taken"],
            )
            for r in rows
        ]

    def update_signal_status(self, signal_id: int, new_status: SignalStatus) -> None:
        self._conn.execute(
            "UPDATE signals SET status = ? WHERE id = ?",
            (new_status, signal_id),
        )
        self._conn.commit()

    # --- MC Params ---

    def insert_mc_params(self, params: MCParams) -> int:
        params_json = json.dumps({
            "irgc_decay_mean": params.irgc_decay_mean,
            "convoy_start_mean": params.convoy_start_mean,
            "disruption_range": list(params.disruption_range),
            "pipeline_max": params.pipeline_max,
            "pipeline_ramp_weeks": params.pipeline_ramp_weeks,
            "spr_rate_mean": params.spr_rate_mean,
            "spr_delay_weeks": params.spr_delay_weeks,
            "surplus_buffer": params.surplus_buffer,
        })
        weights_json = json.dumps(params.path_weights.model_dump())
        cur = self._conn.execute(
            "INSERT INTO mc_params (timestamp, params, path_weights, trigger) VALUES (?, ?, ?, ?)",
            (_ts(params.timestamp), params_json, weights_json, params.trigger),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_latest_mc_params(self) -> MCParams | None:
        row = self._conn.execute(
            "SELECT * FROM mc_params ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        p = json.loads(row["params"])
        w = json.loads(row["path_weights"])
        return MCParams(
            id=row["id"],
            timestamp=_dt(row["timestamp"]),
            irgc_decay_mean=p["irgc_decay_mean"],
            convoy_start_mean=p["convoy_start_mean"],
            disruption_range=tuple(p["disruption_range"]),
            pipeline_max=p["pipeline_max"],
            pipeline_ramp_weeks=p["pipeline_ramp_weeks"],
            spr_rate_mean=p["spr_rate_mean"],
            spr_delay_weeks=p["spr_delay_weeks"],
            surplus_buffer=p["surplus_buffer"],
            path_weights=PathWeights(**w),
            trigger=row["trigger"],
        )

    # --- MC Results ---

    def insert_mc_result(self, result: MCResult) -> int:
        output_json = json.dumps({
            "price_mean": result.price_mean,
            "price_p10": result.price_p10,
            "price_p50": result.price_p50,
            "price_p90": result.price_p90,
            "path_a_price": result.path_a_price,
            "path_b_price": result.path_b_price,
            "path_c_price": result.path_c_price,
            "key_dates": result.key_dates,
        }, default=str)
        cur = self._conn.execute(
            "INSERT INTO mc_results (timestamp, params_id, output) VALUES (?, ?, ?)",
            (_ts(result.timestamp), result.params_id, output_json),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    # --- Position Signals ---

    def insert_position_signal(self, signal: PositionSignal) -> int:
        cur = self._conn.execute(
            "INSERT INTO position_signals (timestamp, trigger, action, executed) "
            "VALUES (?, ?, ?, ?)",
            (_ts(signal.timestamp), signal.trigger, signal.action, int(signal.executed)),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_unexecuted_position_signals(self) -> list[PositionSignal]:
        rows = self._conn.execute(
            "SELECT * FROM position_signals WHERE executed = 0 ORDER BY timestamp"
        ).fetchall()
        return [
            PositionSignal(
                id=r["id"],
                timestamp=_dt(r["timestamp"]),
                trigger=r["trigger"],
                action=r["action"],
                executed=bool(r["executed"]),
            )
            for r in rows
        ]

    def mark_position_executed(self, signal_id: int) -> None:
        self._conn.execute(
            "UPDATE position_signals SET executed = 1 WHERE id = ?",
            (signal_id,),
        )
        self._conn.commit()
