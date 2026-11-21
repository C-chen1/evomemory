"""SQLite persistence for EvoMemory.

One file, no ORM, no migrations framework — but the schema is versioned and the
migration list is append-only, because the whole point of this project is that
memory state must be auditable months later.

Every table that records history is append-only. The single mutable column in
the database is ``experiences.status`` and ``experiences.updated_at``; every
change to them also writes a row into ``experience_events``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import now_iso

MIGRATIONS: List[str] = [
    # v1 — initial schema
    """
    CREATE TABLE tasks (
        task_id      TEXT PRIMARY KEY,
        goal         TEXT NOT NULL,
        context_json TEXT NOT NULL,
        tags_json    TEXT NOT NULL,
        family       TEXT NOT NULL DEFAULT '',
        created_at   TEXT NOT NULL
    );

    CREATE TABLE runs (
        run_id       TEXT PRIMARY KEY,
        mode         TEXT NOT NULL,
        config_json  TEXT NOT NULL,
        started_at   TEXT NOT NULL,
        ended_at     TEXT
    );

    CREATE TABLE traces (
        trace_id    TEXT PRIMARY KEY,
        run_id      TEXT NOT NULL,
        task_id     TEXT NOT NULL,
        model       TEXT NOT NULL DEFAULT '',
        params_json TEXT NOT NULL DEFAULT '{}',
        status      TEXT NOT NULL DEFAULT 'running',
        started_at  TEXT NOT NULL,
        ended_at    TEXT
    );
    CREATE INDEX idx_traces_task ON traces(task_id);
    CREATE INDEX idx_traces_run  ON traces(run_id);

    CREATE TABLE trace_steps (
        trace_id     TEXT NOT NULL,
        idx          INTEGER NOT NULL,
        thought      TEXT NOT NULL DEFAULT '',
        tool         TEXT NOT NULL DEFAULT '',
        args_json    TEXT NOT NULL DEFAULT '{}',
        receipt_json TEXT,
        PRIMARY KEY (trace_id, idx)
    );

    CREATE TABLE outcomes (
        outcome_id  TEXT PRIMARY KEY,
        trace_id    TEXT NOT NULL,
        status      TEXT NOT NULL,
        score       REAL NOT NULL DEFAULT 0.0,
        reason      TEXT NOT NULL DEFAULT '',
        verified_by TEXT NOT NULL,
        checks_json TEXT NOT NULL DEFAULT '[]',
        created_at  TEXT NOT NULL
    );
    CREATE INDEX idx_outcomes_trace ON outcomes(trace_id);

    CREATE TABLE evidence (
        evidence_id     TEXT PRIMARY KEY,
        kind            TEXT NOT NULL,
        sha256          TEXT NOT NULL,
        path            TEXT NOT NULL,
        mime            TEXT NOT NULL DEFAULT '',
        bytes_len       INTEGER NOT NULL DEFAULT 0,
        metadata_json   TEXT NOT NULL DEFAULT '{}',
        derived_from    TEXT,
        created_at      TEXT NOT NULL
    );
    CREATE UNIQUE INDEX idx_evidence_sha ON evidence(sha256, kind);
    CREATE INDEX idx_evidence_derived ON evidence(derived_from);

    CREATE TABLE experiences (
        experience_id   TEXT NOT NULL,
        version         INTEGER NOT NULL,
        task_id         TEXT NOT NULL DEFAULT '',
        trace_id        TEXT NOT NULL DEFAULT '',
        task_context    TEXT NOT NULL DEFAULT '',
        evidence_refs_json  TEXT NOT NULL DEFAULT '[]',
        action_trace_refs_json TEXT NOT NULL DEFAULT '[]',
        outcome         TEXT NOT NULL DEFAULT '',
        failure_signature TEXT NOT NULL DEFAULT '',
        lesson          TEXT NOT NULL,
        applicability_json TEXT NOT NULL DEFAULT '[]',
        status          TEXT NOT NULL,
        verification_status TEXT NOT NULL,
        family          TEXT NOT NULL DEFAULT '',
        source_kind     TEXT NOT NULL DEFAULT 'heuristic',
        prompt_version  TEXT NOT NULL DEFAULT '',
        supersedes      INTEGER,
        confidence      REAL NOT NULL DEFAULT 0.0,
        created_at      TEXT NOT NULL,
        updated_at      TEXT NOT NULL,
        PRIMARY KEY (experience_id, version)
    );
    CREATE INDEX idx_exp_status ON experiences(status);
    CREATE INDEX idx_exp_family ON experiences(family);
    CREATE INDEX idx_exp_task   ON experiences(task_id);

    CREATE TABLE experience_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        experience_id TEXT NOT NULL,
        version     INTEGER NOT NULL,
        from_status TEXT,
        to_status   TEXT NOT NULL,
        actor       TEXT NOT NULL,
        reason      TEXT NOT NULL DEFAULT '',
        evidence_refs_json TEXT NOT NULL DEFAULT '[]',
        created_at  TEXT NOT NULL
    );
    CREATE INDEX idx_evt_exp ON experience_events(experience_id, version);

    CREATE TABLE memory_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      TEXT NOT NULL DEFAULT '',
        task_id     TEXT NOT NULL DEFAULT '',
        trace_id    TEXT NOT NULL DEFAULT '',
        step        INTEGER NOT NULL DEFAULT 0,
        query       TEXT NOT NULL DEFAULT '',
        retrieved_json TEXT NOT NULL DEFAULT '[]',
        injected_json  TEXT NOT NULL DEFAULT '[]',
        rejected_json  TEXT NOT NULL DEFAULT '[]',
        created_at  TEXT NOT NULL
    );
    CREATE INDEX idx_mevents_task ON memory_events(task_id);

    CREATE TABLE llm_calls (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      TEXT NOT NULL DEFAULT '',
        task_id     TEXT NOT NULL DEFAULT '',
        purpose     TEXT NOT NULL,
        prompt_version TEXT NOT NULL DEFAULT '',
        model       TEXT NOT NULL,
        input_tokens  INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        latency_ms  INTEGER NOT NULL DEFAULT 0,
        ok          INTEGER NOT NULL DEFAULT 1,
        error       TEXT NOT NULL DEFAULT '',
        created_at  TEXT NOT NULL
    );
    CREATE INDEX idx_llm_run ON llm_calls(run_id);
    """,
    # v2 — structured, machine-executable experience conditions (P1)
    """
    ALTER TABLE experiences ADD COLUMN condition_json TEXT;
    ALTER TABLE experiences ADD COLUMN recommended_action TEXT NOT NULL DEFAULT 'skip';
    """,
    # v3 — usage tracking so a memory that keeps causing failures can be retired
    """
    CREATE TABLE experience_usage (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        experience_id TEXT NOT NULL,
        version       INTEGER NOT NULL,
        task_id       TEXT NOT NULL,
        outcome       TEXT NOT NULL,
        created_at    TEXT NOT NULL
    );
    CREATE INDEX idx_usage_exp ON experience_usage(experience_id, version);
    CREATE INDEX idx_usage_task ON experience_usage(task_id);

    CREATE TABLE experience_strikes (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        experience_id TEXT NOT NULL,
        task_id       TEXT NOT NULL,
        rid           TEXT NOT NULL,
        reason        TEXT NOT NULL DEFAULT '',
        created_at    TEXT NOT NULL
    );
    CREATE INDEX idx_strike_exp ON experience_strikes(experience_id);
    """,
    # v5 — a task is only fully settled once its memory write-back finished.
    # The outcome row is written before extraction, so a crash in between used to
    # leave a task that looked complete and would never be learned from.
    """
    CREATE TABLE task_memory_settlement (
        run_id     TEXT NOT NULL,
        task_id    TEXT NOT NULL,
        settled_at TEXT NOT NULL,
        PRIMARY KEY (run_id, task_id)
    );
    """,
]


class Database:
    """Thin wrapper around a SQLite connection with versioned schema."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.parent and str(self.path.parent) not in ("", "."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._migrate()

    # -- schema ---------------------------------------------------------
    def _migrate(self) -> None:
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
        )
        row = self.conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            self.conn.execute("INSERT INTO schema_version(version) VALUES (0)")
            current = 0
        else:
            current = int(row["version"])
        for i in range(current, len(MIGRATIONS)):
            self.conn.executescript(MIGRATIONS[i])
            self.conn.execute("UPDATE schema_version SET version = ?", (i + 1,))

    # -- helpers --------------------------------------------------------
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, tuple(params))

    def query(self, sql: str, params: Iterable[Any] = ()) -> List[sqlite3.Row]:
        return list(self.conn.execute(sql, tuple(params)).fetchall())

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        return self.conn.execute(sql, tuple(params)).fetchone()

    def insert(self, table: str, values: Dict[str, Any]) -> None:
        cols = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        self.conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(values.values()))

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def loads(value: Optional[str], default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def record_run(db: Database, run_id: str, mode: str, config: Dict[str, Any]) -> None:
    db.insert(
        "runs",
        {
            "run_id": run_id,
            "mode": mode,
            "config_json": dumps(config),
            "started_at": now_iso(),
            "ended_at": None,
        },
    )


def finish_run(db: Database, run_id: str) -> None:
    db.execute("UPDATE runs SET ended_at = ? WHERE run_id = ?", (now_iso(), run_id))
