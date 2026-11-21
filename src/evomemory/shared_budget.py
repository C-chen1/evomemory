"""Persistent, shared request reservations for a single experiment round.

Unreported usage is not zero: it retains its reservation and blocks further
requests until reconciled. Input token reservations are estimates, especially
for images, and cannot guarantee a provider-side hard token cap.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .llm import BudgetExceeded


class SharedBudget:
    def __init__(self, path: str | Path, max_calls: int, max_tokens: int):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("CREATE TABLE IF NOT EXISTS budget_config (calls INTEGER, tokens INTEGER)")
        self.db.execute("CREATE TABLE IF NOT EXISTS budget_calls (id INTEGER PRIMARY KEY, "
                        "run_id TEXT, task_id TEXT, purpose TEXT, reserved INTEGER, "
                        "actual INTEGER, state TEXT NOT NULL)")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT calls, tokens FROM budget_config").fetchone()
            if row is None:
                self.db.execute("INSERT INTO budget_config VALUES (?, ?)", (max_calls, max_tokens))
            elif (row["calls"], row["tokens"]) != (max_calls, max_tokens):
                raise ValueError("cannot change an existing round's budget")
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        self.max_calls, self.max_tokens = max_calls, max_tokens

    def reserve(self, *, run_id: str, task_id: str, purpose: str, tokens: int) -> int:
        if tokens <= 0:
            raise ValueError("positive token reservation required")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT COUNT(*) AS n, COALESCE(SUM(COALESCE(actual,reserved)),0) AS spent, "
                                  "SUM(CASE WHEN state IN ('pending','unknown','overrun') THEN 1 ELSE 0 END) AS blocked "
                                  "FROM budget_calls").fetchone()
            if row["blocked"]:
                raise BudgetExceeded("unresolved or overrun request in shared budget")
            if row["n"] >= self.max_calls or row["spent"] + tokens > self.max_tokens:
                raise BudgetExceeded("shared call or reserved-token budget exhausted")
            cur = self.db.execute("INSERT INTO budget_calls(run_id,task_id,purpose,reserved,state) "
                                  "VALUES (?,?,?,?, 'pending')", (run_id, task_id, purpose, tokens))
            self.db.execute("COMMIT")
            return int(cur.lastrowid)
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def settle(self, request_id: int, actual: int | None) -> None:
        if actual is not None and actual < 0:
            raise ValueError("usage must be nonnegative")
        row = self.db.execute("SELECT reserved,state FROM budget_calls WHERE id=?", (request_id,)).fetchone()
        if row is None or row["state"] != "pending":
            raise ValueError("request is not pending")
        state = "unknown" if actual is None else ("overrun" if actual > row["reserved"] else "settled")
        self.db.execute("UPDATE budget_calls SET actual=?, state=? WHERE id=?", (actual, state, request_id))

    def summary(self) -> dict:
        rows = self.db.execute("SELECT reserved,actual,state FROM budget_calls").fetchall()
        return {"calls": len(rows), "accounted_tokens": sum(r["actual"] if r["actual"] is not None
                                                            else r["reserved"] for r in rows),
                "unknown": sum(r["state"] in ("pending", "unknown") for r in rows),
                "overruns": sum(r["state"] == "overrun" for r in rows),
                "max_calls": self.max_calls, "max_tokens": self.max_tokens}

    def close(self) -> None:
        self.db.close()
