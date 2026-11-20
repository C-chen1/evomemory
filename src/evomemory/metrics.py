"""Metrics for the A/B/C/D comparison.

Every metric here is computed from recorded rows, not from the agent's opinion.
Two of them are easy to fake and therefore get an explicit definition in the
report:

* ``retrieval_hit_rate`` — the fraction of *eligible* tasks (a task in a family
  that already had at least one active experience) where at least one injected
  experience came from the same family. It is not Recall@K; it does not claim
  the retrieved item was the right one, only that memory of the right family was
  reached.
* ``negative_transfer_rate`` — the fraction of tasks where a cross-family
  experience was injected **and** the run violated the family's hidden rule.
  Both conditions are required: injecting an unrelated memory that happens to
  do no harm is not negative transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .db import Database, loads
from .harness import TaskRunResult


@dataclass
class ModeMetrics:
    mode: str
    n_tasks: int = 0
    n_success: int = 0
    success_rate: float = 0.0
    n_eligible: int = 0
    n_retrieval_hit: int = 0
    retrieval_hit_rate: float = 0.0
    n_negative_transfer: int = 0
    negative_transfer_rate: float = 0.0
    mean_injected: float = 0.0
    mean_tool_calls: float = 0.0
    total_tokens: int = 0
    mean_tokens_per_task: float = 0.0
    mean_steps: float = 0.0
    experiences_written: int = 0
    experiences_admitted: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode, "n_tasks": self.n_tasks, "n_success": self.n_success,
            "success_rate": round(self.success_rate, 4),
            "n_eligible": self.n_eligible, "n_retrieval_hit": self.n_retrieval_hit,
            "retrieval_hit_rate": round(self.retrieval_hit_rate, 4),
            "n_negative_transfer": self.n_negative_transfer,
            "negative_transfer_rate": round(self.negative_transfer_rate, 4),
            "mean_injected": round(self.mean_injected, 3),
            "mean_tool_calls": round(self.mean_tool_calls, 3),
            "total_tokens": self.total_tokens,
            "mean_tokens_per_task": round(self.mean_tokens_per_task, 1),
            "mean_steps": round(self.mean_steps, 3),
            "experiences_written": self.experiences_written,
            "experiences_admitted": self.experiences_admitted,
            **self.extra,
        }


def _family_of(db: Database, experience_id: str) -> str:
    row = db.query_one(
        "SELECT family FROM experiences WHERE experience_id = ? "
        "ORDER BY version DESC LIMIT 1", (experience_id,))
    return row["family"] if row else ""


def _check_failed(result: TaskRunResult, name: str) -> bool:
    for check in result.outcome.checks:
        if check.get("name") == name:
            return not check.get("passed", False)
    return False


def compute_metrics(db: Database, results: Sequence[TaskRunResult],
                    mode: str, task_families: Dict[str, str],
                    already_known: Optional[Dict[str, int]] = None) -> ModeMetrics:
    """``already_known`` maps task_id -> number of active experiences in its
    family *before* the task ran. Tasks with a non-zero count are eligible for a
    retrieval hit."""
    m = ModeMetrics(mode=mode, n_tasks=len(results))
    if not results:
        return m

    known = already_known or {}
    injected_total = 0
    for r in results:
        family = task_families.get(r.task_id, "")
        if r.outcome.status.value == "success":
            m.n_success += 1

        injected_families = [_family_of(db, exp_id) for exp_id in r.injected_ids]
        injected_total += len(r.injected_ids)

        if known.get(r.task_id, 0) > 0:
            m.n_eligible += 1
            if any(f == family for f in injected_families if f):
                m.n_retrieval_hit += 1

        cross = [f for f in injected_families if f and f != family]
        if cross and _check_failed(r, "hidden_rule_respected"):
            m.n_negative_transfer += 1

        m.total_tokens += r.tokens_in + r.tokens_out
        m.mean_tool_calls += r.tool_calls
        m.mean_steps += r.steps
        m.experiences_written += r.memory_written
        m.experiences_admitted += r.memory_admitted

    n = len(results)
    m.success_rate = m.n_success / n
    m.retrieval_hit_rate = (m.n_retrieval_hit / m.n_eligible) if m.n_eligible else 0.0
    m.negative_transfer_rate = m.n_negative_transfer / n
    m.mean_injected = injected_total / n
    m.mean_tool_calls /= n
    m.mean_steps /= n
    m.mean_tokens_per_task = m.total_tokens / n
    return m


def evidence_grounding_accuracy(db: Database, evidence_exists) -> Dict[str, float]:
    """Share of stored experiences whose evidence references all resolve.

    This is the honesty metric for the memory store: an experience that points at
    evidence which no longer exists is unreviewable, and should count against the
    system even if its task succeeded.
    """
    rows = db.query("SELECT experience_id, version, evidence_refs_json FROM experiences")
    total = len(rows)
    if total == 0:
        return {"experiences": 0, "fully_grounded": 0, "grounding_accuracy": 0.0}
    good = 0
    for row in rows:
        refs = loads(row["evidence_refs_json"], [])
        if refs and all(evidence_exists(r) for r in refs):
            good += 1
    return {
        "experiences": total,
        "fully_grounded": good,
        "grounding_accuracy": round(good / total, 4),
    }


def compare_modes(metrics: Sequence[ModeMetrics]) -> Dict[str, Any]:
    """Pairwise deltas against mode A, which is the no-memory baseline."""
    by_mode = {m.mode: m for m in metrics}
    base = by_mode.get("A")
    if base is None:
        return {}
    out: Dict[str, Any] = {}
    for m in metrics:
        if m.mode == "A":
            continue
        out[m.mode] = {
            "delta_success_rate": round(m.success_rate - base.success_rate, 4),
            "delta_negative_transfer": round(
                m.negative_transfer_rate - base.negative_transfer_rate, 4),
            "delta_tokens_per_task": round(
                m.mean_tokens_per_task - base.mean_tokens_per_task, 1),
            "delta_tool_calls": round(m.mean_tool_calls - base.mean_tool_calls, 3),
        }
    return out


def mde_binary(n_cells: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable effect (percentage points) for a paired binary design.

    Reported *before* running anything, so that a null result at n=8 is not read
    as evidence of no effect. Uses the normal approximation to the McNemar test:
    the design can only detect a discordant-pair rate large enough to clear the
    critical value.
    """
    import math
    if n_cells <= 0:
        return float("inf")
    z_alpha = 1.959963985
    z_beta = 0.8416212336
    # detectable difference in discordant proportions, then expressed per cell
    return round(100.0 * (z_alpha + z_beta) * math.sqrt(2.0 / n_cells) / 2.0, 2)
