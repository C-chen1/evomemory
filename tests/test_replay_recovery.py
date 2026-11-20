"""Replay determinism and crash recovery."""

import pytest

from evomemory.bench import build_dataset
from evomemory.db import Database
from evomemory.evidence import EvidenceStore
from evomemory.harness import (
    Harness, MemoryGate, Recovery, ReplayRunner, RunConfig,
)
from evomemory.lifecycle import (
    ExperienceExtractor, ExperienceReconstructor, MemoryCritic,
)
from evomemory.llm import BudgetedLLM, ScriptedLLM
from evomemory.memory import ExperienceStore
from evomemory.models import MemoryMode
from evomemory.policy import ScriptedOpsPolicy


def make_harness(tmp_path, mode=MemoryMode.D_FULL_LIFECYCLE):
    db = Database(tmp_path / "memory.db")
    evidence = EvidenceStore(tmp_path / "evidence", db)
    llm = BudgetedLLM(ScriptedLLM(ScriptedOpsPolicy()), 100, 10_000_000, db)
    gate = MemoryGate(ExperienceStore(db), evidence)
    harness = Harness(db, evidence, llm, gate, ExperienceExtractor(evidence),
                      MemoryCritic(), ExperienceReconstructor(),
                      RunConfig(mode=mode))
    return harness, db, evidence


def test_replay_reproduces_the_recorded_outcome(tmp_path):
    tasks = build_dataset(seed=7, per_family=1)
    harness, db, _ = make_harness(tmp_path)
    result = harness.run_task(tasks[0], "run1")
    report = ReplayRunner(db).replay(result.trace_id, tasks[0])
    assert report.steps_replayed > 0
    assert report.replayed_status == report.recorded_status
    assert report.matches, report.diffs


def test_replay_reports_a_failure_when_the_trace_is_missing(tmp_path):
    _, db, _ = make_harness(tmp_path)
    tasks = build_dataset(seed=7, per_family=1)
    with pytest.raises(KeyError):
        ReplayRunner(db).replay("trace_missing", tasks[0])


def test_replay_is_deterministic_across_runs(tmp_path):
    tasks = build_dataset(seed=7, per_family=1)
    harness, db, _ = make_harness(tmp_path)
    result = harness.run_task(tasks[0], "run1")
    runner = ReplayRunner(db)
    a = runner.replay(result.trace_id, tasks[0])
    b = runner.replay(result.trace_id, tasks[0])
    assert a.to_dict() == b.to_dict()


def test_replay_declares_its_limitation():
    assert "not re-sampled" in ReplayRunner.LIMITATION


def test_recovery_skips_settled_tasks(tmp_path):
    tasks = build_dataset(seed=7, per_family=2)
    harness, db, _ = make_harness(tmp_path)
    harness.run_task(tasks[0], "run-r")
    recovery = Recovery(db)
    assert recovery.completed_task_ids("run-r") == [tasks[0].task.task_id]
    pending = recovery.pending_tasks("run-r", tasks)
    assert tasks[0] not in pending and len(pending) == len(tasks) - 1


def test_recovery_resume_runs_only_the_remainder(tmp_path):
    tasks = build_dataset(seed=7, per_family=2)
    harness, db, _ = make_harness(tmp_path)
    harness.run_task(tasks[0], "run-r2")
    outcome = Recovery(db).resume(harness, "run-r2", tasks)
    assert outcome["skipped"] == 1
    assert outcome["ran"] == len(tasks) - 1
    # a second resume is a no-op: nothing is redone
    again = Recovery(db).resume(harness, "run-r2", tasks)
    assert again["ran"] == 0
    assert again["skipped"] == len(tasks)


def test_resume_does_not_duplicate_memory(tmp_path):
    tasks = build_dataset(seed=7, per_family=2)
    harness, db, _ = make_harness(tmp_path)
    harness.run_task(tasks[0], "run-r3")
    before = db.query_one("SELECT COUNT(*) AS n FROM experiences")["n"]
    Recovery(db).resume(harness, "run-r3", tasks)
    after = db.query_one("SELECT COUNT(*) AS n FROM experiences")["n"]
    # the extra tasks may add *new* lessons, but the settled task is not re-extracted
    assert after >= before
    assert db.query_one(
        "SELECT COUNT(*) AS n FROM traces WHERE run_id = ? AND task_id = ?",
        ("run-r3", tasks[0].task.task_id))["n"] == 1
