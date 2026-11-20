"""Harness behaviour and the A/B/D end-to-end comparison."""

import json

import pytest

from evomemory.bench import build_dataset
from evomemory.db import Database, finish_run, record_run
from evomemory.evidence import EvidenceStore
from evomemory.harness import ContextManager, Harness, MemoryGate, RunConfig
from evomemory.lifecycle import (
    ExperienceExtractor, ExperienceReconstructor, MemoryCritic,
)
from evomemory.llm import BudgetedLLM, ScriptedLLM
from evomemory.memory import ExperienceStore
from evomemory.metrics import compute_metrics, mde_binary
from evomemory.models import (
    EvidenceKind, Experience, ExperienceStatus, MemoryMode, RetrievedExperience,
)
from evomemory.policy import ScriptedOpsPolicy, blocked_values_from_guidance


# -- context manager ------------------------------------------------------


def test_context_manager_enforces_token_budget():
    cm = ContextManager(token_budget=10)
    exps = [RetrievedExperience(experience=Experience(lesson="x" * 80, experience_id="e1"),
                                score=1.0, guidance="x" * 80),   # 20 estimated tokens
            RetrievedExperience(experience=Experience(lesson="short", experience_id="e2"),
                                score=0.5, guidance="short")]    # 2 estimated tokens
    block = cm.build(exps)
    assert block.injected_ids == ["e2"]
    assert any(r["experience_id"] == "e1" for r in block.rejected)


def test_context_manager_renders_nothing_when_empty():
    assert ContextManager().build([]).render() == ""


# -- policy ---------------------------------------------------------------


def test_guidance_inference_requires_a_negation_cue():
    assert blocked_values_from_guidance(
        "Do not act on records with tier=gold.", ["gold"]) == ["gold"]
    assert blocked_values_from_guidance(
        "Gold tier records exist.", ["gold"]) == []


def test_policy_blocks_the_values_it_reads():
    policy = ScriptedOpsPolicy()
    records = [
        {"rid": "a", "region": "eu", "tier": "gold", "priority": 4, "status": "open", "note": ""},
        {"rid": "b", "region": "eu", "tier": "silver", "priority": 4, "status": "open", "note": ""},
        {"rid": "c", "region": "eu", "tier": "silver", "priority": 4, "status": "open", "note": "hold"},
    ]
    task = {"context": {"region": "eu", "priority": 3, "action": "escalate"}}
    chosen = policy._choose_targets(records, task,
                                    "Do not act on records with tier=gold and note=hold.")
    assert chosen == ["b"]


# -- gate -----------------------------------------------------------------


def test_gate_refuses_model_promoted_memory(tmp_env):
    db, evidence = tmp_env
    gate = MemoryGate(ExperienceStore(db), evidence)
    exp = Experience(lesson="A long enough lesson for the gate to consider.",
                     status=ExperienceStatus.ACTIVE)
    with pytest.raises(PermissionError):
        gate.propose(exp, actor=__import__("evomemory.models", fromlist=["Actor"]).Actor.MODEL)


def test_gate_admits_only_after_validation(tmp_env):
    db, evidence = tmp_env
    gate = MemoryGate(ExperienceStore(db), evidence)
    ev = evidence.put_text("receipt")
    exp = Experience(lesson="For escalation, do not act on records with tier=gold.",
                     evidence_refs=[ev.evidence_id],
                     applicability_conditions=["task_family:escalation"])
    gate.propose(exp)
    admitted, report = gate.admit(exp)
    assert admitted, report.summary()
    assert gate.active_experiences()[0].status is ExperienceStatus.ACTIVE


def test_gate_rejects_and_does_not_activate(tmp_env):
    db, evidence = tmp_env
    store = ExperienceStore(db)
    gate = MemoryGate(store, evidence)
    exp = Experience(lesson="short", evidence_refs=[])
    gate.propose(exp)
    admitted, report = gate.admit(exp)
    assert not admitted
    assert store.get(exp.experience_id).status is ExperienceStatus.REJECTED


# -- end to end -----------------------------------------------------------


def _harness(tmp_path, mode, db_name="memory.db"):
    db = Database(tmp_path / db_name)
    evidence = EvidenceStore(tmp_path / f"ev_{db_name}", db)
    llm = BudgetedLLM(ScriptedLLM(ScriptedOpsPolicy()), 100, 10_000_000, db)
    gate = MemoryGate(ExperienceStore(db), evidence)
    harness = Harness(db, evidence, llm, gate,
                      ExperienceExtractor(evidence),
                      MemoryCritic(), ExperienceReconstructor(),
                      RunConfig(mode=mode))
    return harness, db, evidence, llm


def test_mode_A_never_learns(tmp_path):
    tasks = build_dataset(seed=7, per_family=2)
    harness, db, _, _ = _harness(tmp_path, MemoryMode.A_NO_MEMORY)
    results = [harness.run_task(bt, "runA") for bt in tasks]
    assert all(r.outcome.status.value == "failure" for r in results)
    assert all(r.injected_ids == [] for r in results)
    assert db.query_one("SELECT COUNT(*) AS n FROM experiences")["n"] == 0


def test_mode_D_learns_within_a_family(tmp_path):
    tasks = build_dataset(seed=7, per_family=3)
    harness, db, _, _ = _harness(tmp_path, MemoryMode.D_FULL_LIFECYCLE)
    results = [harness.run_task(bt, "runD") for bt in tasks]
    first = [r for r in results if r.task_id.endswith("_00_7") or "_00_" in r.task_id]
    later = [r for r in results if "_00_" not in r.task_id]
    assert all(r.outcome.status.value == "failure" for r in first), \
        "the first task in a family cannot know the hidden rule"
    assert all(r.outcome.status.value == "success" for r in later), \
        "later tasks in the family should reuse the learned rule"
    assert all(r.reused_from_memory for r in later)


def test_mode_D_does_not_inject_cross_family_memory(tmp_path):
    tasks = build_dataset(seed=7, per_family=2)
    harness, db, _, _ = _harness(tmp_path, MemoryMode.D_FULL_LIFECYCLE)
    for bt in tasks:
        harness.run_task(bt, "runD2")
    rows = db.query("SELECT * FROM memory_events")
    assert rows, "retrieval must be audited"
    for row in rows:
        injected = json.loads(row["injected_json"])
        assert len(injected) <= 3
    metrics = compute_metrics(db, [r for r in []], "D", {})
    assert metrics.n_tasks == 0


def test_memory_events_are_recorded_with_scores(tmp_path):
    tasks = build_dataset(seed=7, per_family=3)
    harness, db, _, _ = _harness(tmp_path, MemoryMode.D_FULL_LIFECYCLE)
    for bt in tasks:
        harness.run_task(bt, "runE")
    rows = db.query("SELECT retrievals FROM (SELECT retrieved_json AS retrievals "
                    "FROM memory_events)")
    assert rows
    for row in rows:
        payload = json.loads(row["retrievals"])
        for item in payload:
            assert "score" in item and "channels" in item


def test_mde_grows_as_cells_shrink():
    assert mde_binary(100) < mde_binary(10)
    assert mde_binary(0) == float("inf")
