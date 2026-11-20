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
    EvidenceKind, Experience, ExperienceStatus, MemoryMode, RetrievedExperience, Task,
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
    guidance = "Do not act on records with tier=gold and note=hold."
    by_id = {r["rid"]: r for r in records}
    blocked = blocked_values_from_guidance(guidance, ["gold", "silver", "bronze", "hold"])
    chosen = [rid for rid in policy._in_scope(records, task)
              if not policy._blocked_by_text(by_id[rid], blocked)]
    assert chosen == ["b"]
    # the prose path cannot express a conjunction: it blocks gold AND hold
    assert set(blocked) == {"gold", "hold"}


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


def test_retrieval_is_scoped_to_the_task_family(tmp_path):
    """Regression: cross-family memories used to crowd out the right ones.

    A conditional-escalation task shares almost all of its wording with a plain
    escalation task, so unscoped BM25 ranked the escalation lessons above the
    task's own family. The critic then rejected all of them and the task ran with
    zero memory while the correct memory sat active and unused.
    """
    from evomemory.memory import ExperienceStore as _Store

    db = Database(tmp_path / "m.sqlite")
    evidence = EvidenceStore(tmp_path / "m_ev", db)
    store = _Store(db)
    for family, lesson in (
        ("escalation", "Escalate open records; exclude tier=gold and note=hold."),
        ("conditional", "Escalate open records; exclude tier=gold and note=audit."),
    ):
        e = Experience(lesson=lesson, family=family,
                       applicability_conditions=[f"task_family:{family}"])
        store.add(e)
        store.submit_for_validation(e.experience_id, 1)
        store.accept(e.experience_id, 1, reason="test")
        store.activate(e.experience_id, 1)

    llm = BudgetedLLM(ScriptedLLM(ScriptedOpsPolicy()), 50, 10_000, db)
    gate = MemoryGate(_Store(db), evidence)
    harness = Harness(db, evidence, llm, gate, ExperienceExtractor(evidence),
                      MemoryCritic(), ExperienceReconstructor(),
                      RunConfig(mode=MemoryMode.D_FULL_LIFECYCLE))
    task = Task(goal="Escalate every open record in region eu with priority >= 3.",
                task_id="t1", family="conditional")
    retrieved, block = harness.retrieve(task)
    assert retrieved, "retrieval must not be empty when the family has an active memory"
    assert all(r.experience.family == "conditional" for r in retrieved)
    assert block.injected_ids, "the task's own family memory must be injected"


# -- executable memory (P1) ------------------------------------------------


def test_mode_S_executes_a_conjunction_that_prose_gets_wrong(tmp_path):
    """The headline of P1: the same lesson, two representations, different outcomes.

    B injects "exclude tier=gold and note=audit" as prose; the reader collapses it
    to a disjunction and skips the decoys. S hands the stored predicate to code,
    which cannot collapse it.
    """
    tasks = build_dataset(seed=7, per_family=2, families=("conditional",))

    results = {}
    for mode in (MemoryMode.B_VECTOR_MEMORY, MemoryMode.S_STRUCTURED):
        sub = tmp_path / mode.value
        sub.mkdir()
        db = Database(sub / "m.db")
        evidence = EvidenceStore(sub / "ev", db)
        llm = BudgetedLLM(ScriptedLLM(ScriptedOpsPolicy()), 200, 5_000_000, db)
        gate = MemoryGate(ExperienceStore(db), evidence)
        h = Harness(db, evidence, llm, gate, ExperienceExtractor(evidence),
                    MemoryCritic(), ExperienceReconstructor(), RunConfig(mode=mode))
        results[mode] = [h.run_task(bt, f"run-{mode.value}") for bt in tasks]
        db.close()

    b_second = results[MemoryMode.B_VECTOR_MEMORY][1]
    s_second = results[MemoryMode.S_STRUCTURED][1]
    assert b_second.outcome.status.value == "failure", "prose path should over-exclude"
    assert s_second.outcome.status.value == "success", "executable path should discriminate"
    assert s_second.rule_checks > 0 and s_second.executable_rules > 0


def test_check_rule_reports_unknown_rather_than_false(tmp_path):
    """A rule that needs a missing field must say 'undetermined', not 'not exempt'."""
    db = Database(tmp_path / "m.db")
    evidence = EvidenceStore(tmp_path / "ev", db)
    gate = MemoryGate(ExperienceStore(db), evidence)
    llm = BudgetedLLM(ScriptedLLM(ScriptedOpsPolicy()), 10, 10_000, db)
    h = Harness(db, evidence, llm, gate, ExperienceExtractor(evidence),
                MemoryCritic(), ExperienceReconstructor(),
                RunConfig(mode=MemoryMode.S_STRUCTURED))
    from evomemory.bench import OpsEnv, Record
    env = OpsEnv([Record("r0", "eu", "gold", 3, "open")])
    rules = [("exp_x", {"field": "severity", "op": "eq", "value": "high"})]
    out = h.check_rule("r0", rules, env)
    assert out["ok"]
    assert out["result"]["exempt"] is None
    assert out["result"]["results"][0]["rules"][0]["verdict"] == "unknown"
    # batch form covers several records in one step
    env2 = OpsEnv([Record("r0", "eu", "gold", 3, "open"),
                   Record("r1", "eu", "gold", 3, "open", "audit")])
    batch = h.check_rule(None, [("exp_y", {"all_of": [
        {"field": "tier", "op": "eq", "value": "gold"},
        {"field": "note", "op": "eq", "value": "audit"}]})], env2,
        rids=["r0", "r1"])
    verdicts = {r["rid"]: r["exempt"] for r in batch["result"]["results"]}
    assert verdicts == {"r0": False, "r1": True}
