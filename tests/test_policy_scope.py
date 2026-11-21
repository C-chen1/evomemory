"""Policy scopes: how old evidence stops blocking a correct new rule without letting
a wrong one through.

The mechanism is deliberately small. A *trusted external* release is attached to
each task (``BenchTask.policy_scope``); replay only draws on evidence from the same
release; experiences carry the release they were learned under; and a ``test``
split task writes no evidence at all. Nothing here detects drift on its own — the
caller has to say which release a task belongs to.
"""

import pytest

from evomemory.bench import apply_release, build_dataset
from evomemory.db import Database
from evomemory.evidence import EvidenceStore
from evomemory.harness import Harness, MemoryGate, RunConfig
from evomemory.lifecycle import (
    ExperienceReconstructor, ExtractionResult, MemoryCritic, _violation_attrs,
)
from evomemory.llm import BudgetedLLM, ScriptedLLM
from evomemory.memory import ExperienceStore
from evomemory.models import Experience, ExperienceStatus, MemoryMode
from evomemory.policy import ScriptedOpsPolicy
from evomemory.predicate import from_counterexamples

RULE_A = {"any_of": [{"field": "tier", "op": "eq", "value": "gold"},
                     {"field": "note", "op": "eq", "value": "hold"}]}
RULE_B = {"any_of": [{"field": "tier", "op": "eq", "value": "silver"},
                     {"field": "note", "op": "eq", "value": "hold"}]}


class FixedRule:
    """An extractor that proposes one predetermined condition."""

    def __init__(self, evidence, condition, lesson="A rule proposed for this test."):
        self.evidence, self.condition, self.lesson = evidence, condition, lesson

    def extract(self, task, trace, outcome, checks):
        ref = self.evidence.put_json(outcome.to_dict()).evidence_id
        return ExtractionResult([Experience(
            lesson=self.lesson, condition=self.condition, family=task.family,
            recommended_action="skip", task_id=task.task_id,
            trace_id=trace.trace_id, evidence_refs=[ref],
            applicability_conditions=[f"task_family:{task.family}"])], "test")


def rig(tmp_path, extractor_factory=None, mode=MemoryMode.E_SELECTIVE):
    db = Database(tmp_path / "m.db")
    evidence = EvidenceStore(tmp_path / "ev", db)
    gate = MemoryGate(ExperienceStore(db), evidence)
    llm = BudgetedLLM(ScriptedLLM(ScriptedOpsPolicy()), 500, 1_000_000, db)
    extractor = (extractor_factory or (lambda ev: FixedRule(ev, RULE_A)))(evidence)
    harness = Harness(db, evidence, llm, gate, extractor, MemoryCritic(),
                      ExperienceReconstructor(), RunConfig(mode=mode, evolves=True))
    return harness, db, gate, evidence


def tasks(n=6, drift_at=2, scopes=None, holdout=None):
    ts = build_dataset(seed=7, per_family=n, families=("escalation",),
                       drift_at=drift_at)
    apply_release(ts, scopes=scopes, holdout_from=holdout)
    return ts


# -- the extractor's connective inference (regression) ---------------------


def test_or_detail_learns_a_disjunction_and_and_detail_a_conjunction():
    """The verifier joins independent rules with 'or' and a conjunctive rule with
    'and'. Treating both as 'and' turns two exclusions into one conjunction."""
    or_detail = [{"name": "hidden_rule_respected", "passed": False,
                  "detail": "acted on: r00(tier='silver' or note='hold')"}]
    assert from_counterexamples(_violation_attrs(or_detail)) == {
        "any_of": [{"field": "tier", "op": "eq", "value": "silver"},
                   {"field": "note", "op": "eq", "value": "hold"}]}

    and_detail = [{"name": "hidden_rule_respected", "passed": False,
                   "detail": "acted on: r03(tier='gold' and note='audit')"}]
    assert from_counterexamples(_violation_attrs(and_detail)) == {
        "all_of": [{"field": "tier", "op": "eq", "value": "gold"},
                   {"field": "note", "op": "eq", "value": "audit"}]}


# -- 1/4/5: the headline claim --------------------------------------------


def test_old_evidence_does_not_permanently_block_a_correct_new_rule(tmp_path):
    harness, db, gate, evidence = rig(tmp_path)
    ts = tasks(n=6, drift_at=2, scopes=["A", "A", "B", "B", "B", "B"])

    harness.run_task(ts[0], "run")            # A: 1 task -> pending
    accepted_a = harness.run_task(ts[1], "run")
    assert accepted_a.replay_decisions[0]["decision"] == "accepted"
    old = {e.experience_id for e in gate.active_experiences()}
    assert old

    harness.extractor = FixedRule(evidence, RULE_B)
    first_b = harness.run_task(ts[2], "run")  # B: 1 task -> pending
    assert first_b.replay_decisions[0]["decision"] == "pending"
    second_b = harness.run_task(ts[3], "run")  # B: 2 tasks -> accepted
    assert second_b.replay_decisions[0]["decision"] == "accepted"

    # the old release's experience is still there, untouched and auditable
    for exp_id in old:
        assert gate.store.get(exp_id).status is ExperienceStatus.ACTIVE
    history = db.query_one("SELECT COUNT(*) AS n FROM training_examples")["n"]
    assert history > 0, "old training evidence must be retained, not deleted"


def test_a_wrong_candidate_is_rejected_by_its_own_scope_evidence(tmp_path):
    harness, db, gate, evidence = rig(tmp_path)
    ts = tasks(n=4, drift_at=None, scopes=["B", "B", "B", "B"])
    harness.extractor = FixedRule(evidence, RULE_B)   # silver/hold: wrong here

    harness.run_task(ts[0], "run")
    result = harness.run_task(ts[1], "run")
    assert result.replay_decisions[0]["decision"] == "rejected"
    assert not [e for e in gate.active_experiences()
                if e.condition == RULE_A and e.experience_id in
                {x.experience_id for x in gate.active_experiences()}]


def test_one_task_of_evidence_is_not_enough_for_a_scoped_candidate(tmp_path):
    harness, db, gate, evidence = rig(tmp_path)
    ts = tasks(n=3, drift_at=None, scopes=["B", "B", "B"])
    harness.extractor = FixedRule(evidence, RULE_A)   # consistent, but only 1 task

    result = harness.run_task(ts[0], "run")
    assert result.replay_decisions[0]["decision"] == "pending"
    assert gate.active_experiences() == []


# -- 6/7: scope isolation --------------------------------------------------


def test_scoped_experience_is_not_injected_into_an_unknown_scope(tmp_path):
    harness, db, gate, evidence = rig(tmp_path)
    ts = tasks(n=4, drift_at=None, scopes=["A", "A", None, None])

    harness.run_task(ts[0], "run")
    harness.run_task(ts[1], "run")
    assert gate.active_experiences()

    unknown = harness.run_task(ts[2], "run")
    assert unknown.injected_ids == [], "an unscoped task must not silently adopt A"


def test_scoped_experience_is_not_injected_into_another_scope(tmp_path):
    harness, db, gate, evidence = rig(tmp_path)
    ts = tasks(n=4, drift_at=None, scopes=["A", "A", "B", "B"])
    harness.run_task(ts[0], "run")
    harness.run_task(ts[1], "run")

    other = harness.run_task(ts[2], "run")
    assert other.injected_ids == []


def test_switching_back_to_the_old_scope_reuses_the_old_experience(tmp_path):
    harness, db, gate, evidence = rig(tmp_path)
    ts = tasks(n=6, drift_at=None, scopes=["A", "A", "B", "B", "A", "A"])
    harness.run_task(ts[0], "run")
    harness.run_task(ts[1], "run")
    old = {e.experience_id for e in gate.active_experiences()}

    harness.run_task(ts[2], "run")            # release B, nothing learned yet
    back = harness.run_task(ts[4], "run")     # back to A
    assert set(back.injected_ids) & old, "returning to a known release must reuse it"


# -- 11: the held-out split ------------------------------------------------


def test_test_split_tasks_write_no_evidence_and_leave_memory_alone(tmp_path):
    harness, db, gate, evidence = rig(tmp_path)
    ts = tasks(n=5, drift_at=None, scopes=["A"] * 5, holdout=3)
    harness.run_task(ts[0], "run")
    harness.run_task(ts[1], "run")
    before_examples = db.query_one("SELECT COUNT(*) AS n FROM training_examples")["n"]
    before_active = {e.experience_id for e in gate.active_experiences()}

    for bt in (ts[3], ts[4]):
        assert bt.feedback_split == "test"
        result = harness.run_task(bt, "run")
        assert result.memory_written == 0

    assert db.query_one("SELECT COUNT(*) AS n FROM training_examples")["n"] == before_examples
    assert {e.experience_id for e in gate.active_experiences()} == before_active
    assert not db.query("SELECT * FROM training_examples WHERE task_id IN (?, ?)",
                        (ts[3].task.task_id, ts[4].task.task_id))


# -- 12: recovery ----------------------------------------------------------


def test_recovery_refuses_a_task_whose_scope_changed(tmp_path):
    harness, db, gate, evidence = rig(tmp_path)
    ts = tasks(n=3, drift_at=None, scopes=["A", "A", "A"])
    harness.run_task(ts[0], "run")

    # a resumed process must not relabel an already graded task
    ts[0].policy_scope = "B"
    with pytest.raises(ValueError):
        harness.settle_memory_for(ts[0], "run")


def test_recovery_refuses_to_settle_a_test_task(tmp_path):
    harness, db, gate, evidence = rig(tmp_path)
    ts = tasks(n=3, drift_at=None, scopes=["A"] * 3, holdout=2)
    harness.run_task(ts[2], "run")
    with pytest.raises(ValueError):
        harness.settle_memory_for(ts[2], "run")


def test_interrupted_extraction_does_not_count_one_task_twice(tmp_path):
    from evomemory.llm import LLMError
    harness, db, gate, evidence = rig(tmp_path)
    task = tasks(n=1, drift_at=None, scopes=["A"])[0]

    class Interrupted:
        def extract(self, *args):
            raise LLMError("simulated extraction interruption")

    harness.extractor = Interrupted()
    with pytest.raises(LLMError):
        harness.run_task(task, "first-run")
    before = db.query_one("SELECT COUNT(*) AS n FROM training_examples")["n"]
    assert before > 0
    harness.extractor = FixedRule(evidence, RULE_A)
    result = harness.settle_memory_for(task, "resumed-run")
    assert db.query_one("SELECT COUNT(*) AS n FROM training_examples")["n"] == before
    assert result.replay_decisions[0]["decision"] == "pending"
    assert result.replay_decisions[0]["coverage"]["tasks"] == 1
    assert gate.active_experiences() == []


def test_runner_resume_rejects_changed_scope_and_split(tmp_path):
    from evomemory.config import Settings
    from evomemory.runner import run_experiment

    settings = Settings(root=tmp_path)
    kwargs = dict(settings=settings, modes=[MemoryMode.E_SELECTIVE], llm_kind="scripted",
                  per_family=2, families=("escalation",), run_dir=tmp_path / "run")
    run_experiment(**kwargs, policy_scopes=["A", "A"])
    with pytest.raises(ValueError, match="resume scope or split differs"):
        run_experiment(**kwargs, policy_scopes=["B", "B"], holdout_from=0)


# -- 13: version safety under a scope --------------------------------------


def test_a_rejected_revision_keeps_the_scoped_version_in_force(tmp_path):
    harness, db, gate, evidence = rig(tmp_path)
    ts = tasks(n=4, drift_at=None, scopes=["A"] * 4)
    harness.run_task(ts[0], "run")
    harness.run_task(ts[1], "run")
    live = gate.active_experiences()[0]

    revised = gate.store.revise(live.experience_id, condition=RULE_B,
                                lesson="A revision that replay will refuse.")
    examples = harness._replay_examples(live.family)
    admitted, report = gate.admit_with_replay(revised, examples=examples,
                                              default_action="activate",
                                              previous=live)
    assert not admitted and getattr(report, "decision", "") == "rejected"
    assert gate.store.get(live.experience_id, 1).status is ExperienceStatus.ACTIVE
    assert gate.store.get(live.experience_id, 2).status is ExperienceStatus.REJECTED
