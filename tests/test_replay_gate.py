"""Candidate admission must pass replay over recorded training evidence.

A schema-valid rule is not an acceptable rule. The candidate has to reproduce the
observed action on every training example, cover both action classes, and leave no
unknown evaluations behind. A rejected candidate must not disturb the live version.
"""

import pytest

from evomemory.counterexample import ReplayExample, condition_diff
from evomemory.db import Database
from evomemory.evidence import EvidenceStore
from evomemory.harness import MemoryGate
from evomemory.memory import ExperienceStore
from evomemory.models import Experience, ExperienceStatus, MemoryMode


@pytest.fixture()
def rig(tmp_path):
    db = Database(tmp_path / "m.db")
    evidence = EvidenceStore(tmp_path / "ev", db)
    gate = MemoryGate(ExperienceStore(db), evidence)
    return gate, db, evidence


def _examples(evidence, rows, family="panel"):
    out = []
    for i, (facts, action) in enumerate(rows):
        ev = evidence.put_json({"zones": []}, metadata={"kind": "structured_training_feedback"})
        out.append(ReplayExample(task_id=f"t{i}", run_id="r1", evidence_id=ev.evidence_id,
                                 family=family, facts=facts, expected_action=action,
                                 split="train"))
    return out


THRESHOLD = {"field": "readout", "op": "gte", "value": 50}
ROWS = [
    ({"lamp": "red", "readout": 35}, "activate"),
    ({"lamp": "red", "readout": 56}, "leave_off"),
    ({"lamp": "red", "readout": 49}, "activate"),
    ({"lamp": "red", "readout": 80}, "leave_off"),
]


def _candidate(evidence, condition, action="leave_off"):
    ev = evidence.put_text("receipt")
    return Experience(lesson="For red panel zones, leave the switch off when readout is high.",
                      condition=condition, recommended_action=action,
                      family="panel", evidence_refs=[ev.evidence_id],
                      applicability_conditions=["task_family:panel"])


def test_correct_rule_is_admitted(rig):
    gate, db, evidence = rig
    exp = _candidate(evidence, THRESHOLD)
    gate.propose(exp)
    ok, report = gate.admit_with_replay(exp, examples=_examples(evidence, ROWS),
                                        default_action="activate")
    assert ok, report.reasons
    assert gate.store.get(exp.experience_id).status is ExperienceStatus.ACTIVE


def test_rule_that_contradicts_evidence_is_rejected(rig):
    gate, db, evidence = rig
    wrong = {"field": "readout", "op": "gte", "value": 30}   # would exempt readout 35
    exp = _candidate(evidence, wrong)
    gate.propose(exp)
    ok, report = gate.admit_with_replay(exp, examples=_examples(evidence, ROWS),
                                        default_action="activate")
    assert not ok
    assert report.counterexamples, "the contradiction must be shown, not just refused"
    assert gate.store.get(exp.experience_id).status is ExperienceStatus.REJECTED
    assert gate.active_experiences() == []


def test_underdetermined_rule_stays_inactive(rig):
    """Consistent with one boundary point is not enough to know the boundary."""
    gate, db, evidence = rig
    thin = _examples(evidence, [({"lamp": "red", "readout": 40}, "activate")])
    exp = _candidate(evidence, THRESHOLD)
    gate.propose(exp)
    ok, report = gate.admit_with_replay(exp, examples=thin, default_action="activate")
    assert not ok
    assert report.decision == "pending"
    assert gate.store.get(exp.experience_id).status is ExperienceStatus.CANDIDATE
    assert gate.active_experiences() == []


def test_rejected_candidate_leaves_the_live_version_active(rig):
    gate, db, evidence = rig
    live = _candidate(evidence, THRESHOLD)
    gate.propose(live)
    ok, _ = gate.admit_with_replay(live, examples=_examples(evidence, ROWS),
                                   default_action="activate")
    assert ok

    revised = gate.store.revise(live.experience_id,
                                condition={"field": "readout", "op": "gte", "value": 30},
                                lesson="A wrong revision that replay will refuse.")
    bad, _ = gate.admit_with_replay(revised, examples=_examples(evidence, ROWS),
                                    default_action="activate", previous=live)
    assert not bad
    assert gate.store.get(live.experience_id, 1).status is ExperienceStatus.ACTIVE
    assert gate.store.get(live.experience_id, 2).status is ExperienceStatus.REJECTED


def test_holdout_examples_are_refused(rig):
    gate, db, evidence = rig
    exp = _candidate(evidence, THRESHOLD)
    gate.propose(exp)
    examples = _examples(evidence, ROWS)
    examples[-1] = ReplayExample(**{**examples[-1].__dict__, "split": "test"})
    ok, report = gate.admit_with_replay(exp, examples=examples, default_action="activate")
    assert not ok and any("non-training" in r for r in report.reasons)


# -- structured diffs ------------------------------------------------------


def test_diff_reports_a_connective_change():
    before = {"all_of": [{"field": "tier", "op": "eq", "value": "gold"},
                         {"field": "note", "op": "eq", "value": "audit"}]}
    after = {"any_of": [{"field": "tier", "op": "eq", "value": "gold"},
                        {"field": "note", "op": "eq", "value": "audit"}]}
    diff = condition_diff(before, after)
    assert (diff["connective_before"], diff["connective_after"]) == ("all_of", "any_of")
    assert diff["added"] == [] and diff["removed"] == []
    assert diff["changed"] is True, "a connective flip must register as a change"


def test_diff_reports_added_and_removed_leaves():
    before = {"field": "tier", "op": "eq", "value": "gold"}
    after = {"any_of": [{"field": "tier", "op": "eq", "value": "silver"},
                        {"field": "note", "op": "eq", "value": "hold"}]}
    diff = condition_diff(before, after)
    assert "tier eq 'silver'" in diff["added"]
    assert "tier eq 'gold'" in diff["removed"]


def test_diff_of_identical_conditions_is_empty():
    diff = condition_diff(THRESHOLD, dict(THRESHOLD))
    assert diff["changed"] is False and not diff["added"] and not diff["removed"]


# -- the mainline actually calls the replay gate ---------------------------


def _mainline(tmp_path, mode=MemoryMode.D_FULL_LIFECYCLE):
    from evomemory.bench import build_dataset
    from evomemory.harness import Harness, RunConfig
    from evomemory.llm import BudgetedLLM, ScriptedLLM
    from evomemory.policy import ScriptedOpsPolicy
    from evomemory.lifecycle import (ExperienceExtractor, ExperienceReconstructor,
                                     MemoryCritic)

    db = Database(tmp_path / "main.db")
    evidence = EvidenceStore(tmp_path / "main_ev", db)
    gate = MemoryGate(ExperienceStore(db), evidence)
    harness = Harness(db, evidence,
                      BudgetedLLM(ScriptedLLM(ScriptedOpsPolicy()), 100, 1_000_000, db),
                      gate, ExperienceExtractor(evidence), MemoryCritic(),
                      ExperienceReconstructor(), RunConfig(mode=mode))
    return harness, db, gate, evidence, build_dataset(seed=7, per_family=3,
                                                      families=("escalation",))


def test_mainline_records_replay_evidence_and_decision(tmp_path):
    harness, db, gate, evidence, tasks = _mainline(tmp_path)
    first = harness.run_task(tasks[0], "run")

    assert first.replay_decisions, "the mainline must report a replay decision"
    assert first.replay_decisions[0]["decision"] == "accepted"
    assert db.query_one("SELECT COUNT(*) AS n FROM training_examples")["n"] > 0
    # examples are limited to the records the task filter selected
    scope = db.query_one(
        "SELECT COUNT(DISTINCT rid) AS n FROM training_examples")["n"]
    assert scope <= len(tasks[0].expected_ids) + len(tasks[0].excluded_ids)
    assert gate.active_experiences(), "an accepted candidate must be activated"


def test_accepted_candidate_is_reused_by_the_next_task(tmp_path):
    harness, db, gate, evidence, tasks = _mainline(tmp_path)
    harness.run_task(tasks[0], "run")
    second = harness.run_task(tasks[1], "run")

    assert second.reused_from_memory, "the next task must retrieve the accepted memory"
    assert second.outcome.status.value == "success"


def test_a_candidate_contradicting_training_evidence_is_not_activated(tmp_path):
    """Known counterexample blocks activation, through the mainline path."""
    harness, db, gate, evidence, tasks = _mainline(tmp_path)
    harness.run_task(tasks[0], "run")          # seeds training examples

    wrong = Experience(
        lesson="Every record should be skipped, always, for any reason at all.",
        condition={"field": "region", "op": "exists"},
        recommended_action="skip", family="escalation",
        evidence_refs=[evidence.put_text("receipt").evidence_id],
        applicability_conditions=["task_family:escalation"])

    class _Fixed:
        def extract(self, *a, **k):
            from evomemory.lifecycle import ExtractionResult
            return ExtractionResult([Experience(**{**wrong.__dict__})], source="test")

    harness.extractor = _Fixed()               # type: ignore[assignment]
    result = harness.run_task(tasks[1], "run")

    decisions = [d["decision"] for d in result.replay_decisions]
    assert "rejected" in decisions, decisions
    active_lessons = [e.lesson for e in gate.active_experiences()]
    assert not any("always" in lesson for lesson in active_lessons)
