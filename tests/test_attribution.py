"""Attribution is graded by what was actually recorded, not by what seems likely.

Five distinct facts, and only a specific combination justifies a strike:

1. a rule was evaluated on the record and returned "exempt";
2. the agent saw that result            (guaranteed: the tool result is in the transcript);
3. the agent *acted* on it              — recorded only if it made an explicit skip;
4. no state change was made on the record;
5. the grader says the record had to be actioned.

An earlier version treated "rule said exempt" + "no set_field" as proof, but
silence is indistinguishable from forgetting, timing out, or running out of steps.
Only an explicit, cited skip raises this to a causal claim.
"""

import pytest

from evomemory.bench import build_dataset
from evomemory.db import Database
from evomemory.evidence import EvidenceStore
from evomemory.harness import Harness, MemoryGate, RunConfig
from evomemory.lifecycle import (
    ExperienceExtractor, ExperienceReconstructor, MemoryCritic,
)
from evomemory.llm import BudgetedLLM, StaticLLM
from evomemory.memory import ExperienceStore
from evomemory.models import (
    Experience, ExperienceStatus, MemoryMode, Outcome, OutcomeStatus, VerifiedBy,
)


@pytest.fixture()
def rig(tmp_path):
    db = Database(tmp_path / "m.db")
    evidence = EvidenceStore(tmp_path / "ev", db)
    gate = MemoryGate(ExperienceStore(db), evidence)
    llm = BudgetedLLM(StaticLLM("{}"), 10, 100_000, db)
    harness = Harness(db, evidence, llm, gate, ExperienceExtractor(evidence),
                      MemoryCritic(), ExperienceReconstructor(),
                      RunConfig(mode=MemoryMode.E_SELECTIVE, evolves=True))
    bt = build_dataset(seed=7, per_family=1, families=("escalation",))[0]
    return harness, db, gate, evidence, bt


def _outcome(missing):
    return Outcome(
        status=OutcomeStatus.FAILURE, verified_by=VerifiedBy.PROGRAM, score=0.5,
        reason="test",
        checks=[{"name": "targets_actioned", "passed": False,
                 "detail": f"1/2 done; missing {missing}"}])


def _rule(db, gate, evidence, family="escalation"):
    ev = evidence.put_text("receipt")
    exp = Experience(lesson="For escalation, do not act on tier=gold.",
                     family=family, evidence_refs=[ev.evidence_id],
                     applicability_conditions=[f"task_family:{family}"])
    gate.propose(exp)
    gate.accept_unvalidated(exp)
    return exp


def _decision(harness, task_id, rid, exp_ids, verdict="true"):
    harness._record_rule_decisions(task_id, {"ok": True, "result": {"results": [
        {"rid": rid, "exempt": verdict == "true",
         "rules": [{"experience_id": e, "verdict": verdict} for e in exp_ids]}]}})


# -- strong: explicit, cited skip -----------------------------------------


def test_cited_explicit_skip_is_a_strike(rig):
    harness, db, gate, evidence, bt = rig
    exp = _rule(db, gate, evidence)
    rid = bt.expected_ids[0]
    _decision(harness, bt.task.task_id, rid, [exp.experience_id])
    harness._record_action(bt.task.task_id, "skip_record",
                           {"rid": rid, "reason": "tier=gold is exempt",
                            "because_experience": exp.experience_id})

    assert harness._attribute_failure(bt, None, _outcome([rid])) == [exp.experience_id]
    strike = db.query_one("SELECT * FROM experience_strikes")
    assert "explicitly skipped citing this experience" in strike["reason"]


def test_retirement_needs_repeated_cited_skips(rig):
    harness, db, gate, evidence, bt = rig
    exp = _rule(db, gate, evidence)
    rid = bt.expected_ids[0]
    _decision(harness, bt.task.task_id, rid, [exp.experience_id])

    def one_round():
        harness._record_action(bt.task.task_id, "skip_record",
                               {"rid": rid, "reason": "exempt",
                                "because_experience": exp.experience_id})
        harness._attribute_failure(bt, None, _outcome([rid]))

    one_round()
    assert harness._retire_on_strikes("t1") == [], "one strike is not enough"
    one_round()
    assert harness._retire_on_strikes("t2") == [f"{exp.experience_id}@v1"]
    assert gate.store.get(exp.experience_id).status is ExperienceStatus.DEPRECATED


# -- weak: recorded, but never a strike -----------------------------------


def test_omission_is_weak_not_a_strike(rig):
    """Silence is indistinguishable from forgetting, so it cannot prove causation."""
    harness, db, gate, evidence, bt = rig
    exp = _rule(db, gate, evidence)
    rid = bt.expected_ids[0]
    _decision(harness, bt.task.task_id, rid, [exp.experience_id])

    assert harness._attribute_failure(bt, None, _outcome([rid])) == []
    weak = db.query_one("SELECT * FROM attribution_weak")
    assert weak["kind"] == "omission"
    assert db.query_one("SELECT COUNT(*) AS n FROM experience_strikes")["n"] == 0


def test_skip_without_citation_is_weak(rig):
    harness, db, gate, evidence, bt = rig
    exp = _rule(db, gate, evidence)
    rid = bt.expected_ids[0]
    _decision(harness, bt.task.task_id, rid, [exp.experience_id])
    harness._record_action(bt.task.task_id, "skip_record",
                           {"rid": rid, "reason": "looks exempt"})

    assert harness._attribute_failure(bt, None, _outcome([rid])) == []
    assert db.query_one("SELECT kind FROM attribution_weak")["kind"] == "skip_without_citation"


def test_skip_citing_a_different_experience_is_weak(rig):
    harness, db, gate, evidence, bt = rig
    exp = _rule(db, gate, evidence)
    other = _rule(db, gate, evidence, family="escalation")
    rid = bt.expected_ids[0]
    _decision(harness, bt.task.task_id, rid, [exp.experience_id])
    harness._record_action(bt.task.task_id, "skip_record",
                           {"rid": rid, "reason": "following memory",
                            "because_experience": other.experience_id})

    assert harness._attribute_failure(bt, None, _outcome([rid])) == []
    row = db.query_one("SELECT * FROM attribution_weak")
    assert row["kind"] == "skip_cites_other_experience"
    assert row["experience_id"] == other.experience_id


def test_no_rule_evaluated_is_unknown(rig):
    harness, db, gate, evidence, bt = rig
    _rule(db, gate, evidence)          # active, but never consulted for this record
    rid = bt.expected_ids[0]

    assert harness._attribute_failure(bt, None, _outcome([rid])) == []
    assert db.query_one("SELECT rid FROM attribution_unknown")["rid"] == rid
    assert db.query_one("SELECT COUNT(*) AS n FROM attribution_weak")["n"] == 0


def test_action_on_the_record_disqualifies_attribution(rig):
    harness, db, gate, evidence, bt = rig
    exp = _rule(db, gate, evidence)
    rid = bt.expected_ids[0]
    _decision(harness, bt.task.task_id, rid, [exp.experience_id])
    harness._record_action(bt.task.task_id, "set_field",
                           {"rid": rid, "field": "status", "value": "escalated"})

    assert harness._attribute_failure(bt, None, _outcome([rid])) == []


def test_rule_saying_not_exempt_is_not_a_cause(rig):
    harness, db, gate, evidence, bt = rig
    exp = _rule(db, gate, evidence)
    rid = bt.expected_ids[0]
    _decision(harness, bt.task.task_id, rid, [exp.experience_id], verdict="false")
    harness._record_action(bt.task.task_id, "skip_record",
                           {"rid": rid, "reason": "other reason",
                            "because_experience": exp.experience_id})

    assert harness._attribute_failure(bt, None, _outcome([rid])) == []


# -- conflicting rules and transactional switching ------------------------


def test_disagreeing_rules_are_recorded_not_silently_resolved(rig):
    harness, db, gate, evidence, bt = rig
    a = _rule(db, gate, evidence)
    b = _rule(db, gate, evidence)
    rid = bt.expected_ids[0]
    harness._record_rule_decisions(bt.task.task_id, {"ok": True, "result": {"results": [
        {"rid": rid, "exempt": True, "rules": [
            {"experience_id": a.experience_id, "verdict": "true"},
            {"experience_id": b.experience_id, "verdict": "false"}]}]}})

    row = db.query_one("SELECT * FROM rule_conflicts")
    assert row is not None and row["rid"] == rid
    assert a.experience_id in row["exempting"] and b.experience_id in row["denying"]


def test_version_switch_rolls_back_if_superseding_fails(rig):
    """A crash mid-switch must not leave two active versions."""
    harness, db, gate, evidence, bt = rig
    live = _rule(db, gate, evidence)
    v2 = gate.store.revise(live.experience_id, lesson="The corrected rule, long enough.")
    gate.store.submit_for_validation(v2.experience_id, v2.version)
    gate.store.accept(v2.experience_id, v2.version, reason="replay accepted")

    original = gate.store._apply
    calls = {"n": 0}

    def flaky(exp, target, actor, reason):
        calls["n"] += 1
        if calls["n"] == 1:            # the supersede of the predecessor
            raise RuntimeError("simulated crash while superseding")
        return original(exp, target, actor, reason)

    gate.store._apply = flaky           # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        gate.store.activate(v2.experience_id, v2.version)
    gate.store._apply = original        # type: ignore[assignment]

    assert gate.store.get(live.experience_id, 1).status is ExperienceStatus.ACTIVE
    assert gate.store.get(v2.experience_id, 2).status is ExperienceStatus.ACCEPTED
    assert len(gate.store.active()) == 1, "exactly one version may be active"
