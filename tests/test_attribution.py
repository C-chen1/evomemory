"""Attribution requires a recorded chain: rule exempted -> agent skipped -> grader required.

The previous logic struck any active rule that matched a missed record, without
evidence that the agent ever consulted it. These tests pin the stricter contract.
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
    EvidenceKind, Experience, ExperienceStatus, MemoryMode, Outcome, OutcomeStatus,
    VerifiedBy,
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


def _rule(db, gate, evidence, task_family="escalation"):
    ev = evidence.put_text("receipt")
    exp = Experience(lesson="For escalation, do not act on tier=gold.",
                     family=task_family, evidence_refs=[ev.evidence_id],
                     applicability_conditions=[f"task_family:{task_family}"])
    gate.propose(exp)
    gate.accept_unvalidated(exp)
    return exp


def test_strike_when_the_whole_chain_is_recorded(rig):
    harness, db, gate, evidence, bt = rig
    exp = _rule(db, gate, evidence)
    rid = bt.expected_ids[0]

    # 1. the rule said "exempt" for this record
    harness._record_rule_decisions(bt.task.task_id, {"ok": True, "result": {"results": [
        {"rid": rid, "exempt": True,
         "rules": [{"experience_id": exp.experience_id, "verdict": "true"}]}]}})
    # 2. the agent did not act on it (no task_actions row)
    # 3. the grader says it had to be acted on
    strikes = harness._attribute_failure(bt, None, _outcome([rid]))

    assert strikes == [exp.experience_id]
    assert db.query_one("SELECT COUNT(*) AS n FROM experience_strikes")["n"] == 1


def test_no_strike_when_the_agent_acted_anyway(rig):
    """The rule matched, but the action happened: the skip was not rule-caused."""
    harness, db, gate, evidence, bt = rig
    exp = _rule(db, gate, evidence)
    rid = bt.expected_ids[0]
    harness._record_rule_decisions(bt.task.task_id, {"ok": True, "result": {"results": [
        {"rid": rid, "exempt": True,
         "rules": [{"experience_id": exp.experience_id, "verdict": "true"}]}]}})
    harness._record_action(bt.task.task_id, "set_field",
                           {"rid": rid, "field": "status", "value": "escalated"})

    assert harness._attribute_failure(bt, None, _outcome([rid])) == []


def test_no_strike_when_the_agent_never_consulted_the_rule(rig):
    """Regression for the disabled logic: a matching rule alone is not evidence."""
    harness, db, gate, evidence, bt = rig
    _rule(db, gate, evidence)          # an active rule would match many records
    rid = bt.expected_ids[0]

    strikes = harness._attribute_failure(bt, None, _outcome([rid]))

    assert strikes == [], "no decision was recorded, so nothing can be attributed"
    assert db.query_one("SELECT COUNT(*) AS n FROM experience_strikes")["n"] == 0
    unknown = db.query_one("SELECT * FROM attribution_unknown")
    assert unknown is not None and unknown["rid"] == rid


def test_no_strike_when_the_rule_said_not_exempt(rig):
    harness, db, gate, evidence, bt = rig
    exp = _rule(db, gate, evidence)
    rid = bt.expected_ids[0]
    harness._record_rule_decisions(bt.task.task_id, {"ok": True, "result": {"results": [
        {"rid": rid, "exempt": False,
         "rules": [{"experience_id": exp.experience_id, "verdict": "false"}]}]}})

    assert harness._attribute_failure(bt, None, _outcome([rid])) == []


def test_unknown_verdict_is_not_a_strike(rig):
    harness, db, gate, evidence, bt = rig
    exp = _rule(db, gate, evidence)
    rid = bt.expected_ids[0]
    harness._record_rule_decisions(bt.task.task_id, {"ok": True, "result": {"results": [
        {"rid": rid, "exempt": None,
         "rules": [{"experience_id": exp.experience_id, "verdict": "unknown"}]}]}})

    assert harness._attribute_failure(bt, None, _outcome([rid])) == []


def test_retirement_needs_repeated_attributed_strikes(rig):
    harness, db, gate, evidence, bt = rig
    exp = _rule(db, gate, evidence)
    rid = bt.expected_ids[0]
    harness._record_rule_decisions(bt.task.task_id, {"ok": True, "result": {"results": [
        {"rid": rid, "exempt": True,
         "rules": [{"experience_id": exp.experience_id, "verdict": "true"}]}]}})

    harness._attribute_failure(bt, None, _outcome([rid]))
    assert harness._retire_on_strikes("t1") == [], "one strike is not enough"
    assert gate.store.get(exp.experience_id).status is ExperienceStatus.ACTIVE

    harness._attribute_failure(bt, None, _outcome([rid]))
    retired = harness._retire_on_strikes("t2")
    assert retired == [f"{exp.experience_id}@v1"]
    assert gate.store.get(exp.experience_id).status is ExperienceStatus.DEPRECATED
