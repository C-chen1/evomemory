"""Extraction, validation, critique, reconstruction."""

import pytest

from evomemory.lifecycle import (
    ExperienceExtractor,
    ExperienceReconstructor,
    ExperienceValidator,
    MemoryCritic,
)
from evomemory.models import (
    CritiqueVerdict,
    ExecutionTrace,
    Experience,
    Outcome,
    OutcomeStatus,
    Task,
    ToolReceipt,
    TraceStep,
    VerifiedBy,
)


def make_trace(task_id="t1", tools=("list_records", "set_field")):
    trace = ExecutionTrace(task_id=task_id)
    for i, tool in enumerate(tools):
        trace.add(TraceStep(index=i, tool=tool, args={"i": i},
                            receipt=ToolReceipt(call_id=f"c{i}", tool=tool, ok=True,
                                                output={"ok": True})))
    trace.finish()
    return trace


def make_outcome(status=OutcomeStatus.FAILURE, verified=VerifiedBy.PROGRAM):
    return Outcome(status=status, verified_by=verified, score=0.5, reason="test",
                   checks=[{"name": "hidden_rule_respected", "passed": False,
                            "detail": "acted on records the rule excludes: "
                                      "r03(tier='gold'), r04(note='hold')"}])


def test_extractor_refuses_model_judged_outcome(tmp_env):
    _, evidence = tmp_env
    extractor = ExperienceExtractor(evidence)
    result = extractor.extract(Task(goal="g", family="escalation"),
                               make_trace(), make_outcome(verified=VerifiedBy.MODEL),
                               [])
    assert result.experiences == []
    assert "not program-verified" in result.error


def test_strict_extractor_does_not_hide_api_errors(tmp_env):
    from evomemory.llm import LLMError
    _, evidence = tmp_env
    class Failing:
        def complete(self, *args, **kwargs):
            raise LLMError("provider failure")
    extractor = ExperienceExtractor(evidence, Failing(), strict_errors=True)
    with pytest.raises(LLMError):
        extractor.extract(Task(goal="g", family="escalation"), make_trace(),
                          make_outcome(), make_outcome().checks)


def test_extractor_derives_the_blocking_attribute(tmp_env):
    _, evidence = tmp_env
    extractor = ExperienceExtractor(evidence)
    result = extractor.extract(Task(goal="g", family="escalation"), make_trace(),
                               make_outcome(), make_outcome().checks)
    lesson = result.experiences[0].lesson
    assert "tier=gold" in lesson
    assert "note=hold" in lesson


def test_extraction_creates_resolvable_evidence(tmp_env):
    _, evidence = tmp_env
    extractor = ExperienceExtractor(evidence)
    exp = extractor.extract(Task(goal="g", family="f"), make_trace(),
                            make_outcome(), make_outcome().checks).experiences[0]
    assert exp.evidence_refs
    assert all(evidence.exists(r) for r in exp.evidence_refs)
    assert exp.action_trace_refs == ["step:0:list_records", "step:1:set_field"]


def test_validator_accepts_a_well_formed_experience(tmp_env):
    _, evidence = tmp_env
    extractor = ExperienceExtractor(evidence)
    trace = make_trace()
    outcome = make_outcome()
    exp = extractor.extract(Task(goal="g", family="f"), trace, outcome,
                            outcome.checks).experiences[0]
    report = ExperienceValidator(evidence).validate(exp, trace=trace, outcome=outcome)
    assert report.passed, report.summary()


def test_validator_rejects_missing_evidence(tmp_env):
    _, evidence = tmp_env
    exp = Experience(lesson="A sufficiently long lesson that would otherwise pass.",
                     evidence_refs=["ev_does_not_exist"])
    report = ExperienceValidator(evidence).validate(exp)
    assert not report.passed
    assert any(c.name == "evidence_resolvable" for c in report.failures)


def test_validator_rejects_short_lesson(tmp_env):
    _, evidence = tmp_env
    ev = evidence.put_text("x")
    report = ExperienceValidator(evidence).validate(
        Experience(lesson="too short", evidence_refs=[ev.evidence_id]))
    assert not report.passed


def test_validator_rejects_evaluation_leakage(tmp_env):
    _, evidence = tmp_env
    ev = evidence.put_text("x")
    exp = Experience(lesson="The hidden answer is r03 and it must be gold tier.",
                     evidence_refs=[ev.evidence_id],
                     applicability_conditions=["task_family:f"])
    report = ExperienceValidator(evidence, leak_guard=["r03"]).validate(exp)
    assert not report.passed
    assert any(c.name == "no_eval_leakage" for c in report.failures)


def test_validator_rejects_duplicate(tmp_env):
    _, evidence = tmp_env
    ev = evidence.put_text("x")
    lesson = "For escalation, do not act on records with tier=gold."
    exp = Experience(lesson=lesson, evidence_refs=[ev.evidence_id],
                     applicability_conditions=["task_family:f"])
    report = ExperienceValidator(evidence, existing_texts=[lesson],
                                 duplicate_threshold=0.9).validate(exp)
    assert not report.passed
    assert any(c.name == "not_duplicate" for c in report.failures)


def test_model_veto_can_only_block(tmp_env):
    _, evidence = tmp_env
    ev = evidence.put_text("x")
    exp = Experience(lesson="A perfectly fine lesson that is long enough to pass.",
                     evidence_refs=[ev.evidence_id],
                     applicability_conditions=["task_family:f"])
    ok = ExperienceValidator(evidence).validate(exp, model_veto=(False, "looks fine"))
    bad = ExperienceValidator(evidence).validate(exp, model_veto=(True, "stale rule"))
    assert ok.passed
    assert not bad.passed


def test_critic_rejects_family_mismatch():
    critic = MemoryCritic()
    exp = Experience(lesson="l", family="closure",
                     applicability_conditions=["task_family:closure"])
    verdict = critic.critique(exp, Task(goal="escalate", family="escalation"))
    assert verdict.verdict is CritiqueVerdict.REJECT


def test_critic_accepts_matching_family_and_tags():
    critic = MemoryCritic()
    exp = Experience(lesson="l", family="escalation",
                     applicability_conditions=["task_family:escalation", "tag:ops"])
    verdict = critic.critique(exp, Task(goal="escalate records", family="escalation",
                                        tags=["ops"]))
    assert verdict.verdict in (CritiqueVerdict.APPLY, CritiqueVerdict.ADAPT)


def test_reconstructor_returns_none_on_reject():
    rec = ExperienceReconstructor()
    exp = Experience(lesson="something", family="a")
    from evomemory.models import Critique
    assert rec.reconstruct(exp, Task(goal="g"), Critique(CritiqueVerdict.REJECT)) is None


def test_reconstructor_produces_scoped_guidance():
    rec = ExperienceReconstructor()
    exp = Experience(lesson="do not escalate gold records", family="escalation")
    from evomemory.models import Critique
    text = rec.reconstruct(exp, Task(goal="g", family="escalation"),
                           Critique(CritiqueVerdict.APPLY))
    assert text and "escalation" in text and "gold" in text


class _StubLLM:
    """A model that always returns one fixed critique verdict."""

    model = "stub"

    def __init__(self, verdict: str):
        self.verdict = verdict
        self.calls = 0

    def complete(self, messages, **kwargs):
        import json as _json
        from evomemory.llm import LLMResponse
        self.calls += 1
        return LLMResponse(
            text=_json.dumps({"verdict": self.verdict,
                              "reasons": ["stub says so"], "confidence": 0.99}))


def test_family_mismatch_cannot_be_overridden_by_the_model():
    """A structural gate is not up for a vote.

    Regression test for a bug found in a live run: the critic asked the model for
    a verdict and the model enthusiastically applied an escalation lesson to a
    closure task — the exact negative transfer the critic exists to stop.
    """
    llm = _StubLLM("apply")
    critic = MemoryCritic(llm)
    exp = Experience(lesson="l", family="escalation",
                     applicability_conditions=["task_family:escalation"])
    verdict = critic.critique(exp, Task(goal="close records", family="closure"))
    assert verdict.verdict is CritiqueVerdict.REJECT
    assert verdict.hard is True
    assert llm.calls == 0, "the model must not even be consulted for a hard gate"


def test_model_may_still_veto_within_a_family():
    llm = _StubLLM("reject")
    critic = MemoryCritic(llm)
    exp = Experience(lesson="l", family="escalation",
                     applicability_conditions=["task_family:escalation"])
    verdict = critic.critique(exp, Task(goal="escalate", family="escalation"))
    assert verdict.verdict is CritiqueVerdict.REJECT
    assert llm.calls == 1
    assert verdict.source == "llm"
