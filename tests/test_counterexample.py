import json
from dataclasses import replace

import pytest

from evomemory.counterexample import CounterexampleMemoryValidator, ReplayExample
from evomemory.models import EvidenceKind, Experience
from evomemory.visual_bench import (PanelEnv, build_panel_task, structured_training_feedback,
                                    verify, visual_condition)


def test_feedback_is_post_scoring_and_excludes_holdout(tmp_env):
    _, store = tmp_env
    task = build_panel_task(40)
    env = PanelEnv(task)
    env.call("set_switch", {"zone": task.lockdown[0], "state": "on"})
    image = store.put(task.image, EvidenceKind.IMAGE)
    kwargs = dict(run_id="run", image_evidence_id=image.evidence_id, evidence_store=store)
    with pytest.raises(ValueError):
        structured_training_feedback(task, env, verify(task, env), split="test", **kwargs)
    ref = structured_training_feedback(task, env, verify(task, env), split="train", **kwargs)
    data = json.loads(store.read_text(ref))
    row = next(x for x in data["zones"] if x["zone"] == task.lockdown[0])
    assert row["action"] == "activate" and row["expected_action"] == "leave_off"
    assert row["ground_truth"]["lamp"] == "red"
    assert data["agent_observation"] is None and data["run_id"] == "run"
    assert store.get(ref).derived_from == image.evidence_id


def test_replay_rejects_wrong_rule_and_preserves_unknown(tmp_env):
    _, store = tmp_env
    evidence = store.put_text("verified observation")
    ex = lambda facts, action: ReplayExample("task", "run", evidence.evidence_id,
                                             "panel", facts, action)
    samples = [ex({"lamp": "red", "readout": 56}, "leave_off"),
               ex({"lamp": "red", "readout": 49}, "activate"),
               ex({"lamp": "amber", "readout": 56}, "activate")]
    candidate = Experience(lesson="Leave lockdown zones off", family="panel",
                           condition=visual_condition(), recommended_action="leave_off",
                           evidence_refs=[evidence.evidence_id])
    validator = CounterexampleMemoryValidator(store.exists)
    assert validator.validate(candidate, samples, "activate").decision == "accepted"
    wrong = replace(candidate, condition={"field": "lamp", "op": "eq", "value": "amber"})
    report = validator.validate(wrong, samples, "activate")
    assert report.decision == "rejected" and report.counterexamples
    unknown = [samples[0], ex({"lamp": "red"}, "activate")]
    assert validator.validate(candidate, unknown, "activate").decision == "pending"
    assert validator.validate(candidate, [replace(samples[0], split="test")], "activate").decision == "rejected"
