import pytest

from evomemory.llm import BudgetedLLM, ScriptedLLM
from evomemory.visual_bench import visual_condition
from evomemory.visual_trial import PanelTrainer, execute_panel, make_split, make_training_probe


def test_holdout_does_not_enter_feedback_or_extraction(tmp_env):
    db, evidence = tmp_env
    train, test = make_split([100, 101], [200, 201])
    observed = []
    def respond(messages):
        observed.append(messages)
        return '{"tool":"finish","args":{}}'
    llm = BudgetedLLM(ScriptedLLM(respond), 20, 100_000, db)
    trainer = PanelTrainer(db, evidence, "r", {t.task.task_id for t in train})
    result = execute_panel(llm, db, train[0], "r", [])
    trainer.record(train[0], result)
    holdout_result = execute_panel(llm, db, test[0], "r", [])
    with pytest.raises(ValueError):
        trainer.record(test[0], holdout_result)
    assert {f["task_id"] for f in trainer.feedback_for_extraction()} == {train[0].task.task_id}
    assert "ground_truth" not in str(observed) and "expected_action" not in str(observed)
    exp, report, _ = trainer.propose(visual_condition())
    assert report.decision == "accepted"
    assert trainer.gate.active_experiences()[0].experience_id == exp.experience_id


def test_split_rejects_overlapping_seeds():
    with pytest.raises(ValueError):
        make_split([1, 2], [2, 3])


def test_new_task_wording_does_not_disclose_the_exception():
    train, test = make_split([3000], [4000], dataset_version="panel-v3")
    goal = train[0].task.goal.lower()
    assert "operating restrictions" in goal
    assert "lockdown" not in goal and "50" not in goal
    assert train[0].task.context["dataset_version"] == "panel-v3"
    assert test[0].task.context["dataset_version"] == "panel-v3"


def test_threshold_trainer_preserves_ambiguous_boundary(tmp_env):
    db, evidence = tmp_env
    train, _ = make_split([1000], [2000])
    trainer = PanelTrainer(db, evidence, "r", {train[0].task.task_id})
    llm = BudgetedLLM(ScriptedLLM(lambda _: '{"tool":"finish","args":{}}'), 10, 100_000, db)
    result = execute_panel(llm, db, train[0], "r", [])
    trainer.record(train[0], result)
    induction = trainer.induce_threshold()
    assert induction["induction"]["decision"] == "pending"
    assert evidence.exists(induction["induction_evidence_id"])
    assert trainer.guidance() == []


def test_adaptive_training_enumerates_boundary_without_holdout(tmp_env):
    db, evidence = tmp_env
    train, test = make_split(range(1000, 1008), [2000])
    trainer = PanelTrainer(db, evidence, "r", {x.task.task_id for x in train})
    llm = BudgetedLLM(ScriptedLLM(lambda _: '{"tool":"finish","args":{}}'), 40, 100_000, db)
    induction = None
    for i, bt in enumerate(train):
        if induction:
            values = induction["induction"].get("coverage", {}).get("consistent_thresholds", [])
            if len(values) > 1:
                bt = make_training_probe(1000 + i, bt.task.task_id, (values[0] + values[-1]) // 2)
        trainer.record(bt, execute_panel(llm, db, bt, "r", trainer.guidance()))
        induction = trainer.induce_threshold()
    assert induction["induction"]["coverage"]["consistent_thresholds"] == [50]
    assert len(trainer.guidance()) == 1
    assert all(f["task_id"] != test[0].task.task_id for f in trainer.feedback_for_extraction())
    with pytest.raises(ValueError):
        make_training_probe(2000, "test-000", 50)
