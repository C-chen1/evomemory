import pytest

from evomemory.llm import BudgetedLLM, ScriptedLLM
from evomemory.visual_bench import visual_condition
from evomemory.visual_trial import PanelTrainer, execute_panel, make_split


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
