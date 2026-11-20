"""Benchmark environment and programmatic verification."""

import pytest

from evomemory.bench import (
    FAMILIES,
    OpsEnv,
    Record,
    build_dataset,
    build_table,
    task_env,
    verify,
)
from evomemory.models import OutcomeStatus, VerifiedBy


def test_dataset_is_deterministic():
    a = build_dataset(seed=7, per_family=3)
    b = build_dataset(seed=7, per_family=3)
    assert [t.task.task_id for t in a] == [t.task.task_id for t in b]
    assert [t.expected_ids for t in a] == [t.expected_ids for t in b]


def test_every_task_has_targets_and_exclusions():
    """A task with no targets would be trivially passable and is worthless."""
    for bt in build_dataset(seed=11, per_family=4):
        assert bt.expected_ids, f"{bt.task.task_id} has no targets"
        assert bt.excluded_ids, f"{bt.task.task_id} has no exclusions"
        assert bt.untouched_ids


def test_do_nothing_agent_fails():
    bt = build_dataset(seed=7, per_family=1)[0]
    outcome = verify(bt, task_env(bt))
    assert outcome.status is OutcomeStatus.FAILURE
    assert outcome.verified_by is VerifiedBy.PROGRAM


def test_correct_agent_succeeds():
    bt = build_dataset(seed=7, per_family=1)[0]
    env = task_env(bt)
    target = "escalated" if bt.action == "escalate" else "closed"
    for rid in bt.expected_ids:
        env.call("set_field", {"rid": rid, "field": "status", "value": target})
    env.call("log_entry", {"text": "done"})
    outcome = verify(bt, env)
    assert outcome.status is OutcomeStatus.SUCCESS, outcome.reason


def test_hidden_rule_violation_fails_and_names_the_attribute():
    bt = build_dataset(seed=7, per_family=1)[0]
    env = task_env(bt)
    target = "escalated" if bt.action == "escalate" else "closed"
    for rid in bt.expected_ids + bt.excluded_ids:
        env.call("set_field", {"rid": rid, "field": "status", "value": target})
    env.call("log_entry", {"text": "done"})
    outcome = verify(bt, env)
    assert outcome.status is OutcomeStatus.FAILURE
    detail = next(c["detail"] for c in outcome.checks
                  if c["name"] == "hidden_rule_respected")
    assert "tier='gold'" in detail or "note='hold'" in detail or "note='audit'" in detail
    # and it must not name attributes that were not the cause
    for rule in bt.rules:
        if rule.field == "tier":
            assert f"{rule.field}=" in detail


def test_collateral_change_fails():
    bt = build_dataset(seed=7, per_family=1)[0]
    env = task_env(bt)
    target = "escalated" if bt.action == "escalate" else "closed"
    for rid in bt.expected_ids:
        env.call("set_field", {"rid": rid, "field": "status", "value": target})
    env.call("log_entry", {"text": "done"})
    victim = bt.untouched_ids[0]
    env.call("set_field", {"rid": victim, "field": "status", "value": "closed"})
    assert verify(bt, env).status is OutcomeStatus.FAILURE


def test_missing_log_fails():
    bt = build_dataset(seed=7, per_family=1)[0]
    env = task_env(bt)
    target = "escalated" if bt.action == "escalate" else "closed"
    for rid in bt.expected_ids:
        env.call("set_field", {"rid": rid, "field": "status", "value": target})
    assert verify(bt, env).status is OutcomeStatus.FAILURE


def test_unknown_tool_is_reported_not_raised():
    env = OpsEnv([Record("r0", "eu", "silver", 3, "open")])
    result = env.call("drop_database", {})
    assert result["ok"] is False
    assert "unknown tool" in result["error"]


def test_write_to_unknown_field_is_refused():
    env = OpsEnv([Record("r0", "eu", "silver", 3, "open")])
    result = env.call("set_field", {"rid": "r0", "field": "password", "value": "x"})
    assert result["ok"] is False


def test_env_reset_restores_initial_state():
    env = OpsEnv([Record("r0", "eu", "silver", 3, "open")])
    env.call("set_field", {"rid": "r0", "field": "status", "value": "closed"})
    env.reset()
    assert env.records["r0"].status == "open"


def test_table_composition_is_controlled():
    table = build_table(7, "escalate", "eu", 3)
    tiers = [r.tier for r in table]
    assert "gold" in tiers
    assert any(r.note == "hold" for r in table)
    assert len(table) == 8


# -- conjunctive rule shape ------------------------------------------------


def test_conditional_family_has_conjunctive_decoy_structure():
    """The conjunction must be learnable *and* punishable from both sides."""
    tasks = build_dataset(seed=7, per_family=2, families=("conditional",))
    for bt in tasks:
        assert bt.expected_ids, "conditional tasks need targets"
        assert len(bt.excluded_ids) == 1, "exactly one blocked conjunction"
        env = task_env(bt)
        blocked = env.records[bt.excluded_ids[0]]
        assert blocked.tier == "gold" and blocked.note == "audit"
        # the two decoys each satisfy only one half and must be acted on
        decoys = [env.records[i] for i in bt.expected_ids
                  if env.records[i].tier == "gold" or env.records[i].note == "audit"]
        assert len(decoys) == 2, f"expected 2 decoys, got {[(d.rid, d.tier, d.note) for d in decoys]}"


def test_conditional_blocks_only_both_attributes():
    rule = FAMILIES["conditional"].rules[0]
    assert rule.blocks(Record("a", "eu", "gold", 4, "open", "audit")) is True
    assert rule.blocks(Record("b", "eu", "gold", 4, "open", "")) is False
    assert rule.blocks(Record("c", "eu", "silver", 4, "open", "audit")) is False


def test_conditional_violation_message_names_both_attributes():
    bt = build_dataset(seed=7, per_family=1, families=("conditional",))[0]
    env = task_env(bt)
    for rid in bt.expected_ids + bt.excluded_ids:
        env.call("set_field", {"rid": rid, "field": "status", "value": "escalated"})
    env.call("log_entry", {"text": "done"})
    outcome = verify(bt, env)
    detail = next(c["detail"] for c in outcome.checks
                  if c["name"] == "hidden_rule_respected")
    assert "tier='gold'" in detail and "note='audit'" in detail
