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
