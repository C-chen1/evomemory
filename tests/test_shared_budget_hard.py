"""Shared budget behaviour under interruption, unknown usage and concurrency.

The rule the ledger exists to enforce: a request that cannot be reconciled must
block the round rather than be counted as free.
"""

import sqlite3

import pytest

from evomemory.llm import BudgetExceeded
from evomemory.shared_budget import SharedBudget


def make(tmp_path, calls=10, tokens=10_000):
    return SharedBudget(tmp_path / "round.db", max_calls=calls, max_tokens=tokens)


def test_reserve_then_settle_accounts_actual_usage(tmp_path):
    b = make(tmp_path)
    rid = b.reserve(run_id="r", task_id="t", purpose="act", tokens=100)
    b.settle(rid, actual=40)
    summary = b.summary()
    assert summary["calls"] == 1
    assert summary["accounted_tokens"] == 40


def test_call_cap_is_enforced(tmp_path):
    b = make(tmp_path, calls=2)
    b.settle(b.reserve(run_id="r", task_id="t", purpose="act", tokens=10), 10)
    b.settle(b.reserve(run_id="r", task_id="t", purpose="act", tokens=10), 10)
    with pytest.raises(BudgetExceeded):
        b.reserve(run_id="r", task_id="t", purpose="act", tokens=10)


def test_token_cap_counts_reservations_before_they_are_settled(tmp_path):
    b = make(tmp_path, tokens=100)
    b.reserve(run_id="r", task_id="t", purpose="act", tokens=60)
    with pytest.raises(BudgetExceeded):
        b.reserve(run_id="r", task_id="t", purpose="act", tokens=60)


def test_crash_between_reserve_and_settle_blocks_the_round(tmp_path):
    """A request we cannot reconcile must not be treated as free."""
    b = make(tmp_path)
    b.reserve(run_id="r", task_id="t", purpose="act", tokens=100)
    # process dies here: no settle ever happens
    with pytest.raises(BudgetExceeded):
        b.reserve(run_id="r", task_id="t", purpose="act", tokens=100)


def test_unknown_usage_retains_the_reservation_and_blocks(tmp_path):
    b = make(tmp_path)
    rid = b.reserve(run_id="r", task_id="t", purpose="act", tokens=500)
    b.settle(rid, actual=None)
    summary = b.summary()
    assert summary["unknown"] == 1
    assert summary["accounted_tokens"] == 500, "unknown usage keeps its reservation"
    with pytest.raises(BudgetExceeded):
        b.reserve(run_id="r", task_id="t", purpose="act", tokens=1)


def test_overrun_is_recorded_and_blocks(tmp_path):
    """Provider usage above the estimate means the estimate cannot be trusted."""
    b = make(tmp_path)
    rid = b.reserve(run_id="r", task_id="t", purpose="act", tokens=100)
    b.settle(rid, actual=250)
    assert b.summary()["overruns"] == 1
    with pytest.raises(BudgetExceeded):
        b.reserve(run_id="r", task_id="t", purpose="act", tokens=1)


def test_settling_twice_is_refused(tmp_path):
    b = make(tmp_path)
    rid = b.reserve(run_id="r", task_id="t", purpose="act", tokens=10)
    b.settle(rid, actual=5)
    with pytest.raises(ValueError):
        b.settle(rid, actual=5)


def test_cannot_change_the_budget_of_an_existing_round(tmp_path):
    make(tmp_path, calls=10, tokens=1000)
    with pytest.raises(ValueError):
        make(tmp_path, calls=99, tokens=1000)


def test_two_handles_share_one_ledger(tmp_path):
    """Two experiment scripts share the ledger; settled usage is what counts."""
    a = make(tmp_path, calls=10, tokens=100)
    b = make(tmp_path, calls=10, tokens=100)
    a.settle(a.reserve(run_id="a", task_id="t", purpose="act", tokens=60), 60)
    b.settle(b.reserve(run_id="b", task_id="t", purpose="act", tokens=30), 30)
    with pytest.raises(BudgetExceeded):
        b.reserve(run_id="b", task_id="t", purpose="act", tokens=30)


def test_an_in_flight_request_blocks_other_handles(tmp_path):
    """Stricter than the token cap: an unsettled request may still overrun.

    The second handle cannot know whether the first one's reservation will be
    honoured, so it waits. This is deliberate; the cost is that a crashed script
    stops the round until an operator reconciles it.
    """
    a = make(tmp_path, calls=10, tokens=1000)
    b = make(tmp_path, calls=10, tokens=1000)
    a.reserve(run_id="a", task_id="t", purpose="act", tokens=10)
    with pytest.raises(BudgetExceeded):
        b.reserve(run_id="b", task_id="t", purpose="act", tokens=10)


def test_summary_reports_the_cap_for_operator_checks(tmp_path):
    b = make(tmp_path, calls=7, tokens=777)
    s = b.summary()
    assert s["max_calls"] == 7 and s["max_tokens"] == 777 and s["calls"] == 0


def test_reconcile_is_deliberate_and_not_automatic(tmp_path):
    """There is no public 'forgive everything' path; recovery must be explicit.

    If an operator decides a pending reservation never reached the provider, they
    void it by hand. The point is that no code path does this silently.
    """
    b = make(tmp_path)
    b.reserve(run_id="r", task_id="t", purpose="act", tokens=100)
    assert not hasattr(b, "forgive") and not hasattr(b, "reset")
    raw = sqlite3.connect(tmp_path / "round.db")
    states = [r[0] for r in raw.execute("SELECT state FROM budget_calls")]
    assert states == ["pending"]
