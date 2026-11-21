import pytest

from evomemory.llm import BudgetedLLM, BudgetExceeded, LLMError, LLMResponse
from evomemory.shared_budget import SharedBudget


class Client:
    model = "test"
    def __init__(self, result):
        self.result = result
    def complete(self, messages, **kwargs):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_shared_round_persists_across_clients_and_counts_failures(tmp_path):
    path = tmp_path / "round.db"
    shared = SharedBudget(path, 2, 100_000)
    first = BudgetedLLM(Client(LLMResponse("ok", 12, 8, estimated=False)), 10, 100_000,
                        shared_budget=shared)
    first.complete([{"role": "user", "content": "hello"}], purpose="act")
    failure = BudgetedLLM(Client(LLMError("empty", input_tokens=30, output_tokens=120)),
                          10, 100_000, shared_budget=shared)
    with pytest.raises(LLMError):
        failure.complete([{"role": "user", "content": "hello"}], purpose="extract")
    assert shared.summary()["accounted_tokens"] == 170
    shared.close()
    resumed = SharedBudget(path, 2, 100_000)
    assert resumed.summary()["calls"] == 2
    with pytest.raises(BudgetExceeded):
        first = BudgetedLLM(Client(LLMResponse("ok", 1, 1, estimated=False)), 10, 100_000,
                            shared_budget=resumed)
        first.complete([{"role": "user", "content": "x"}], purpose="act")
    resumed.close()


def test_unknown_failure_blocks_future_requests_after_restart(tmp_path):
    path = tmp_path / "round.db"
    shared = SharedBudget(path, 5, 100_000)
    llm = BudgetedLLM(Client(TimeoutError("timeout")), 5, 100_000, shared_budget=shared)
    with pytest.raises(TimeoutError):
        llm.complete([{"role": "user", "content": "x"}], purpose="act")
    assert shared.summary()["calls"] == 1 and shared.summary()["unknown"] == 1
    shared.close()
    resumed = SharedBudget(path, 5, 100_000)
    with pytest.raises(BudgetExceeded):
        resumed.reserve(run_id="r", task_id="t", purpose="act", tokens=1)
    resumed.close()
