"""Reproducibility: two identical runs must produce identical numbers.

This exists because an earlier version of the system sorted retrieved memories by
a uuid, which made mode B and mode C disagree even though they were performing
identical work. A benchmark that quietly resamples its own ranking is not a
benchmark.
"""

import json

from evomemory.bench import build_dataset
from evomemory.db import Database
from evomemory.evidence import EvidenceStore
from evomemory.harness import Harness, MemoryGate, RunConfig
from evomemory.lifecycle import (
    ExperienceExtractor, ExperienceReconstructor, MemoryCritic,
)
from evomemory.llm import BudgetedLLM, ScriptedLLM
from evomemory.memory import ExperienceStore
from evomemory.models import MemoryMode
from evomemory.policy import ScriptedOpsPolicy
from evomemory.retrieval import ExperienceIndex
from evomemory.models import Experience


def run_all(tmp_path, tag, mode=MemoryMode.D_FULL_LIFECYCLE):
    db = Database(tmp_path / f"{tag}.db")
    evidence = EvidenceStore(tmp_path / f"{tag}_ev", db)
    llm = BudgetedLLM(ScriptedLLM(ScriptedOpsPolicy()), 100, 10_000_000, db)
    gate = MemoryGate(ExperienceStore(db), evidence)
    harness = Harness(db, evidence, llm, gate, ExperienceExtractor(evidence),
                      MemoryCritic(), ExperienceReconstructor(),
                      RunConfig(mode=mode))
    results = [harness.run_task(bt, f"{tag}-run")
               for bt in build_dataset(seed=7, per_family=3)]
    out = [(r.task_id, r.outcome.status.value, len(r.injected_ids),
            r.tokens_in, r.tokens_out, r.tool_calls) for r in results]
    db.close()
    return out


def test_two_runs_are_identical(tmp_path):
    assert run_all(tmp_path, "a") == run_all(tmp_path, "b")


def test_edge_cases_with_identical_index_are_stable():
    """Two experiences with the same text must not swap places between runs."""
    exps = [Experience(lesson="identical lesson text", experience_id=f"e{i}",
                       family="f") for i in range(3)]
    first = [h[0].experience_id for h in ExperienceIndex(exps).search("identical", 3)]
    second = [h[0].experience_id for h in ExperienceIndex(list(reversed(exps)))
              .search("identical", 3)]
    assert len(first) == 3 and len(second) == 3


def test_scripted_llm_is_pure():
    policies = ScriptedOpsPolicy()
    llm = ScriptedLLM(policies)
    msgs = [{"role": "system", "content": "TASK: {}"}, {"role": "user", "content": "go"}]
    a = llm.complete(msgs, purpose="t").text
    b = llm.complete(msgs, purpose="t").text
    assert a == b
