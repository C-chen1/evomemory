"""Programmatic replay of candidate exception rules over supervised examples.

This module returns a decision; callers must enforce it at their admission gate.
Consistency with observed examples is not proof of future correctness.
"""
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from .models import Experience
from .predicate import PredicateError, Tri, describe, evaluate, validate


@dataclass(frozen=True)
class ReplayExample:
    task_id: str
    run_id: str
    evidence_id: str
    family: str
    facts: Mapping[str, Any]
    expected_action: str
    split: str = "train"


@dataclass
class ReplayReport:
    decision: str
    reasons: list[str] = field(default_factory=list)
    counterexamples: list[dict] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)


def panel_guidance(condition: dict) -> str:
    """Generate guidance from the checked predicate, never model-written prose."""
    validate(condition)
    return f"For red panel zones, leave the switch off when {describe(condition)}; otherwise activate."


class CounterexampleMemoryValidator:
    """Validate a rule whose match recommends an action, with a default action.

    Both positive and negative examples are required; unknown evaluations remain
    pending. Holdout examples are forbidden, even when a caller passes them by
    accident. Evidence lookup is supplied by the owning evidence store.
    """

    def __init__(self, evidence_exists: Callable[[str], bool]):
        self.evidence_exists = evidence_exists

    def validate(self, candidate: Experience, examples: Sequence[ReplayExample],
                 default_action: str, previous: Optional[Experience] = None) -> ReplayReport:
        errors = []
        try:
            validate(candidate.condition)
        except (PredicateError, TypeError, ValueError) as exc:
            errors.append(f"schema: {exc}")
        if not candidate.family or not candidate.recommended_action or not default_action:
            errors.append("family and actions are required")
        if candidate.recommended_action == default_action:
            errors.append("recommended action must differ from default")
        if not candidate.evidence_refs or any(not self.evidence_exists(x) for x in candidate.evidence_refs):
            errors.append("candidate evidence missing or unresolved")
        if previous and (candidate.experience_id != previous.experience_id or
                         candidate.version != previous.version + 1 or
                         candidate.supersedes != previous.version):
            errors.append("invalid memory version lineage")
        if any(e.split != "train" for e in examples):
            errors.append("non-training example supplied")
        if any(e.family != candidate.family for e in examples):
            errors.append("task family mismatch")
        if any(not e.task_id or not e.run_id or not e.evidence_id or
               not self.evidence_exists(e.evidence_id) for e in examples):
            errors.append("example provenance missing or unresolved")
        if errors:
            return ReplayReport("rejected", errors)

        counts = {"positive": 0, "negative": 0, "unknown": 0, "tasks": 0}
        counts["tasks"] = len({(e.run_id, e.task_id) for e in examples})
        conflicts = []
        for e in examples:
            verdict = evaluate(candidate.condition, e.facts)
            if verdict is Tri.UNKNOWN:
                counts["unknown"] += 1
                continue
            counts["positive" if verdict is Tri.TRUE else "negative"] += 1
            action = candidate.recommended_action if verdict is Tri.TRUE else default_action
            if action != e.expected_action:
                conflicts.append({"task_id": e.task_id, "run_id": e.run_id,
                                  "evidence_id": e.evidence_id, "facts": dict(e.facts),
                                  "expected_action": e.expected_action, "actual_action": action})
        if conflicts:
            return ReplayReport("rejected", ["action conflicts with observed evidence"], conflicts, counts)
        if counts["unknown"] or not counts["positive"] or not counts["negative"]:
            return ReplayReport("pending", ["coverage insufficient or unknown"], coverage=counts)
        return ReplayReport("accepted", ["consistent with supplied training evidence"], coverage=counts)


def panel_examples_from_feedback(store, feedback_ids: Sequence[str]) -> list[ReplayExample]:
    """Read supervised panel examples within the red-lamp task scope.

    The exception predicts whether an otherwise eligible red zone must remain off.
    Amber and green zones are outside that scope, not default-activate examples.
    """
    result = []
    for ref in feedback_ids:
        ev = store.get(ref)
        raw = store.read_bytes(ref)
        if hashlib.sha256(raw).hexdigest() != ev.sha256:
            raise ValueError("feedback evidence hash mismatch")
        data = json.loads(raw)
        if (ev.metadata.get("kind") != "structured_training_feedback" or
                data.get("split") != "train" or not data.get("trace_id") or
                not data.get("task_id") or not data.get("run_id") or
                ev.derived_from != data.get("image_evidence_id")):
            raise ValueError("invalid training feedback provenance")
        image = store.get(ev.derived_from)
        if image.kind.value != "image" or hashlib.sha256(store.read_bytes(image.evidence_id)).hexdigest() != image.sha256:
            raise ValueError("image evidence hash mismatch")
        for row in data["zones"]:
            facts = row["ground_truth"]
            if facts.get("lamp") != "red":
                continue
            result.append(ReplayExample(data["task_id"], data["run_id"], ref, "panel",
                                        facts, row["expected_action"], "train"))
    return result
