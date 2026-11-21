"""Train/holdout separation and replay admission for panel experiments."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from .counterexample import (CounterexampleMemoryValidator, infer_red_readout_threshold,
                             panel_examples_from_feedback, panel_guidance)
from .evidence import EvidenceStore
from .harness import MemoryGate, TraceRecorder
from .memory import ExperienceStore
from .models import EvidenceKind, Experience, ToolReceipt
from .predicate import PredicateError
from .visual_bench import (PanelEnv, PanelTask, build_panel_task_v2, build_panel_task_v3, render_panel,
                           structured_training_feedback, verify)


def make_split(train_seeds: Sequence[int], test_seeds: Sequence[int], *, dataset_version="panel-v2"):
    if not train_seeds or not test_seeds or set(train_seeds) & set(test_seeds):
        raise ValueError("nonempty disjoint train and test seeds required")
    if len(set(train_seeds)) != len(train_seeds) or len(set(test_seeds)) != len(test_seeds):
        raise ValueError("duplicate seeds")
    builders = {"panel-v2": build_panel_task_v2, "panel-v3": build_panel_task_v3}
    if dataset_version not in builders:
        raise ValueError("unknown dataset version")
    builder = builders[dataset_version]
    train = [builder(s, f"train-{i:03d}") for i, s in enumerate(train_seeds)]
    test = [builder(s, f"test-{i:03d}") for i, s in enumerate(test_seeds)]
    # Content can collide even when seeds differ; no image may cross the boundary.
    import hashlib
    train_hashes = {hashlib.sha256(t.image).digest() for t in train}
    if any(hashlib.sha256(t.image).digest() in train_hashes for t in test):
        raise ValueError("identical image in train and test")
    return train, test


def make_training_probe(seed: int, task_id: str, readout: int, *, dataset_version="panel-v2") -> PanelTask:
    """Construct a training panel probing one readout, without returning its label."""
    if not task_id.startswith("train-") or isinstance(readout, bool) or not 0 <= readout <= 99:
        raise ValueError("valid training ID and readout required")
    builder = {"panel-v2": build_panel_task_v2, "panel-v3": build_panel_task_v3}.get(dataset_version)
    if builder is None:
        raise ValueError("unknown dataset version")
    bt = builder(seed, task_id)
    next(z for z in bt.zones if z.lamp == "red").readout = readout
    bt.image = render_panel(bt.zones)
    # These stay in the environment and are used only by the scorer.
    bt.expected_on = [z.name for z in bt.zones if z.is_target]
    bt.lockdown = [z.name for z in bt.zones if z.is_lockdown]
    bt.untouched = [z.name for z in bt.zones if not z.is_target]
    return bt


@dataclass
class PanelResult:
    outcome: object
    trace: object
    env: PanelEnv


def execute_panel(llm, db, bt: PanelTask, run_id: str, guidance: Sequence[str],
                  *, image_enabled: bool = True, action_max_tokens: int = 2400) -> PanelResult:
    """Run one task without exposing scorer state to the acting model."""
    env = PanelEnv(bt)
    recorder = TraceRecorder(db, run_id, bt.task, model=llm.model)
    system = ("Control the panel. Reply with one JSON object: "
              '{"thought":"...","tool":"...","args":{}}. '
              "Use finish when done. Never claim success.\nGOAL: " + bt.task.goal +
              "\nTOOLS: " + json.dumps(env.spec()) +
              ("\nGuidance from earlier training tasks:\n" + "\n".join(guidance) if guidance else ""))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Inspect the panel and act."}]
    parse_failures = 0
    failed_operations = {}
    for step in range(bt.task.max_steps):
        # Errors propagate; an incomplete task must not be reported as a model failure.
        resp = llm.complete(messages, purpose="act", task_id=bt.task.task_id,
                            run_id=run_id, max_tokens=action_max_tokens,
                            images=[bt.image] if image_enabled else None)
        try:
            text = resp.text.strip()
            action = json.loads(text[text.index("{"):text.rindex("}") + 1])
            if not isinstance(action, dict) or not isinstance(action.get("tool"), str):
                raise ValueError("invalid action")
        except (ValueError, json.JSONDecodeError):
            recorder.record("invalid model response", "__parse__", {}, None)
            parse_failures += 1
            if parse_failures >= 3:
                raise ValueError("action parse failed three times")
            messages.append({"role": "user", "content": "Reply with a JSON tool action."})
            continue
        parse_failures = 0
        tool, args = action["tool"], action.get("args") or {}
        if tool == "finish":
            recorder.record(str(action.get("thought", "")), tool, args, None)
            break
        result = env.call(tool, args)
        if not result.get("ok"):
            key = json.dumps([tool, args], sort_keys=True)
            failed_operations[key] = failed_operations.get(key, 0) + 1
            if failed_operations[key] >= 3:
                raise ValueError("the same tool operation failed three times")
        receipt = ToolReceipt(call_id=f"c{step}", tool=tool, ok=bool(result.get("ok")),
                              output=result.get("result"), error=result.get("error", ""))
        recorder.record(str(action.get("thought", "")), tool, args, receipt)
        messages.extend([{"role": "assistant", "content": json.dumps(action)},
                         {"role": "user", "content": json.dumps(result)}])
        messages = [messages[0]] + messages[-8:]
    outcome = verify(bt, env)
    return PanelResult(outcome, recorder.finish(outcome), env)


class PanelTrainer:
    """Owns training feedback; no API accepts holdout task objects."""
    def __init__(self, db, evidence: EvidenceStore, run_id: str, training_ids: set[str]):
        self.db, self.evidence, self.run_id = db, evidence, run_id
        self.training_ids = frozenset(training_ids)
        self.feedback_ids: list[str] = []
        self.gate = MemoryGate(ExperienceStore(db), evidence)

    def record(self, bt: PanelTask, result: PanelResult) -> str:
        if bt.task.task_id not in self.training_ids:
            raise ValueError("task is not in the training manifest")
        image = self.evidence.put(bt.image, EvidenceKind.IMAGE)
        ref = structured_training_feedback(bt, result.env, result.outcome,
                                           run_id=self.run_id, trace=result.trace,
                                           image_evidence_id=image.evidence_id,
                                           split="train", evidence_store=self.evidence)
        self.feedback_ids.append(ref)
        return ref

    def feedback_for_extraction(self) -> list[dict]:
        return [json.loads(self.evidence.read_text(ref)) for ref in self.feedback_ids]

    def guidance(self) -> list[str]:
        """Recheck every active rule against all training evidence before use."""
        if not self.feedback_ids:
            return []
        examples = panel_examples_from_feedback(self.evidence, self.feedback_ids)
        validator = CounterexampleMemoryValidator(self.evidence.exists)
        return [exp.lesson for exp in self.gate.active_experiences()
                if exp.family == "panel" and exp.recommended_action == "leave_off"
                and exp.lesson == panel_guidance(exp.condition)
                and validator.validate(exp, examples, "activate").decision == "accepted"]

    def induce_threshold(self) -> dict:
        """Persist bounded induction and send only a uniquely supported candidate to the gate."""
        examples = panel_examples_from_feedback(self.evidence, self.feedback_ids)
        induction = infer_red_readout_threshold(examples)
        audit = self.evidence.put_json({"kind": "bounded_threshold_induction",
                                        "feedback_ids": list(self.feedback_ids),
                                        "result": induction},
                                       metadata={"kind": "threshold_induction"})
        result = {"induction": induction, "induction_evidence_id": audit.evidence_id}
        if induction["decision"] == "candidate":
            existing = next((e for e in self.gate.active_experiences()
                             if e.family == "panel" and e.condition == induction["condition"]
                             and e.lesson in self.guidance()), None)
            if existing is not None:
                result.update({"experience_id": existing.experience_id, "admission": "already_active"})
                return result
            proposed = self.propose(induction["condition"])
            if proposed:
                exp, report, validation_ref = proposed
                result.update({"experience_id": exp.experience_id,
                               "admission": report.decision,
                               "validation_evidence_id": validation_ref})
        return result

    def propose(self, condition: dict):
        if not self.feedback_ids:
            raise ValueError("training feedback required")
        try:
            lesson = panel_guidance(condition)
        except (PredicateError, TypeError, ValueError):
            return None
        exp = Experience(lesson=lesson, family="panel", condition=condition,
                         recommended_action="leave_off", evidence_refs=[self.feedback_ids[-1]])
        self.gate.propose(exp)
        report, audit_id = self.gate.admit_panel_replayed(exp, self.feedback_ids)
        return exp, report, audit_id
