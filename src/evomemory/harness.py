"""The harness: everything the agent is *not* allowed to do for itself.

Responsibilities, and the boundary against the agent:

* the agent reasons and proposes actions        -> ``AgentLoop``
* the harness executes tools, records receipts  -> ``TraceRecorder`` + ``OpsEnv``
* the harness decides what memory is visible    -> ``ContextManager``
* the harness decides what memory may be written-> ``MemoryGate``
* the harness decides whether a run succeeded   -> ``bench.verify`` (program)

The agent never writes a status, never declares success, and never sees an
experience that has not passed admission.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .bench import BenchTask, OpsEnv, Task, task_env, verify
from .db import Database, dumps, loads
from .evidence import EvidenceStore
from .lifecycle import (
    ExperienceExtractor,
    ExperienceReconstructor,
    ExperienceValidator,
    MemoryCritic,
    ValidationReport,
)
from .llm import BudgetedLLM, BudgetExceeded, LLMError, estimate_tokens
from .memory import ExperienceStore, MemoryWriteError
from .models import (
    Actor,
    Critique,
    CritiqueVerdict,
    Evidence,
    EvidenceKind,
    ExecutionTrace,
    Experience,
    ExperienceStatus,
    MemoryEvent,
    MemoryMode,
    Outcome,
    OutcomeStatus,
    RetrievedExperience,
    Task,
    TraceStep,
    ToolReceipt,
    VerifiedBy,
    now_iso,
)
from .predicate import Tri, applicable, describe as describe_predicate
from .retrieval import ExperienceIndex


# --------------------------------------------------------------------------
# run configuration
# --------------------------------------------------------------------------


@dataclass
class RunConfig:
    mode: MemoryMode = MemoryMode.D_FULL_LIFECYCLE
    top_k: int = 3
    memory_token_budget: int = 300
    max_steps: int = 20
    #: Retire an experience after it has been injected into several failed tasks
    #: without ever being present for a success. Learned rules can go stale; this
    #: is the only mechanism that lets the store notice.
    evolves: bool = False
    evolve_min_failures: int = 2
    #: Correlation-based retirement (retire anything injected into failing tasks).
    #: Measured to also retire correct rules and leave the agent memoryless, so it
    #: is off by default; verifier-attributed strikes are the primary mechanism.
    evolve_correlation: bool = False

    @property
    def reads_memory(self) -> bool:
        return self.mode is not MemoryMode.A_NO_MEMORY

    @property
    def writes_memory(self) -> bool:
        return self.mode is not MemoryMode.A_NO_MEMORY

    @property
    def uses_critique(self) -> bool:
        """Always-on critique (mode D) — unconditioned, one call per candidate."""
        return self.mode is MemoryMode.D_FULL_LIFECYCLE

    @property
    def uses_selective_critique(self) -> bool:
        """Critique only when the structured condition does not decide the case."""
        return self.mode is MemoryMode.E_SELECTIVE

    @property
    def uses_reconstruction(self) -> bool:
        return self.mode in (MemoryMode.D_FULL_LIFECYCLE, MemoryMode.E_SELECTIVE)

    @property
    def validates(self) -> bool:
        return self.mode in (MemoryMode.D_FULL_LIFECYCLE, MemoryMode.G_STRUCTURED_GATE,
                             MemoryMode.E_SELECTIVE)

    @property
    def uses_structured(self) -> bool:
        return self.mode in (MemoryMode.S_STRUCTURED, MemoryMode.G_STRUCTURED_GATE,
                             MemoryMode.E_SELECTIVE)

    @property
    def uses_visual_evidence(self) -> bool:
        return self.mode in (MemoryMode.C_MULTIMODAL_MEMORY,
                             MemoryMode.D_FULL_LIFECYCLE)


# --------------------------------------------------------------------------
# trace recorder
# --------------------------------------------------------------------------


class TraceRecorder:
    def __init__(self, db: Database, run_id: str, task: Task, model: str = ""):
        self.db = db
        self.trace = ExecutionTrace(task_id=task.task_id, run_id=run_id, model=model)
        db.insert("traces", {
            "trace_id": self.trace.trace_id, "run_id": run_id,
            "task_id": task.task_id, "model": model, "params_json": dumps({}),
            "status": "running", "started_at": self.trace.started_at,
            "ended_at": None,
        })

    def record(self, thought: str, tool: str, args: Dict[str, Any],
               receipt: Optional[ToolReceipt]) -> TraceStep:
        step = TraceStep(index=len(self.trace.steps), thought=thought,
                         tool=tool, args=args, receipt=receipt)
        self.trace.add(step)
        self.db.insert("trace_steps", {
            "trace_id": self.trace.trace_id, "idx": step.index,
            "thought": thought, "tool": tool, "args_json": dumps(args),
            "receipt_json": dumps(receipt.to_dict()) if receipt else None,
        })
        return step

    def finish(self, outcome: Outcome) -> ExecutionTrace:
        self.trace.finish("completed")
        self.db.execute(
            "UPDATE traces SET status = ?, ended_at = ? WHERE trace_id = ?",
            ("completed", self.trace.ended_at, self.trace.trace_id))
        self.db.insert("outcomes", {
            "outcome_id": f"out_{self.trace.trace_id}",
            "trace_id": self.trace.trace_id,
            "status": outcome.status.value,
            "score": outcome.score,
            "reason": outcome.reason,
            "verified_by": outcome.verified_by.value,
            "checks_json": dumps(outcome.checks),
            "created_at": now_iso(),
        })
        return self.trace


# --------------------------------------------------------------------------
# memory gate
# --------------------------------------------------------------------------


class GateError(PermissionError):
    pass


class MemoryGate:
    """The only path into and out of the experience store.

    Read policy  : only ``ACTIVE`` experiences, and only the latest version.
    Write policy : the extractor may create candidates; only the validator may
                   accept; only the gate may activate. A model-originated write
                   that tries to skip a step raises.
    """

    def __init__(self, store: ExperienceStore, evidence: EvidenceStore,
                 validator_factory: Any = None):
        self.store = store
        self.evidence = evidence
        self._validator_factory = validator_factory

    # -- write -----------------------------------------------------------
    def propose(self, exp: Experience, actor: Actor = Actor.MODEL) -> Experience:
        if actor is Actor.MODEL:
            # A model may only ever create candidates; downgrade silently is worse
            # than refusing, so refuse anything that claims to be further along.
            if exp.status is not ExperienceStatus.CANDIDATE:
                raise GateError("model-originated memory must start as a candidate")
        return self.store.add(exp, actor=Actor.EXTRACTOR)

    def admit(self, exp: Experience, *, trace: Optional[ExecutionTrace] = None,
              outcome: Optional[Outcome] = None,
              existing_lessons: Optional[Sequence[str]] = None,
              leak_guard: Optional[Sequence[str]] = None,
              model_veto: Optional[Tuple[bool, str]] = None) -> Tuple[bool, ValidationReport]:
        """Validate then accept then activate. Returns (admitted, report)."""
        validator = ExperienceValidator(
            self.evidence,
            existing_texts=existing_lessons or [],
            leak_guard=leak_guard or [],
        )
        self.store.submit_for_validation(exp.experience_id, exp.version)
        report = validator.validate(exp, trace=trace, outcome=outcome,
                                    model_veto=model_veto)
        if not report.passed:
            self.store.reject(exp.experience_id, exp.version,
                              reason=report.summary())
            return False, report
        self.store.accept(exp.experience_id, exp.version,
                          reason=f"programmatic checks passed ({len(report.checks)})")
        self.store.activate(exp.experience_id, exp.version,
                            reason="admission gate: active")
        return True, report

    def accept_unvalidated(self, exp: Experience) -> Experience:
        """The naive-memory path (modes B and C).

        It is deliberately lax: candidate -> accepted -> active with no checks
        beyond a non-empty lesson. This is the baseline the validation gate has
        to beat, not a shortcut for the main system.
        """
        self.store.submit_for_validation(exp.experience_id, exp.version)
        self.store.accept(exp.experience_id, exp.version,
                          reason="naive memory: no programmatic validation")
        self.store.activate(exp.experience_id, exp.version, reason="naive admission")
        return exp

    # -- read ------------------------------------------------------------
    def active_experiences(self) -> List[Experience]:
        return self.store.active()

    def deprecate(self, exp: Experience, reason: str) -> None:
        self.store.deprecate(exp.experience_id, exp.version, reason=reason)


# --------------------------------------------------------------------------
# context manager
# --------------------------------------------------------------------------


@dataclass
class MemoryBlock:
    guidance: List[str] = field(default_factory=list)
    injected_ids: List[str] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    tokens: int = 0

    def render(self) -> str:
        if not self.guidance:
            return ""
        lines = ["Reusable guidance from earlier tasks:"]
        lines.extend(f"- {g}" for g in self.guidance)
        return "\n".join(lines)


class ContextManager:
    """Chooses what memory enters the prompt, under an explicit token budget."""

    def __init__(self, token_budget: int = 300):
        self.token_budget = token_budget

    def build(self, retrieved: Sequence[RetrievedExperience]) -> MemoryBlock:
        block = MemoryBlock()
        for item in retrieved:
            text = item.guidance if item.guidance is not None else item.experience.lesson
            if not text:
                block.rejected.append({
                    "experience_id": item.experience.experience_id,
                    "reason": "no guidance after reconstruction"})
                continue
            cost = estimate_tokens(text)
            if block.tokens + cost > self.token_budget:
                block.rejected.append({
                    "experience_id": item.experience.experience_id,
                    "reason": f"token budget: {block.tokens}+{cost}>{self.token_budget}"})
                continue
            block.guidance.append(text)
            block.injected_ids.append(item.experience.experience_id)
            block.tokens += cost
        return block


# --------------------------------------------------------------------------
# agent loop
# --------------------------------------------------------------------------


ACTION_SYSTEM = (
    "You are an operations agent. Reply with exactly one JSON object:\n"
    '{"thought": "<brief>", "tool": "<tool name>", "args": {...}}\n'
    "Available tools are listed in the task. Use \"finish\" when done.\n"
    "Never claim success: the environment decides.\n"
    "Keep reasoning short — a long chain of thought is truncated before your answer.\n"
)

#: Reasoning models spend completion tokens on hidden thinking before they emit
#: anything, and that cost grows with the conversation. Two defences: never send
#: more than this many messages, and give the call enough headroom that a
#: reasonable amount of thinking still leaves room for the answer.
MAX_HISTORY_MESSAGES = 9
ACTION_MAX_TOKENS = 1200


def _trim_history(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    return [messages[0]] + messages[-(MAX_HISTORY_MESSAGES - 1):]


@dataclass
class TaskRunResult:
    task_id: str
    mode: str
    trace_id: str
    outcome: Outcome
    injected_ids: List[str] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    steps: int = 0
    memory_written: int = 0
    memory_admitted: int = 0
    memory_candidate_ids: List[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    reused_from_memory: bool = False
    rule_checks: int = 0
    executable_rules: int = 0
    deprecated: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id, "mode": self.mode, "trace_id": self.trace_id,
            "outcome": self.outcome.to_dict(),
            "injected_ids": self.injected_ids, "rejected": self.rejected,
            "retrieved": self.retrieved, "tool_calls": self.tool_calls,
            "steps": self.steps, "memory_written": self.memory_written,
            "memory_admitted": self.memory_admitted,
            "memory_candidate_ids": self.memory_candidate_ids,
            "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
            "reused_from_memory": self.reused_from_memory,
            "rule_checks": self.rule_checks,
            "executable_rules": self.executable_rules,
            "deprecated": self.deprecated,
        }


class Harness:
    def __init__(self, db: Database, evidence: EvidenceStore, llm: BudgetedLLM,
                 gate: MemoryGate, extractor: ExperienceExtractor,
                 critic: MemoryCritic, reconstructor: ExperienceReconstructor,
                 config: RunConfig):
        self.db = db
        self.evidence = evidence
        self.llm = llm
        self.gate = gate
        self.extractor = extractor
        self.critic = critic
        self.reconstructor = reconstructor
        self.config = config
        self.context = ContextManager(config.memory_token_budget)
        self._index: Optional[ExperienceIndex] = None
        self._index_size = -1

    # -- memory access ---------------------------------------------------
    def _experience_index(self) -> ExperienceIndex:
        live = self.gate.active_experiences()
        if self._index is None or self._index_size != len(live):
            captions = self._caption_map(live) if self.config.uses_visual_evidence else None
            self._index = ExperienceIndex(live, caption_map=captions)
            self._index_size = len(live)
        return self._index

    def _caption_map(self, experiences: Sequence[Experience]) -> Dict[str, str]:
        """Join each experience to the captions of the images it cites."""
        captions: Dict[str, str] = {}
        for exp in experiences:
            texts: List[str] = []
            for ref in exp.evidence_refs:
                try:
                    ev = self.evidence.get(ref)
                except KeyError:
                    continue
                if ev.kind in (EvidenceKind.IMAGE, EvidenceKind.THUMBNAIL):
                    caption = ev.metadata.get("caption", "")
                    if caption:
                        texts.append(caption)
            if texts:
                captions[exp.experience_id] = " ".join(texts)
        return captions

    def retrieve(self, task: Task, state_text: str = "") -> Tuple[List[RetrievedExperience], MemoryBlock]:
        if not self.config.reads_memory:
            return [], MemoryBlock()
        index = self._experience_index()
        # Scope retrieval to the task's family. Without this, a family whose goal
        # text is lexically similar to another's (escalation vs a conditional
        # escalation) never sees its own memories at all: the other family's
        # lessons fill the top-k, and the critic then rejects every one of them as
        # a family mismatch, leaving the task with no memory and no diagnostics.
        # That is exactly what happened in the first 18-task live run.
        hits = index.search(task.goal, limit=self.config.top_k, family=task.family)
        items: List[RetrievedExperience] = []
        for exp, score, channels in hits:
            item = RetrievedExperience(experience=exp, score=score, channels=channels)
            if self.config.uses_critique:
                item.critique = self.critic.critique(exp, task, state_text)
                if item.critique.verdict is CritiqueVerdict.REJECT:
                    items.append(item)
                    continue
            if self.config.uses_reconstruction:
                item.guidance = self.reconstructor.reconstruct(
                    exp, task, item.critique or Critique(CritiqueVerdict.APPLY))
            else:
                item.guidance = exp.lesson
            items.append(item)

        usable = [i for i in items
                  if i.critique is None or i.critique.verdict is not CritiqueVerdict.REJECT]
        block = self.context.build(usable)
        rejected = [{"experience_id": i.experience.experience_id,
                     "reason": i.critique.reasons[0] if i.critique and i.critique.reasons
                     else "rejected by critique"}
                    for i in items
                    if i.critique is not None
                    and i.critique.verdict is CritiqueVerdict.REJECT]
        block.rejected.extend(rejected)
        return items, block

    # -- executable memory -------------------------------------------------
    def _rule_set(self, task: Task) -> List[Tuple[str, Dict[str, Any]]]:
        """Active experiences of this family that carry an executable condition."""
        if not self.config.uses_structured:
            return []
        out: List[Tuple[str, Dict[str, Any]]] = []
        for exp in self.gate.active_experiences():
            if exp.condition is None:
                continue
            if exp.family and task.family and exp.family != task.family:
                continue
            out.append((exp.experience_id, exp.condition))
        return out

    def check_rule(self, rid: Optional[str], rules: Sequence[Tuple[str, Dict[str, Any]]],
                   env: OpsEnv, rids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Evaluate every stored condition against one or more records.

        This is the whole point of the structured representation: the rule is
        *executed*, not re-read by a model. The agent still decides whether to
        consult it and what to do with the answer.

        Batch form is the default on purpose. A one-record-per-call tool cost one
        step per record, and with a step budget sized for the work itself the
        agent ran out of steps before writing anything — measured in the first
        structured live run, where every rule verdict was correct and no record
        was updated.
        """
        targets = list(rids) if rids is not None else ([rid] if rid else [])
        if not targets:
            return {"ok": False, "error": "provide 'rids' (list) or 'rid' (str)"}
        if len(targets) > 64:
            return {"ok": False, "error": "at most 64 records per call"}

        results = [self._evaluate_one(r, rules, env) for r in targets]
        payload: Dict[str, Any] = {"results": results}
        if len(results) == 1:
            payload.update({"rid": results[0]["rid"], "exempt": results[0]["exempt"]})
        return {"ok": True, "result": payload}

    def _evaluate_one(self, rid: str, rules: Sequence[Tuple[str, Dict[str, Any]]],
                      env: OpsEnv) -> Dict[str, Any]:
        fetched = env.call("get_record", {"rid": rid})
        if not fetched.get("ok"):
            return {"rid": rid, "exempt": None, "error": fetched.get("error", "unknown record"),
                    "rules": []}
        record = fetched["result"]

        verdicts = []
        any_true = any_unknown = False
        for exp_id, condition in rules:
            result = applicable(condition, record)
            if result.verdict is Tri.TRUE:
                any_true = True
            elif result.verdict is Tri.UNKNOWN:
                any_unknown = True
            verdicts.append({
                "experience_id": exp_id,
                "verdict": result.verdict.value,
                "condition": describe_predicate(condition),
                "reason": result.reason,
            })
        exempt = True if any_true else (None if any_unknown else False)
        return {"rid": rid, "exempt": exempt,
                "note": ("exempt = the learned rules say leave this record alone"
                         if exempt is True else
                         "undetermined = a required field was missing or mistyped"
                         if exempt is None else
                         "no learned rule exempts this record"),
                "rules": verdicts}

    def _tool_specs(self, env: OpsEnv, rules: Sequence[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        spec = list(env.spec())
        if rules:
            spec.append({
                "name": "check_rule",
                "args": {"rids": "list[str] — check several records in one call"},
                "returns": ("per record: exempt=true means leave it alone, "
                            "exempt=null means the rule could not be decided "
                            "(do not treat null as false)"),
            })
        return spec

    # -- one task --------------------------------------------------------
    def run_task(self, bt: BenchTask, run_id: str) -> TaskRunResult:
        task = bt.task
        env = task_env(bt)
        if self.config.evolves and self.config.evolve_correlation:
            self._retire_stale(task.task_id)
        retrieved, block = self.retrieve(task, state_text=json.dumps(bt.task.context))

        if self.config.reads_memory:
            self.db.insert("memory_events", {
                "run_id": run_id, "task_id": task.task_id, "trace_id": "",
                "step": 0, "query": task.goal,
                "retrieved_json": dumps([
                    {"experience_id": i.experience.experience_id,
                     "score": round(i.score, 6),
                     "channels": {k: round(v, 6) for k, v in i.channels.items()},
                     "verdict": i.critique.verdict.value if i.critique else None}
                    for i in retrieved]),
                "injected_json": dumps(block.injected_ids),
                "rejected_json": dumps(block.rejected),
                "created_at": now_iso(),
            })

        rules = self._rule_set(task)
        recorder = TraceRecorder(self.db, run_id, task, model=self.llm.model)
        messages = self._initial_messages(bt, env, block, rules)
        tokens_in = tokens_out = 0
        finished = False
        rule_checks = 0

        step_budget = min(task.max_steps, self.config.max_steps)
        if self.config.uses_structured and rules:
            step_budget += 4          # one batch evaluation plus slack
        for step in range(step_budget):
            try:
                resp = self.llm.complete(_trim_history(messages), purpose="act",
                                         task_id=task.task_id, run_id=run_id,
                                         prompt_version="action-v2",
                                         max_tokens=ACTION_MAX_TOKENS)
            except (LLMError, BudgetExceeded) as exc:
                recorder.record(f"llm error: {exc}", "", {}, None)
                break
            tokens_in += resp.input_tokens
            tokens_out += resp.output_tokens

            action = _parse_action(resp.text)
            if action is None:
                receipt = ToolReceipt(call_id=f"c{step}", tool="__parse__", ok=False,
                                      error="could not parse an action from the reply")
                recorder.record(resp.text[:200], "__parse__", {}, receipt)
                messages.append({"role": "assistant", "content": resp.text[:500]})
                messages.append({"role": "user", "content": json.dumps(
                    {"ok": False, "error": receipt.error,
                     "hint": 'reply with {"tool": ..., "args": {...}}'},
                    ensure_ascii=False)})
                continue

            thought = str(action.get("thought", ""))[:300]
            tool = str(action.get("tool", ""))
            args = action.get("args") or {}
            if tool == "finish":
                recorder.record(thought, "finish", args, None)
                finished = True
                break

            if tool == "check_rule" and rules:
                rids = args.get("rids")
                if not isinstance(rids, list):
                    rids = None
                result = self.check_rule(args.get("rid"), rules, env, rids=rids)
                rule_checks += 1
            else:
                result = env.call(tool, args)
            receipt = ToolReceipt(
                call_id=f"c{step}", tool=tool, ok=bool(result.get("ok")),
                output=result.get("result"), error=result.get("error", ""))
            recorder.record(thought, tool, args, receipt)
            messages.append({"role": "assistant", "content": json.dumps(action,
                                                                        ensure_ascii=False)})
            messages.append({"role": "user", "content": json.dumps(result,
                                                                  ensure_ascii=False)[:2000]})

        outcome = verify(bt, env)
        trace = recorder.finish(outcome)

        result = TaskRunResult(
            task_id=task.task_id, mode=self.config.mode.value,
            trace_id=trace.trace_id, outcome=outcome,
            injected_ids=list(block.injected_ids), rejected=list(block.rejected),
            retrieved=[{"experience_id": i.experience.experience_id,
                        "score": round(i.score, 6)} for i in retrieved],
            tool_calls=len(trace.tool_calls()), steps=len(trace.steps),
            tokens_in=tokens_in, tokens_out=tokens_out,
            reused_from_memory=bool(block.injected_ids),
            rule_checks=rule_checks,
            executable_rules=len(rules),
        )

        if self.config.writes_memory:
            self._record_usage(task.task_id, outcome, block.injected_ids)
            if self.config.evolves:
                # Attribution from the verifier's own complaint, not from
                # correlation: a rule earns a strike only when the record it
                # exempted turns out to be a record that should have been acted on.
                strikes = self._attribute_failure(bt, env, outcome)
                result.deprecated = self._retire_on_strikes(task.task_id) + strikes
            self._write_memory(bt, trace, outcome, result, run_id)
        return result

    # -- experience evolution -----------------------------------------------
    def _record_usage(self, task_id: str, outcome: Outcome,
                      injected: Sequence[str]) -> None:
        """Log which experiences were in play for this task's outcome.

        Attribution is deliberately weak and stated as such: an experience being
        injected does not mean it caused the outcome. The retirement rule below is
        built to survive that weakness by requiring repeated, consistent failure
        with never a success.
        """
        for exp_id in injected:
            exp = self.gate.store.get(exp_id)
            if exp is None:
                continue
            self.db.insert("experience_usage", {
                "experience_id": exp_id, "version": exp.version,
                "task_id": task_id, "outcome": outcome.status.value,
                "created_at": now_iso(),
            })

    def _attribute_failure(self, bt: BenchTask, env: OpsEnv,
                           outcome: Outcome) -> List[str]:
        """Turn "you missed r02" into a strike against whichever rule exempted r02.

        This is the only causal signal available without ground truth: the verifier
        names the records that were not actioned, and each active rule either
        exempted one of them or did not. A rule that exempted a record the grader
        expected to be actioned has demonstrably done harm.
        """
        missed: List[str] = []
        for check in outcome.checks:
            if check.get("name") == "targets_actioned" and not check.get("passed"):
                found = re.search(r"missing \[([^\]]*)\]", str(check.get("detail", "")))
                if found:
                    missed = [m.strip().strip("'\"") for m in found.group(1).split(",")
                              if m.strip()]
        if not missed:
            return []

        rules = self._rule_set(bt.task)
        if not rules:
            return []
        struck: List[str] = []
        for rid in missed:
            fetched = env.call("get_record", {"rid": rid})
            if not fetched.get("ok"):
                continue
            record = fetched["result"]
            for exp_id, condition in rules:
                if applicable(condition, record).verdict is not Tri.TRUE:
                    continue
                self.db.insert("experience_strikes", {
                    "experience_id": exp_id, "task_id": bt.task.task_id,
                    "rid": rid, "reason": "exempted a record that had to be actioned",
                    "created_at": now_iso()})
                if exp_id not in struck:
                    struck.append(exp_id)
        return struck

    def _retire_on_strikes(self, task_id: str) -> List[str]:
        """Retire rules that have accrued enough direct strikes."""
        threshold = max(1, self.config.evolve_min_failures)
        retired: List[str] = []
        for row in self.db.query(
                "SELECT experience_id, COUNT(*) AS n FROM experience_strikes "
                "GROUP BY experience_id HAVING n >= ?", (threshold,)):
            exp = self.gate.store.get(row["experience_id"])
            if exp is None or exp.status is not ExperienceStatus.ACTIVE:
                continue
            self.gate.store.deprecate(
                exp.experience_id, exp.version,
                reason=(f"retired after {int(row['n'])} verifier-attributed strikes "
                        f"(last: {task_id})"),
                actor=Actor.EVOLUTION)
            retired.append(f"{exp.experience_id}@v{exp.version}")
        return retired

    def _retire_stale(self, task_id: str) -> List[str]:
        """Retire memories whose post-success record has gone bad, before they act again.

        Retiring *after* a failing task is one task too late: the stale rule still
        exempted a legitimate record during the very task that failed. This runs
        before retrieval so a memory that has already failed twice since it last
        helped simply stops being offered.

        "Since it last helped" rather than "consecutively" matters because a stale
        rule does not appear in every task — counting only adjacent failures lets
        it survive indefinitely by interleaving.
        """
        retired: List[str] = []
        for exp in self.gate.active_experiences():
            rows = self.db.query(
                "SELECT outcome FROM experience_usage WHERE experience_id = ? "
                "ORDER BY id DESC", (exp.experience_id,))
            failures = 0
            for row in rows:
                if row["outcome"] == "success":
                    break
                failures += 1
            if failures >= max(1, self.config.evolve_min_failures):
                self.gate.store.deprecate(
                    exp.experience_id, exp.version,
                    reason=(f"retired before use: {failures} failed tasks since it "
                            f"last accompanied a success (last: {task_id})"),
                    actor=Actor.EVOLUTION)
                retired.append(f"{exp.experience_id}@v{exp.version}")
        return retired

    def _evolve(self, task_id: str, injected: Sequence[str]) -> List[str]:
        """Retire an experience whose *recent* usages are all failures.

        The first version required a lifetime record of zero successes, which made
        retirement impossible in exactly the case that matters: a rule that was
        right for a while and then went stale. Counting consecutive failures
        instead treats "the world changed" as the event it is, while still
        requiring more than one bad task before acting.
        """
        if not injected:
            return []
        retired: List[str] = []
        window = max(1, self.config.evolve_min_failures)
        for exp_id in dict.fromkeys(injected):
            recent = self.db.query(
                "SELECT outcome FROM experience_usage WHERE experience_id = ? "
                "ORDER BY id DESC LIMIT ?", (exp_id, window))
            if len(recent) < window:
                continue
            if any(row["outcome"] != "failure" for row in recent):
                continue
            exp = self.gate.store.get(exp_id)
            if exp is None or exp.status is not ExperienceStatus.ACTIVE:
                continue
            self.gate.store.deprecate(
                exp_id, exp.version,
                reason=(f"retired after {window} consecutive failed tasks while "
                        f"injected (last: {task_id})"),
                actor=Actor.EVOLUTION)
            retired.append(f"{exp_id}@v{exp.version}")
        return retired

    # -- memory write-back -----------------------------------------------
    def _write_memory(self, bt: BenchTask, trace: ExecutionTrace, outcome: Outcome,
                      result: TaskRunResult, run_id: str) -> None:
        extraction = self.extractor.extract(bt.task, trace, outcome, outcome.checks)
        existing = [e.lesson for e in self.gate.active_experiences()]
        for exp in extraction.experiences:
            try:
                self.gate.propose(exp, actor=Actor.EXTRACTOR)
            except MemoryWriteError as exc:
                result.rejected.append({"candidate": exp.lesson[:60],
                                        "reason": f"write refused: {exc}"})
                continue
            result.memory_written += 1
            result.memory_candidate_ids.append(
                f"{exp.experience_id}@v{exp.version}")

            if self.config.validates:
                admitted, report = self.gate.admit(
                    exp, trace=trace, outcome=outcome,
                    existing_lessons=existing,
                )
                if admitted:
                    result.memory_admitted += 1
                else:
                    result.rejected.append({
                        "candidate": exp.experience_id,
                        "reason": f"validation failed: {report.summary()}"})
            else:
                self.gate.accept_unvalidated(exp)
                result.memory_admitted += 1

    # -- prompt assembly --------------------------------------------------
    def _initial_messages(self, bt: BenchTask, env: OpsEnv, block: MemoryBlock,
                          rules: Sequence[Tuple[str, Dict[str, Any]]] = ()) -> List[Dict[str, str]]:
        task = bt.task
        payload: Dict[str, Any] = {
            "task_id": task.task_id,
            "goal": task.goal,
            "context": task.context,
            "tools": self._tool_specs(env, rules),
        }
        system = ACTION_SYSTEM + "\nTASK:\n" + json.dumps(payload, ensure_ascii=False)
        if block.guidance:
            system += "\n\n" + block.render()
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "Begin. Reply with one JSON action."},
        ]


def _parse_action(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict) or "tool" not in value:
        return None
    return value


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------


@dataclass
class ReplayReport:
    trace_id: str
    task_id: str
    steps_replayed: int
    recorded_status: str
    replayed_status: str
    matches: bool
    diffs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id, "task_id": self.task_id,
            "steps_replayed": self.steps_replayed,
            "recorded_status": self.recorded_status,
            "replayed_status": self.replayed_status,
            "matches": self.matches, "diffs": self.diffs,
        }


class ReplayRunner:
    """Re-executes a recorded trace's tool calls against a fresh environment.

    What this proves: the recorded action sequence deterministically produces the
    recorded environment outcome. What it does **not** prove: that the model would
    make the same decisions again. Live-model replay is not reproducible by
    construction, and the report says so rather than implying otherwise.
    """

    LIMITATION = ("tool-level replay only; model decisions are not re-sampled, "
                  "so this validates determinism of the harness, not of the model")

    def __init__(self, db: Database):
        self.db = db

    def replay(self, trace_id: str, bt: BenchTask) -> ReplayReport:
        rows = self.db.query(
            "SELECT idx, tool, args_json, receipt_json FROM trace_steps "
            "WHERE trace_id = ? ORDER BY idx", (trace_id,))
        if not rows:
            raise KeyError(f"no steps recorded for trace {trace_id}")

        env = task_env(bt)
        diffs: List[str] = []
        replayed = 0
        for row in rows:
            tool = row["tool"]
            if tool in ("", "finish", "__parse__"):
                continue
            args = loads(row["args_json"], {})
            if tool == "check_rule" and rules:
                rids = args.get("rids")
                if not isinstance(rids, list):
                    rids = None
                result = self.check_rule(args.get("rid"), rules, env, rids=rids)
                rule_checks += 1
            else:
                result = env.call(tool, args)
            replayed += 1
            recorded = loads(row["receipt_json"], {}) or {}
            if bool(recorded.get("ok")) != bool(result.get("ok")):
                diffs.append(f"step {row['idx']}: ok {recorded.get('ok')} != {result.get('ok')}")
            if not result.get("ok"):
                diffs.append(f"step {row['idx']}: replay error: {result.get('error')}")

        outcome = verify(bt, env)
        recorded_status = self._recorded_status(trace_id)
        return ReplayReport(
            trace_id=trace_id, task_id=bt.task.task_id, steps_replayed=replayed,
            recorded_status=recorded_status, replayed_status=outcome.status.value,
            matches=(recorded_status == outcome.status.value and not diffs),
            diffs=diffs,
        )

    def _recorded_status(self, trace_id: str) -> str:
        row = self.db.query_one(
            "SELECT status FROM outcomes WHERE trace_id = ?", (trace_id,))
        return row["status"] if row else "unknown"

    def replay_task(self, run_id: str, task_id: str, bt: BenchTask) -> ReplayReport:
        row = self.db.query_one(
            "SELECT trace_id FROM traces WHERE run_id = ? AND task_id = ? "
            "AND status = 'completed' ORDER BY started_at DESC LIMIT 1",
            (run_id, task_id))
        if row is None:
            raise KeyError(f"no completed trace for {task_id} in run {run_id}")
        return self.replay(row["trace_id"], bt)


# --------------------------------------------------------------------------
# recovery
# --------------------------------------------------------------------------


class Recovery:
    """Resume a run without redoing settled work.

    Progress is derived from the database, not from a separate checkpoint file —
    there is one source of truth, so they cannot disagree. A task counts as done
    only if its trace finished *and* an outcome was written.
    """

    def __init__(self, db: Database):
        self.db = db

    def completed_task_ids(self, run_id: str) -> List[str]:
        rows = self.db.query(
            "SELECT t.task_id AS task_id FROM traces t "
            "JOIN outcomes o ON o.trace_id = t.trace_id "
            "WHERE t.run_id = ? AND t.status = 'completed'", (run_id,))
        return sorted({r["task_id"] for r in rows})

    def pending_tasks(self, run_id: str, tasks: Sequence[BenchTask]) -> List[BenchTask]:
        done = set(self.completed_task_ids(run_id))
        return [bt for bt in tasks if bt.task.task_id not in done]

    def resume(self, harness: "Harness", run_id: str,
               tasks: Sequence[BenchTask]) -> Dict[str, Any]:
        pending = self.pending_tasks(run_id, tasks)
        skipped = len(tasks) - len(pending)
        results = [harness.run_task(bt, run_id) for bt in pending]
        return {"resumed": True, "skipped": skipped, "ran": len(results),
                "results": results}
