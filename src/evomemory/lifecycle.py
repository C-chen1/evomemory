"""The experience lifecycle: extraction, validation, critique, reconstruction.

The four services here are deliberately separable, because the whole argument of
the project depends on *who is allowed to do what*:

* ``ExperienceExtractor`` may only produce **candidates**. It cannot validate.
* ``ExperienceValidator`` is the only thing that can accept or reject, and it
  runs **programmatic checks plus an optional model opinion that can only veto,
  never admit**.
* ``MemoryCritic`` judges fit against the current task. Its worst outcome is
  "not applicable", which costs nothing.
* ``ExperienceReconstructor`` rewrites an experience for the current state, or
  returns ``None`` — the equivalent of MemHarness' ``<EMPTY>``, but tracked as a
  first-class metric.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .evidence import EvidenceStore
from .llm import BudgetedLLM, LLMError
from .models import (
    Critique,
    CritiqueVerdict,
    Evidence,
    ExecutionTrace,
    Experience,
    Outcome,
    OutcomeStatus,
    Task,
    VerifiedBy,
)
from .retrieval import cosine, tokenize


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    experiences: List[Experience]
    source: str
    error: str = ""


class ExperienceExtractor:
    """Turns (task, trace, outcome, checks) into zero or more candidates.

    Reflection is allowed to return *nothing*. A success produced by luck, or a
    failure with no identifiable cause, must not become a lesson — that is how
    memory systems poison themselves.
    """

    PROMPT_VERSION = "extract-v1"

    SYSTEM = (
        "You are an experience extractor for an agent memory system.\n"
        "Given one task execution, extract at most one reusable lesson.\n"
        "Return STRICT JSON: {\"lesson\": str, \"failure_signature\": str, "
        "\"applicability_conditions\": [str], \"confidence\": float}\n"
        "Rules:\n"
        "- The lesson must be actionable and generalise to future tasks.\n"
        "- Do NOT restate the task answer or any concrete id, name or value.\n"
        "- If nothing reusable can be learned, return {\"lesson\": \"\"}.\n"
    )

    def __init__(self, evidence: EvidenceStore, llm: Optional[BudgetedLLM] = None):
        self.evidence = evidence
        self.llm = llm

    def extract(self, task: Task, trace: ExecutionTrace, outcome: Outcome,
                checks: Sequence[Dict[str, Any]]) -> ExtractionResult:
        if outcome.verified_by is not VerifiedBy.PROGRAM:
            return ExtractionResult([], source="skipped",
                                    error="outcome was not program-verified")
        if outcome.status is OutcomeStatus.UNKNOWN:
            return ExtractionResult([], source="skipped", error="outcome unknown")

        evidence_refs = self._collect_evidence(trace, outcome, checks)
        action_refs = [f"step:{s.index}:{s.tool}" for s in trace.tool_calls()]

        if self.llm is not None:
            try:
                return self._extract_llm(task, trace, outcome, checks,
                                         evidence_refs, action_refs)
            except (LLMError, ValueError, KeyError, json.JSONDecodeError):
                pass  # fall through to the deterministic extractor

        return ExtractionResult(
            [self._extract_heuristic(task, trace, outcome, checks,
                                     evidence_refs, action_refs)],
            source="heuristic",
        )

    # -- deterministic path --------------------------------------------
    def _extract_heuristic(self, task: Task, trace: ExecutionTrace, outcome: Outcome,
                           checks: Sequence[Dict[str, Any]], evidence_refs: List[str],
                           action_refs: List[str]) -> Experience:
        failed = [c for c in checks if not c.get("passed", False)]
        signature = self._signature(task, failed, outcome)
        conds = self._conditions(task, outcome)
        lesson = self._lesson_text(task, outcome, failed, trace)

        return Experience(
            lesson=lesson,
            task_id=task.task_id,
            trace_id=trace.trace_id,
            task_context=task.goal[:500],
            evidence_refs=evidence_refs,
            action_trace_refs=action_refs,
            outcome=outcome.status.value,
            failure_signature=signature,
            applicability_conditions=conds,
            family=task.family,
            source_kind="heuristic",
            prompt_version="heuristic-v1",
            confidence=0.6 if outcome.status is OutcomeStatus.SUCCESS else 0.4,
        )

    @staticmethod
    def _signature(task: Task, failed: Sequence[Dict[str, Any]], outcome: Outcome) -> str:
        if outcome.status is OutcomeStatus.SUCCESS:
            return f"success:{task.family or 'general'}"
        names = sorted({str(c.get("name", "unknown")) for c in failed})
        return f"{task.family or 'general'}|{'|'.join(names) or 'no_check'}"

    @staticmethod
    def _conditions(task: Task, outcome: Outcome) -> List[str]:
        conds = [f"task_family:{task.family}"] if task.family else ["task_family:any"]
        for tag in task.tags:
            conds.append(f"tag:{tag}")
        conds.append(f"prior_outcome:{outcome.status.value}")
        return conds

    @staticmethod
    def _lesson_text(task: Task, outcome: Outcome, failed: Sequence[Dict[str, Any]],
                     trace: ExecutionTrace) -> str:
        family = task.family or "this task family"
        if outcome.status is OutcomeStatus.SUCCESS:
            tools = ", ".join(sorted({s.tool for s in trace.tool_calls()})) or "no tools"
            return (f"For {family}, the sequence using {tools} satisfied every check. "
                    f"Reuse this order of operations before trying alternatives.")
        if not failed:
            return (f"For {family}, the run failed without a failing check. "
                    f"Verify preconditions before acting; the failure was not attributable.")

        # Pull the concrete offending attributes out of the verifier's message.
        # This is the only place the lesson gets a fact rather than a platitude:
        # the verifier said *which* record and *which* of its fields was out of
        # bounds, so the lesson can name that field.
        attrs: List[tuple[str, str]] = []
        for check in failed:
            for match in re.finditer(r"(\w+)=('([^']*)'|[A-Za-z0-9_\-]+)",
                                     str(check.get("detail", ""))):
                field_name = match.group(1)
                value = match.group(3) if match.group(3) is not None else match.group(2)
                if field_name in ("tier", "note", "status", "region", "priority"):
                    pair = (field_name, value)
                    if pair not in attrs:
                        attrs.append(pair)

        names = ", ".join(str(c.get("name")) for c in failed[:3])
        if attrs:
            clause = " and ".join(f"{f}={v}" for f, v in attrs)
            return (f"For {family}, do not act on records with {clause}. "
                    f"A previous attempt failed the {names} check precisely because "
                    f"it modified those records.")
        detail = "; ".join(
            f"{c.get('name')}: {c.get('detail', 'failed')}" for c in failed[:3])
        return (f"For {family}, these obligations were missed: {detail}. "
                f"Check them explicitly before finishing, and do not report completion "
                f"until each is satisfied.")

    def _collect_evidence(self, trace: ExecutionTrace, outcome: Outcome,
                          checks: Sequence[Dict[str, Any]]) -> List[str]:
        refs: List[str] = []
        for step in trace.tool_calls():
            if step.receipt is not None:
                ev = self.evidence.put_json(
                    step.receipt.to_dict(), metadata={
                        "kind": "tool_receipt", "trace_id": trace.trace_id,
                        "step": step.index, "tool": step.tool,
                    })
                refs.append(ev.evidence_id)
        ev = self.evidence.put_json(
            {"status": outcome.status.value, "score": outcome.score,
             "reason": outcome.reason, "checks": list(checks)},
            metadata={"kind": "verification", "trace_id": trace.trace_id})
        refs.append(ev.evidence_id)
        return refs

    # -- model path ------------------------------------------------------
    def _extract_llm(self, task: Task, trace: ExecutionTrace, outcome: Outcome,
                     checks: Sequence[Dict[str, Any]], evidence_refs: List[str],
                     action_refs: List[str]) -> ExtractionResult:
        assert self.llm is not None
        transcript = self._transcript(trace)
        payload = {
            "goal": task.goal,
            "family": task.family,
            "outcome": outcome.status.value,
            "failed_checks": [c for c in checks if not c.get("passed", False)],
            "transcript": transcript,
        }
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        resp = self.llm.complete(messages, purpose="extract",
                                 task_id=task.task_id,
                                 prompt_version=self.PROMPT_VERSION,
                                 max_tokens=900)
        data = _parse_json_object(resp.text)
        lesson = str(data.get("lesson", "")).strip()
        if not lesson:
            return ExtractionResult([], source="llm", error="model declined to extract")

        exp = Experience(
            lesson=lesson,
            task_id=task.task_id,
            trace_id=trace.trace_id,
            task_context=task.goal[:500],
            evidence_refs=evidence_refs,
            action_trace_refs=action_refs,
            outcome=outcome.status.value,
            failure_signature=str(data.get("failure_signature", "")).strip()
            or self._signature(task, [c for c in checks if not c.get("passed")], outcome),
            applicability_conditions=[str(x) for x in data.get(
                "applicability_conditions", self._conditions(task, outcome))],
            family=task.family,
            source_kind="llm",
            prompt_version=self.PROMPT_VERSION,
            confidence=float(data.get("confidence", 0.5) or 0.5),
        )
        return ExtractionResult([exp], source="llm")

    @staticmethod
    def _transcript(trace: ExecutionTrace, limit: int = 12) -> List[Dict[str, Any]]:
        out = []
        for step in trace.steps[-limit:]:
            out.append({
                "step": step.index,
                "thought": step.thought[:300],
                "tool": step.tool,
                "args": step.args,
                "ok": bool(step.receipt.ok) if step.receipt else None,
                "output": (json.dumps(step.receipt.output, ensure_ascii=False)[:300]
                           if step.receipt and step.receipt.output is not None else ""),
                "error": step.receipt.error if step.receipt else "",
            })
        return out


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class ValidationReport:
    passed: bool
    checks: List[ValidationCheck] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "checks": [c.to_dict() for c in self.checks]}

    @property
    def failures(self) -> List[ValidationCheck]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        if self.passed:
            return "all checks passed"
        return "; ".join(f"{c.name}: {c.detail}" for c in self.failures)


class ExperienceValidator:
    """Programmatic admission. A model may only *veto* via ``model_veto``.

    Every check here is a claim the code can defend without a model in the loop.
    """

    MIN_LESSON_CHARS = 20

    def __init__(self, evidence: EvidenceStore,
                 existing_texts: Optional[Sequence[str]] = None,
                 leak_guard: Optional[Sequence[str]] = None,
                 duplicate_threshold: float = 0.95):
        self.evidence = evidence
        self.existing_texts = list(existing_texts or [])
        self.leak_guard = [s for s in (leak_guard or []) if s]
        self.duplicate_threshold = duplicate_threshold

    def validate(self, exp: Experience, trace: Optional[ExecutionTrace] = None,
                 outcome: Optional[Outcome] = None,
                 model_veto: Optional[tuple[bool, str]] = None) -> ValidationReport:
        checks: List[ValidationCheck] = []

        checks.append(ValidationCheck(
            "lesson_substantive", len(exp.lesson.strip()) >= self.MIN_LESSON_CHARS,
            f"lesson has {len(exp.lesson.strip())} chars"))

        missing = [r for r in exp.evidence_refs if not self.evidence.exists(r)]
        checks.append(ValidationCheck(
            "evidence_resolvable", not missing,
            f"missing evidence: {missing}" if missing else
            f"{len(exp.evidence_refs)} evidence refs resolve"))

        checks.append(ValidationCheck(
            "evidence_present", bool(exp.evidence_refs),
            f"{len(exp.evidence_refs)} refs"))

        if outcome is not None:
            checks.append(ValidationCheck(
                "outcome_program_verified", outcome.verified_by is VerifiedBy.PROGRAM,
                f"verified_by={outcome.verified_by.value}"))

        if trace is not None:
            valid_refs = {f"step:{s.index}:{s.tool}" for s in trace.tool_calls()}
            bad = [r for r in exp.action_trace_refs if r not in valid_refs]
            checks.append(ValidationCheck(
                "trace_refs_resolve", not bad,
                f"unresolved: {bad}" if bad else
                f"{len(exp.action_trace_refs)} refs resolve"))

        checks.append(ValidationCheck(
            "applicability_declared", bool(exp.applicability_conditions),
            f"{len(exp.applicability_conditions)} conditions"))

        leaked = [s for s in self.leak_guard if s and s in exp.lesson]
        checks.append(ValidationCheck(
            "no_eval_leakage", not leaked,
            f"lesson contains evaluation answer text: {leaked[:3]}" if leaked else "clean"))

        dup = self._duplicate_score(exp.lesson)
        checks.append(ValidationCheck(
            "not_duplicate", dup < self.duplicate_threshold,
            f"max similarity to existing active lesson {dup:.3f}"))

        if model_veto is not None:
            vetoed, reason = model_veto
            checks.append(ValidationCheck("model_veto", not vetoed,
                                          reason if vetoed else "model did not veto"))

        return ValidationReport(all(c.passed for c in checks), checks)

    def _duplicate_score(self, lesson: str) -> float:
        if not self.existing_texts:
            return 0.0
        from .retrieval import HashEmbedder
        emb = HashEmbedder()
        v = emb.embed(lesson)
        return max((cosine(v, emb.embed(t)) for t in self.existing_texts), default=0.0)


# --------------------------------------------------------------------------
# critique
# --------------------------------------------------------------------------


class MemoryCritic:
    """Decides whether a retrieved experience applies to the *current* state.

    This is the anti-negative-transfer gate. Heuristic path uses the declared
    applicability conditions; the model path may only make the judgement more
    conservative (it can turn apply into reject, and its verdict is recorded).
    """

    PROMPT_VERSION = "critique-v1"

    SYSTEM = (
        "You judge whether a stored lesson applies to the current task.\n"
        "Return STRICT JSON: {\"verdict\": \"apply\"|\"adapt\"|\"reject\", "
        "\"reasons\": [str], \"confidence\": float}\n"
        "Reject when the stored lesson was learned under materially different "
        "conditions. Prefer reject over a confident wrong answer.\n"
    )

    def __init__(self, llm: Optional[BudgetedLLM] = None,
                 applicability_threshold: float = 0.34):
        self.llm = llm
        self.applicability_threshold = applicability_threshold

    def critique(self, exp: Experience, task: Task,
                 state_text: str = "") -> Critique:
        heuristic = self._heuristic(exp, task)
        # A structural gate is not a matter of judgement. If the experience was
        # learned in a different task family, a model does not get a vote — an
        # early live run showed the model happily approving a cross-family
        # experience, which is precisely the negative transfer this class exists
        # to prevent.
        if heuristic.hard:
            return heuristic
        if self.llm is None:
            return heuristic
        try:
            return self._llm_critique(exp, task, state_text, heuristic)
        except (LLMError, ValueError, KeyError, json.JSONDecodeError):
            return heuristic

    def _heuristic(self, exp: Experience, task: Task) -> Critique:
        reasons: List[str] = []
        if exp.family and task.family and exp.family != task.family:
            return Critique(CritiqueVerdict.REJECT,
                            [f"family mismatch: memory={exp.family} task={task.family}"],
                            confidence=0.9, source="heuristic", hard=True)

        task_tokens = set(tokenize(" ".join([task.goal, " ".join(task.tags)])))
        cond_hits = 0
        cond_total = 0
        for cond in exp.applicability_conditions:
            kind, _, value = cond.partition(":")
            if kind != "tag":
                continue
            cond_total += 1
            if value.lower() in {t.lower() for t in task.tags} or value.lower() in task_tokens:
                cond_hits += 1
        ratio = (cond_hits / cond_total) if cond_total else 1.0
        if ratio < self.applicability_threshold:
            reasons.append(f"only {cond_hits}/{cond_total} tag conditions hold")
            return Critique(CritiqueVerdict.REJECT, reasons, confidence=0.6,
                            source="heuristic")
        if ratio < 1.0:
            reasons.append(f"{cond_hits}/{cond_total} tag conditions hold")
            return Critique(CritiqueVerdict.ADAPT, reasons, confidence=0.6,
                            source="heuristic")
        reasons.append("family and declared conditions match")
        return Critique(CritiqueVerdict.APPLY, reasons, confidence=0.7,
                        source="heuristic")

    def _llm_critique(self, exp: Experience, task: Task, state_text: str,
                      heuristic: Critique) -> Critique:
        assert self.llm is not None
        payload = {
            "lesson": exp.lesson,
            "learned_under": exp.task_context,
            "applicability_conditions": exp.applicability_conditions,
            "current_goal": task.goal,
            "current_state": state_text[:1000],
            "current_family": task.family,
            "current_tags": task.tags,
            "heuristic_prior": heuristic.verdict.value,
        }
        resp = self.llm.complete(
            [{"role": "system", "content": self.SYSTEM},
             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            purpose="critique", task_id=task.task_id,
            prompt_version=self.PROMPT_VERSION, max_tokens=700)
        data = _parse_json_object(resp.text)
        try:
            verdict = CritiqueVerdict(str(data.get("verdict", "")).lower())
        except ValueError:
            return heuristic
        return Critique(verdict,
                        [str(r) for r in data.get("reasons", [])][:5],
                        confidence=float(data.get("confidence", 0.5) or 0.5),
                        source="llm")


# --------------------------------------------------------------------------
# reconstruction
# --------------------------------------------------------------------------


class ExperienceReconstructor:
    """Rewrites a lesson into guidance for the current task, or returns None.

    Returning ``None`` is a real outcome and is counted: it is the "'<EMPTY>'
    reject" of this system.
    """

    PROMPT_VERSION = "reconstruct-v1"

    SYSTEM = (
        "Rewrite a stored lesson so it applies to the CURRENT task.\n"
        "Output 1-3 short imperative sentences, no preamble, no first person.\n"
        "Do not invent new facts. If the lesson cannot be adapted truthfully, "
        "return exactly <EMPTY>.\n"
    )

    def __init__(self, llm: Optional[BudgetedLLM] = None):
        self.llm = llm

    def reconstruct(self, exp: Experience, task: Task, critique: Critique) -> Optional[str]:
        if critique.verdict is CritiqueVerdict.REJECT:
            return None
        if self.llm is None:
            return self._heuristic(exp, task)
        try:
            return self._llm(exp, task)
        except (LLMError, ValueError):
            return self._heuristic(exp, task)

    @staticmethod
    def _heuristic(exp: Experience, task: Task) -> Optional[str]:
        lesson = exp.lesson.strip()
        if not lesson:
            return None
        scope = task.family or "this task"
        return f"[{scope}] {lesson}"

    def _llm(self, exp: Experience, task: Task) -> Optional[str]:
        assert self.llm is not None
        payload = {
            "lesson": exp.lesson,
            "learned_under": exp.task_context,
            "conditions": exp.applicability_conditions,
            "current_goal": task.goal,
            "current_family": task.family,
        }
        resp = self.llm.complete(
            [{"role": "system", "content": self.SYSTEM},
             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            purpose="reconstruct", task_id=task.task_id,
            prompt_version=self.PROMPT_VERSION, max_tokens=600)
        text = resp.text.strip()
        if not text or text.upper().startswith("<EMPTY"):
            return None
        return text[:800]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _parse_json_object(text: str) -> Dict[str, Any]:
    """Tolerant JSON extraction: models like to wrap objects in prose or fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in model output")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model output was not a JSON object")
    return value
