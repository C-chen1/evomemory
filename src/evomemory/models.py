"""Core data model for EvoMemory.

Everything the system persists is defined here so that the SQLite schema in
``db.py`` is a mechanical projection of these types rather than a parallel
universe.

Design notes that matter:

* ``Experience`` has a stable ``experience_id`` and an increasing ``version``.
  Content fields are immutable once written; only ``status`` may change, and
  every change goes through the state machine and lands in ``experience_events``.
* Verification is never the model's own opinion. ``VerifiedBy`` distinguishes a
  programmatic check from a model judgement, and the validator refuses to
  promote anything that was only ever judged by a model.
* Evidence is content addressed (``sha256``) and never mutated in place.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


# --------------------------------------------------------------------------
# enums
# --------------------------------------------------------------------------


class EvidenceKind(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TOOL_OUTPUT = "tool_output"
    ENV_STATE = "env_state"
    THUMBNAIL = "thumbnail"


class OutcomeStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class VerifiedBy(str, Enum):
    """Who decided the outcome. ``MODEL`` is allowed to exist, never to gate."""

    PROGRAM = "program"
    MODEL = "model"
    NONE = "none"


class ExperienceStatus(str, Enum):
    CANDIDATE = "candidate"
    PENDING_VALIDATION = "pending_validation"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DEPRECATED = "deprecated"

    @property
    def is_retrievable(self) -> bool:
        return self is ExperienceStatus.ACTIVE


class Actor(str, Enum):
    """Who is trying to change memory state.

    The model is allowed to *propose*. It is never allowed to declare an
    experience verified, active, or a task complete. This mirrors the write
    permission split that makes Recuris' ledger trustworthy.
    """

    MODEL = "model"
    EXTRACTOR = "extractor"
    VALIDATOR = "validator"
    GATE = "gate"
    EVOLUTION = "evolution"
    HUMAN = "human"


class CritiqueVerdict(str, Enum):
    APPLY = "apply"        # experience fits the current state as-is
    ADAPT = "adapt"        # useful but must be rewritten for this state
    REJECT = "reject"      # not applicable; do not inject


class MemoryMode(str, Enum):
    """The four benchmark arms."""

    A_NO_MEMORY = "A"
    B_VECTOR_MEMORY = "B"
    C_MULTIMODAL_MEMORY = "C"
    D_FULL_LIFECYCLE = "D"


# --------------------------------------------------------------------------
# state machine
# --------------------------------------------------------------------------

ALLOWED_TRANSITIONS: Dict[ExperienceStatus, set[ExperienceStatus]] = {
    ExperienceStatus.CANDIDATE: {ExperienceStatus.PENDING_VALIDATION, ExperienceStatus.REJECTED},
    ExperienceStatus.PENDING_VALIDATION: {ExperienceStatus.ACCEPTED, ExperienceStatus.REJECTED},
    ExperienceStatus.ACCEPTED: {ExperienceStatus.ACTIVE, ExperienceStatus.DEPRECATED},
    ExperienceStatus.ACTIVE: {ExperienceStatus.SUPERSEDED, ExperienceStatus.DEPRECATED},
    ExperienceStatus.REJECTED: set(),
    ExperienceStatus.SUPERSEDED: set(),
    ExperienceStatus.DEPRECATED: set(),
}


class TransitionError(Exception):
    """Raised when a status change violates the memory lifecycle."""


#: Only these actors may move an experience out of ``candidate``/``pending``.
#: The extractor may submit; only the validator may accept; only the gate may
#: activate.
TRANSITION_ACTORS: Dict[ExperienceStatus, set[Actor]] = {
    ExperienceStatus.PENDING_VALIDATION: {Actor.EXTRACTOR, Actor.EVOLUTION},
    ExperienceStatus.ACCEPTED: {Actor.VALIDATOR},
    ExperienceStatus.REJECTED: {Actor.VALIDATOR},
    ExperienceStatus.ACTIVE: {Actor.GATE},
    ExperienceStatus.SUPERSEDED: {Actor.EVOLUTION, Actor.GATE},
    ExperienceStatus.DEPRECATED: {Actor.EVOLUTION, Actor.GATE, Actor.HUMAN},
}


def assert_transition(current: ExperienceStatus, target: ExperienceStatus, actor: Actor) -> None:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise TransitionError(f"illegal transition {current.value} -> {target.value}")
    allowed_actors = TRANSITION_ACTORS.get(target, set())
    if actor not in allowed_actors:
        raise TransitionError(
            f"actor {actor.value} may not move experience to {target.value}"
            f" (allowed: {sorted(a.value for a in allowed_actors)})"
        )


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


@dataclass
class Task:
    goal: str
    task_id: str = field(default_factory=lambda: new_id("task"))
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    family: str = ""  # tasks sharing a family share the hidden rule
    max_steps: int = 12

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Task":
        return cls(**d)


@dataclass
class ToolReceipt:
    call_id: str
    tool: str
    ok: bool
    output: Any = None
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TraceStep:
    index: int
    thought: str = ""
    tool: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    receipt: Optional[ToolReceipt] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class ExecutionTrace:
    task_id: str
    trace_id: str = field(default_factory=lambda: new_id("trace"))
    run_id: str = ""
    model: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    steps: List[TraceStep] = field(default_factory=list)
    status: str = "running"
    started_at: str = field(default_factory=now_iso)
    ended_at: Optional[str] = None

    def add(self, step: TraceStep) -> None:
        step.index = len(self.steps)
        self.steps.append(step)

    def finish(self, status: str = "completed") -> None:
        self.status = status
        self.ended_at = now_iso()

    def tool_calls(self) -> List[TraceStep]:
        return [s for s in self.steps if s.tool]


@dataclass
class Outcome:
    status: OutcomeStatus
    verified_by: VerifiedBy
    score: float = 0.0
    reason: str = ""
    checks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["verified_by"] = self.verified_by.value
        return d


# --------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------


@dataclass
class Evidence:
    kind: EvidenceKind
    sha256: str
    path: str
    evidence_id: str = field(default_factory=lambda: new_id("ev"))
    mime: str = ""
    bytes_len: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    derived_from: Optional[str] = None
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


# --------------------------------------------------------------------------
# experience
# --------------------------------------------------------------------------


@dataclass
class Experience:
    """One immutable version of one experience.

    ``experience_id`` groups versions. ``supersedes`` points at the version
    number this row replaces (None for v1).
    """

    lesson: str
    experience_id: str = field(default_factory=lambda: new_id("exp"))
    version: int = 1
    task_id: str = ""
    trace_id: str = ""
    task_context: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    action_trace_refs: List[str] = field(default_factory=list)
    outcome: str = ""
    failure_signature: str = ""
    applicability_conditions: List[str] = field(default_factory=list)
    verification_status: str = ExperienceStatus.CANDIDATE.value
    status: ExperienceStatus = ExperienceStatus.CANDIDATE
    family: str = ""
    source_kind: str = "heuristic"       # heuristic | llm
    prompt_version: str = ""
    supersedes: Optional[int] = None
    confidence: float = 0.0
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def to_row(self) -> Dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "version": self.version,
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "task_context": self.task_context,
            "evidence_refs_json": self.evidence_refs,
            "action_trace_refs_json": self.action_trace_refs,
            "outcome": self.outcome,
            "failure_signature": self.failure_signature,
            "lesson": self.lesson,
            "applicability_json": self.applicability_conditions,
            "status": self.status.value,
            "verification_status": self.verification_status,
            "family": self.family,
            "source_kind": self.source_kind,
            "prompt_version": self.prompt_version,
            "supersedes": self.supersedes,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class Critique:
    verdict: CritiqueVerdict
    reasons: List[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reasons": self.reasons,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass
class RetrievedExperience:
    experience: Experience
    score: float
    channels: Dict[str, float] = field(default_factory=dict)
    critique: Optional[Critique] = None
    applicability: float = 1.0
    guidance: Optional[str] = None


@dataclass
class MemoryEvent:
    """Per-step audit record of what memory was considered and what was used."""

    task_id: str
    step: int
    query: str
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    injected: List[str] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    trace_id: str = ""
    run_id: str = ""
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
