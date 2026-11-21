"""Versioned experience memory.

This is the module that answers the seven questions the project brief asks of
the memory closed loop:

1. why was this experience kept   -> ``Experience.lesson`` + the validator's checks
2. which execution it came from   -> ``task_id`` / ``trace_id`` / ``action_trace_refs``
3. whether it was verified        -> ``verification_status`` written only by the validator
4. why it was retrieved           -> ``memory_events.retrieved_json`` (per-channel scores)
5. whether it fits the new task   -> ``Critique``
6. how a wrong memory is fixed    -> ``revise()`` creates version n+1, old row superseded
7. how it is kept from spreading  -> status machine + gate + programmatic validation

Nothing here mutates content. The only mutable column is ``status``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .db import Database, dumps, loads
from .models import (
    Actor,
    Experience,
    ExperienceStatus,
    TransitionError,
    assert_transition,
    now_iso,
)


class MemoryWriteError(Exception):
    pass


class ExperienceStore:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------
    # creation
    # ------------------------------------------------------------------
    def add(self, exp: Experience, actor: Actor = Actor.EXTRACTOR) -> Experience:
        """Insert a new experience version. New rows always start as candidates."""
        if not exp.lesson or not exp.lesson.strip():
            raise MemoryWriteError("an experience must carry a non-empty lesson")
        exp.status = ExperienceStatus.CANDIDATE
        exp.verification_status = ExperienceStatus.CANDIDATE.value
        exp.created_at = exp.created_at or now_iso()
        exp.updated_at = exp.created_at
        row = exp.to_row()
        row["evidence_refs_json"] = dumps(exp.evidence_refs)
        row["action_trace_refs_json"] = dumps(exp.action_trace_refs)
        row["applicability_json"] = dumps(exp.applicability_conditions)
        row["condition_json"] = dumps(exp.condition) if exp.condition is not None else None
        self.db.insert("experiences", row)
        self._event(exp, from_status=None, to_status=exp.status, actor=actor,
                    reason="created")
        return exp

    def revise(
        self,
        experience_id: str,
        *,
        lesson: Optional[str] = None,
        condition: Optional[Dict[str, Any]] = None,
        applicability: Optional[List[str]] = None,
        failure_signature: Optional[str] = None,
        actor: Actor = Actor.EVOLUTION,
        reason: str = "revised",
    ) -> Experience:
        """Create version n+1 as a *candidate*, leaving the current version in force.

        An earlier version superseded the predecessor here, at creation time. That
        was wrong: if the candidate is then rejected by replay, the good version is
        already retired and the agent is left with nothing. Superseding now happens
        only when the new version is actually activated (see ``transition``), so a
        failed candidate costs nothing.
        """
        prev = self.latest(experience_id)
        if prev is None:
            raise MemoryWriteError(f"unknown experience {experience_id}")
        new = Experience(
            experience_id=prev.experience_id,
            version=prev.version + 1,
            lesson=lesson if lesson is not None else prev.lesson,
            condition=condition if condition is not None else prev.condition,
            recommended_action=prev.recommended_action,
            task_id=prev.task_id,
            trace_id=prev.trace_id,
            task_context=prev.task_context,
            evidence_refs=list(prev.evidence_refs),
            action_trace_refs=list(prev.action_trace_refs),
            outcome=prev.outcome,
            failure_signature=failure_signature if failure_signature is not None
            else prev.failure_signature,
            applicability_conditions=applicability if applicability is not None
            else list(prev.applicability_conditions),
            family=prev.family,
            source_kind=prev.source_kind,
            prompt_version=prev.prompt_version,
            supersedes=prev.version,
            confidence=prev.confidence,
        )
        self.add(new, actor=actor)
        return new

    # ------------------------------------------------------------------
    # state machine
    # ------------------------------------------------------------------
    def transition(
        self,
        experience_id: str,
        version: int,
        target: ExperienceStatus,
        *,
        actor: Actor,
        reason: str = "",
        evidence_refs: Optional[List[str]] = None,
    ) -> Experience:
        exp = self.get(experience_id, version)
        if exp is None:
            raise MemoryWriteError(f"unknown experience {experience_id} v{version}")
        previous_status = exp.status
        try:
            assert_transition(exp.status, target, actor)
        except TransitionError as exc:
            raise MemoryWriteError(str(exc)) from exc

        self.db.execute(
            "UPDATE experiences SET status = ?, updated_at = ? "
            "WHERE experience_id = ? AND version = ?",
            (target.value, now_iso(), experience_id, version),
        )
        if target in (ExperienceStatus.ACCEPTED, ExperienceStatus.ACTIVE):
            # verification_status mirrors acceptance; only the validator can set it
            self.db.execute(
                "UPDATE experiences SET verification_status = ? "
                "WHERE experience_id = ? AND version = ?",
                (target.value, experience_id, version),
            )
        exp.status = target
        # from_status must be the status *before* the move; recording exp.status
        # here logged every transition as "X -> X" and made the audit trail useless.
        self._event(exp, from_status=previous_status, to_status=target, actor=actor,
                    reason=reason, evidence_refs=evidence_refs)
        # Activating a new version is what retires the one it replaces. Doing it
        # here rather than at creation means a rejected candidate never disturbs
        # the version that is currently in force.
        if target is ExperienceStatus.ACTIVE and exp.supersedes is not None:
            prior = self.get(exp.experience_id, exp.supersedes)
            if prior is not None and prior.status is ExperienceStatus.ACTIVE:
                self.transition(prior.experience_id, prior.version,
                                ExperienceStatus.SUPERSEDED, actor=Actor.GATE,
                                reason=f"superseded by v{exp.version}")
        return self.get(experience_id, version)  # type: ignore[return-value]

    def submit_for_validation(self, experience_id: str, version: int,
                              actor: Actor = Actor.EXTRACTOR) -> Experience:
        return self.transition(experience_id, version,
                               ExperienceStatus.PENDING_VALIDATION, actor=actor,
                               reason="submitted")

    def accept(self, experience_id: str, version: int, *, reason: str,
               evidence_refs: Optional[List[str]] = None) -> Experience:
        return self.transition(experience_id, version, ExperienceStatus.ACCEPTED,
                               actor=Actor.VALIDATOR, reason=reason,
                               evidence_refs=evidence_refs)

    def reject(self, experience_id: str, version: int, *, reason: str) -> Experience:
        return self.transition(experience_id, version, ExperienceStatus.REJECTED,
                               actor=Actor.VALIDATOR, reason=reason)

    def activate(self, experience_id: str, version: int, *, reason: str = "") -> Experience:
        return self.transition(experience_id, version, ExperienceStatus.ACTIVE,
                               actor=Actor.GATE, reason=reason)

    def deprecate(self, experience_id: str, version: int, *, reason: str,
                  actor: Actor = Actor.GATE) -> Experience:
        return self.transition(experience_id, version, ExperienceStatus.DEPRECATED,
                               actor=actor, reason=reason)

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------
    def get(self, experience_id: str, version: Optional[int] = None) -> Optional[Experience]:
        if version is None:
            row = self.db.query_one(
                "SELECT * FROM experiences WHERE experience_id = ? "
                "ORDER BY version DESC LIMIT 1", (experience_id,))
        else:
            row = self.db.query_one(
                "SELECT * FROM experiences WHERE experience_id = ? AND version = ?",
                (experience_id, version))
        return self._row_to_experience(row) if row else None

    def latest(self, experience_id: str) -> Optional[Experience]:
        return self.get(experience_id, None)

    def active(self, family: Optional[str] = None) -> List[Experience]:
        # Ordering by lesson as a secondary key keeps repeated runs identical even
        # when two experiences share a timestamp.
        if family is None:
            rows = self.db.query(
                "SELECT * FROM experiences WHERE status = ? "
                "ORDER BY created_at, lesson",
                (ExperienceStatus.ACTIVE.value,))
        else:
            rows = self.db.query(
                "SELECT * FROM experiences WHERE status = ? AND family = ? "
                "ORDER BY created_at, lesson",
                (ExperienceStatus.ACTIVE.value, family))
        return [self._row_to_experience(r) for r in rows]

    def all_versions(self, experience_id: str) -> List[Experience]:
        rows = self.db.query(
            "SELECT * FROM experiences WHERE experience_id = ? ORDER BY version",
            (experience_id,))
        return [self._row_to_experience(r) for r in rows]

    def events(self, experience_id: str) -> List[Dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM experience_events WHERE experience_id = ? ORDER BY id",
            (experience_id,))
        return [dict(r) for r in rows]

    def counts_by_status(self) -> Dict[str, int]:
        rows = self.db.query(
            "SELECT status, COUNT(*) AS n FROM experiences GROUP BY status")
        return {r["status"]: int(r["n"]) for r in rows}

    def find_by_signature(self, failure_signature: str,
                          family: str = "") -> List[Experience]:
        rows = self.db.query(
            "SELECT * FROM experiences WHERE failure_signature = ? "
            "AND status = ? AND (family = ? OR ? = '')",
            (failure_signature, ExperienceStatus.ACTIVE.value, family, family))
        return [self._row_to_experience(r) for r in rows]

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _event(self, exp: Experience, *, from_status: Optional[ExperienceStatus],
               to_status: ExperienceStatus, actor: Actor, reason: str,
               evidence_refs: Optional[Iterable[str]] = None) -> None:
        self.db.insert(
            "experience_events",
            {
                "experience_id": exp.experience_id,
                "version": exp.version,
                "from_status": from_status.value if from_status else None,
                "to_status": to_status.value,
                "actor": actor.value,
                "reason": reason,
                "evidence_refs_json": dumps(list(evidence_refs or [])),
                "created_at": now_iso(),
            },
        )

    @staticmethod
    def _row_to_experience(row: Any) -> Experience:
        return Experience(
            experience_id=row["experience_id"],
            version=int(row["version"]),
            task_id=row["task_id"],
            trace_id=row["trace_id"],
            task_context=row["task_context"],
            evidence_refs=loads(row["evidence_refs_json"], []),
            action_trace_refs=loads(row["action_trace_refs_json"], []),
            outcome=row["outcome"],
            failure_signature=row["failure_signature"],
            lesson=row["lesson"],
            condition=loads(row["condition_json"], None),
            recommended_action=row["recommended_action"] or "skip",
            applicability_conditions=loads(row["applicability_json"], []),
            verification_status=row["verification_status"],
            status=ExperienceStatus(row["status"]),
            family=row["family"],
            source_kind=row["source_kind"],
            prompt_version=row["prompt_version"],
            supersedes=row["supersedes"],
            confidence=float(row["confidence"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
