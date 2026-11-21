"""Memory lifecycle: versions, state machine, write permissions."""

import pytest

from evomemory.memory import MemoryWriteError
from evomemory.models import (
    Actor,
    Experience,
    ExperienceStatus,
    TransitionError,
    assert_transition,
)


def make_exp(lesson="For escalation, do not act on records with tier='gold'."):
    return Experience(lesson=lesson, family="escalation", evidence_refs=["ev_x"])


def test_new_experience_starts_as_candidate(store):
    exp = store.add(make_exp())
    assert exp.status is ExperienceStatus.CANDIDATE
    assert exp.version == 1
    assert exp.verification_status == "candidate"


def test_empty_lesson_refused(store):
    with pytest.raises(MemoryWriteError):
        store.add(Experience(lesson="   "))


def test_full_happy_path_to_active(store):
    exp = store.add(make_exp())
    store.submit_for_validation(exp.experience_id, exp.version)
    store.accept(exp.experience_id, exp.version, reason="checks passed")
    store.activate(exp.experience_id, exp.version)
    assert store.get(exp.experience_id).status is ExperienceStatus.ACTIVE


def test_model_cannot_accept_or_activate(store):
    """The load-bearing permission: a model may propose, never promote."""
    exp = store.add(make_exp())
    store.submit_for_validation(exp.experience_id, exp.version)
    with pytest.raises(MemoryWriteError):
        store.transition(exp.experience_id, exp.version,
                         ExperienceStatus.ACCEPTED, actor=Actor.MODEL)
    with pytest.raises(MemoryWriteError):
        store.transition(exp.experience_id, exp.version,
                         ExperienceStatus.ACTIVE, actor=Actor.EXTRACTOR)


def test_cannot_skip_validation(store):
    exp = store.add(make_exp())
    with pytest.raises(MemoryWriteError):
        store.activate(exp.experience_id, exp.version)  # candidate -> active


def test_terminal_states_are_terminal():
    assert TransitionError  # imported for symmetry
    from evomemory.models import ALLOWED_TRANSITIONS
    assert ALLOWED_TRANSITIONS[ExperienceStatus.REJECTED] == set()
    assert ALLOWED_TRANSITIONS[ExperienceStatus.SUPERSEDED] == set()


def test_transition_is_logged(store):
    exp = store.add(make_exp())
    store.submit_for_validation(exp.experience_id, exp.version)
    store.accept(exp.experience_id, exp.version, reason="ok")
    events = store.events(exp.experience_id)
    assert [e["to_status"] for e in events] == [
        "candidate", "pending_validation", "accepted"]
    assert all(e["actor"] for e in events)


def test_revise_leaves_the_live_version_in_force(store):
    """A candidate is just a candidate: creating it must not retire anything.

    Regression for the flaw where ``revise`` superseded the predecessor at
    creation time, so a rejected candidate left the agent with no active memory.
    """
    exp = store.add(make_exp())
    store.submit_for_validation(exp.experience_id, exp.version)
    store.accept(exp.experience_id, exp.version, reason="ok")
    store.activate(exp.experience_id, exp.version)

    new = store.revise(exp.experience_id, lesson="A better lesson, still actionable.",
                       reason="contradicted by later evidence")

    assert new.version == 2
    assert new.supersedes == 1
    assert new.status is ExperienceStatus.CANDIDATE
    old = store.get(exp.experience_id, 1)
    assert old.status is ExperienceStatus.ACTIVE, "the live version must stay in force"
    assert [e.experience_id for e in store.active()] == [exp.experience_id]
    # the old text is still readable: nothing was destroyed
    assert "tier='gold'" in old.lesson
    assert len(store.all_versions(exp.experience_id)) == 2


def test_rejected_candidate_does_not_disturb_the_live_version(store):
    exp = store.add(make_exp())
    store.submit_for_validation(exp.experience_id, 1)
    store.accept(exp.experience_id, 1, reason="ok")
    store.activate(exp.experience_id, 1)

    v2 = store.revise(exp.experience_id, lesson="A candidate that will fail replay.")
    store.submit_for_validation(v2.experience_id, v2.version)
    store.reject(v2.experience_id, v2.version, reason="replay found counterexamples")

    assert store.get(exp.experience_id, 1).status is ExperienceStatus.ACTIVE
    assert store.get(exp.experience_id, 2).status is ExperienceStatus.REJECTED
    assert len(store.active()) == 1


def test_activating_a_new_version_supersedes_its_predecessor(store):
    exp = store.add(make_exp())
    store.submit_for_validation(exp.experience_id, 1)
    store.accept(exp.experience_id, 1, reason="ok")
    store.activate(exp.experience_id, 1)

    v2 = store.revise(exp.experience_id, lesson="The corrected rule, long enough to pass.")
    store.submit_for_validation(v2.experience_id, v2.version)
    store.accept(v2.experience_id, v2.version, reason="replay passed")
    store.activate(v2.experience_id, v2.version)

    assert store.get(exp.experience_id, 1).status is ExperienceStatus.SUPERSEDED
    assert store.get(exp.experience_id, 2).status is ExperienceStatus.ACTIVE
    assert len(store.active()) == 1
    # the audit trail records the real transition, not "active -> active"
    events = store.events(exp.experience_id)
    superseded = [e for e in events if e["to_status"] == "superseded"]
    assert superseded and superseded[0]["from_status"] == "active"


def test_revise_keeps_old_version_readable_after_new_one_activates(store):
    exp = store.add(make_exp())
    store.submit_for_validation(exp.experience_id, 1)
    store.accept(exp.experience_id, 1, reason="ok")
    store.activate(exp.experience_id, 1)
    v2 = store.revise(exp.experience_id, lesson="Revised lesson text that is long enough.")
    store.submit_for_validation(v2.experience_id, v2.version)
    store.accept(v2.experience_id, v2.version, reason="ok")
    store.activate(v2.experience_id, v2.version)

    assert store.get(exp.experience_id, 1) is not None
    assert store.get(exp.experience_id, 2).status is ExperienceStatus.ACTIVE
    assert len(store.active()) == 1, "only one version may be active"


def test_content_is_immutable_after_write(store):
    exp = store.add(make_exp())
    store.submit_for_validation(exp.experience_id, exp.version)
    store.accept(exp.experience_id, exp.version, reason="ok")
    row = store.db.query_one(
        "SELECT lesson FROM experiences WHERE experience_id = ?",
        (exp.experience_id,))
    assert row["lesson"] == exp.lesson
    # there is no update path for lesson; status is the only mutable column
    assert store.db.query_one(
        "SELECT COUNT(*) AS n FROM experiences WHERE experience_id = ?",
        (exp.experience_id,))["n"] == 1


def test_counts_by_status(store):
    e1 = store.add(make_exp())
    store.add(make_exp("Another sufficiently long lesson for a second record."))
    store.submit_for_validation(e1.experience_id, 1)
    counts = store.counts_by_status()
    assert counts.get("candidate") == 1
    assert counts.get("pending_validation") == 1
