"""Content-addressed evidence: dedupe, immutability, derivation lineage."""

import pytest

from evomemory.models import EvidenceKind


def test_put_and_read_roundtrip(tmp_env):
    _, evidence = tmp_env
    ev = evidence.put_text("hello evidence")
    assert ev.bytes_len == len("hello evidence")
    assert evidence.read_text(ev.evidence_id) == "hello evidence"


def test_identical_bytes_dedupe(tmp_env):
    _, evidence = tmp_env
    a = evidence.put_text("same")
    b = evidence.put_text("same")
    assert a.evidence_id == b.evidence_id
    assert evidence.count() == 1


def test_same_bytes_different_kind_are_distinct(tmp_env):
    _, evidence = tmp_env
    a = evidence.put(b"x", EvidenceKind.TEXT)
    b = evidence.put(b"x", EvidenceKind.IMAGE)
    assert a.evidence_id != b.evidence_id


def test_path_is_content_addressed_and_file_never_rewritten(tmp_env):
    _, evidence = tmp_env
    ev = evidence.put_text("original")
    path = ev.path
    # writing different bytes produces a different record, not an overwrite
    other = evidence.put_text("modified")
    assert other.path != path
    assert evidence.read_text(ev.evidence_id) == "original"


def test_derived_evidence_points_at_parent(tmp_env):
    _, evidence = tmp_env
    parent = evidence.put(b"\x89PNG-fake", EvidenceKind.IMAGE)
    thumb = evidence.put(b"tiny", EvidenceKind.THUMBNAIL, derived_from=parent.evidence_id)
    children = evidence.children(parent.evidence_id)
    assert [c.evidence_id for c in children] == [thumb.evidence_id]
    assert evidence.get(thumb.evidence_id).derived_from == parent.evidence_id


def test_unknown_evidence_raises(tmp_env):
    _, evidence = tmp_env
    with pytest.raises(KeyError):
        evidence.get("ev_missing")
    assert evidence.exists("ev_missing") is False


def test_json_evidence_sorted_for_stability(tmp_env):
    _, evidence = tmp_env
    a = evidence.put_json({"b": 1, "a": 2})
    b = evidence.put_json({"a": 2, "b": 1})
    assert a.sha256 == b.sha256, "dict ordering must not change the hash"
