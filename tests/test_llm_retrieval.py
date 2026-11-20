"""Budget enforcement and retrieval behaviour."""

import pytest

from evomemory.llm import BudgetedLLM, BudgetExceeded, ScriptedLLM, estimate_tokens
from evomemory.retrieval import (
    BM25Index, ExperienceIndex, HashEmbedder, cosine, rrf, tokenize,
)
from evomemory.models import Experience


# -- budget ---------------------------------------------------------------


def test_budget_stops_before_the_call_that_would_exceed():
    llm = BudgetedLLM(ScriptedLLM(lambda m: "ok"), budget_calls=2, budget_tokens=10_000)
    llm.complete([{"role": "user", "content": "a"}], purpose="test")
    llm.complete([{"role": "user", "content": "a"}], purpose="test")
    with pytest.raises(BudgetExceeded):
        llm.complete([{"role": "user", "content": "a"}], purpose="test")
    assert llm.calls == 2


def test_budget_counts_tokens():
    llm = BudgetedLLM(ScriptedLLM(lambda m: "x" * 400), budget_calls=10,
                      budget_tokens=50)
    with pytest.raises(BudgetExceeded):
        llm.complete([{"role": "user", "content": "start"}], purpose="test")


def test_budget_records_every_call_in_db(tmp_env):
    db, _ = tmp_env
    llm = BudgetedLLM(ScriptedLLM(lambda m: "reply"), budget_calls=5,
                      budget_tokens=10_000, db=db)
    llm.complete([{"role": "user", "content": "hi"}], purpose="act",
                 prompt_version="v1", task_id="t1")
    rows = db.query("SELECT * FROM llm_calls")
    assert len(rows) == 1
    assert rows[0]["purpose"] == "act"
    assert rows[0]["ok"] == 1


def test_estimate_tokens_is_deterministic():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("") == 0


# -- retrieval ------------------------------------------------------------


def test_tokenize_handles_cjk_bigrams():
    tokens = tokenize("升级记录")
    assert all(ch in tokens for ch in "升级记录")  # individual chars
    assert "升级" in tokens                        # bigrams
    assert "级记" in tokens


def test_cosine_rejects_length_mismatch():
    """MemHarness silently returns 0.0 here, which hides bugs as 'dissimilar'."""
    with pytest.raises(ValueError):
        cosine([1.0, 0.0], [1.0])


def test_hash_embedder_is_stable_across_calls():
    emb = HashEmbedder()
    assert emb.embed("gold tier records") == emb.embed("gold tier records")


def test_bm25_ranks_relevant_document_first():
    index = BM25Index({
        "a": "do not escalate gold tier records",
        "b": "reconcile invoices for the finance team",
        "c": "gold records require audit",
    })
    hits = index.search("gold records", limit=2)
    assert hits[0][0] in {"a", "c"}
    assert hits[0][1] > 0


def test_rrf_prefers_items_ranked_by_both_channels():
    fused = rrf({"bm25": [("x", 9.0), ("y", 1.0)],
                 "dense": [("x", 0.9), ("y", 0.8)]})
    assert fused["x"] > fused["y"]


def _exp(lesson, family="escalation", eid=None):
    kwargs = {"lesson": lesson, "family": family}
    if eid:
        kwargs["experience_id"] = eid
    return Experience(**kwargs)


def test_experience_index_filters_by_family():
    index = ExperienceIndex([
        _exp("do not escalate gold tier records", "escalation", "e1"),
        _exp("close escalated records only", "closure", "e2"),
    ])
    hits = index.search("escalate records", limit=5, family="escalation")
    assert [h[0].experience_id for h in hits] == ["e1"]


def test_experience_index_returns_channel_scores():
    index = ExperienceIndex([_exp("do not escalate gold tier records", eid="e1")])
    hits = index.search("gold", limit=3)
    assert hits
    _e, _score, channels = hits[0]
    assert set(channels) == {"bm25", "dense"}


def test_caption_map_participates_in_retrieval():
    plain = ExperienceIndex([_exp("records must be reconciled", eid="e1")])
    with_caption = ExperienceIndex(
        [_exp("records must be reconciled", eid="e1")],
        caption_map={"e1": "screenshot showing a crimson banner"})
    plain_hits = plain.search("crimson banner", limit=1)
    cap_hits = with_caption.search("crimson banner", limit=1)
    assert cap_hits and cap_hits[0][2]["bm25"] > 0
    assert not plain_hits or plain_hits[0][2]["bm25"] == 0.0
