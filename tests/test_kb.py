"""Knowledge base: import, chunking, provenance, retrieval, citations.

The embedding adapter is tested through a controllable stand-in. Whether the real
model actually ran is recorded separately in the results, not asserted here.
"""

import pytest

from evomemory.embeddings import (
    FastEmbedEmbedder, LexicalHashEmbedder, StubEmbedder, build_embedder, cosine,
)
from evomemory.kb import KnowledgeBase, split_document


@pytest.fixture()
def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.md").write_text(
        "# 发布流程\n\n灰度第一阶段只放 5% 的流量，观察 30 分钟。\n\n"
        "两阶段都通过后才允许全量发布，全量后观察 60 分钟。\n", encoding="utf-8")
    (root / "b.md").write_text(
        "# 回滚流程\n\n回滚使用发布前构建的回滚包，把流量切到上一个稳定版本。\n\n"
        "回滚完成后必须写事故记录，记录回滚耗时。\n", encoding="utf-8")
    return root


@pytest.fixture()
def kb(tmp_path):
    base = KnowledgeBase(tmp_path / "kb.db")
    yield base
    base.close()


# -- import and chunking ---------------------------------------------------


def test_import_creates_documents_and_chunks(kb, corpus):
    result = kb.import_directory(corpus)
    assert sorted(result["added"]) == ["a.md", "b.md"]
    stats = kb.stats()
    assert stats["documents"] == 2 and stats["chunks"] >= 2
    assert all(c.doc_id and c.chunk_id and c.sha256 for c in kb.chunks())


def test_line_ranges_point_at_the_real_text(kb, corpus):
    kb.import_directory(corpus)
    for chunk in kb.chunks():
        lines = (corpus / chunk.rel_path).read_text(encoding="utf-8").splitlines()
        span = "\n".join(lines[chunk.start_line - 1:chunk.end_line])
        first = chunk.text.splitlines()[0]
        assert first in span, f"{chunk.chunk_id} does not start at its recorded line"


def test_reimporting_identical_bytes_changes_nothing(kb, corpus):
    kb.import_directory(corpus)
    before = (kb.stats()["chunks"], kb.documents()[0]["sha256"])
    again = kb.import_directory(corpus)
    assert again == {"added": [], "updated": [], "skipped": ["a.md", "b.md"]}
    assert (kb.stats()["chunks"], kb.documents()[0]["sha256"]) == before


def test_updating_a_document_supersedes_old_chunks(kb, corpus):
    kb.import_directory(corpus)
    old_ids = {c.chunk_id for c in kb.chunks() if c.rel_path == "a.md"}
    (corpus / "a.md").write_text(
        "# 发布流程\n\n灰度第一阶段只放 2% 的流量，观察 45 分钟。\n", encoding="utf-8")

    assert kb.import_directory(corpus)["updated"] == ["a.md"]
    active = {c.chunk_id for c in kb.chunks() if c.rel_path == "a.md"}
    assert not (active & old_ids), "old chunks must not stay retrievable"
    assert kb.stats()["superseded_chunks"] >= len(old_ids)
    text = " ".join(c.text for c in kb.chunks() if c.rel_path == "a.md")
    assert "2%" in text and "5%" not in text


def test_split_document_keeps_overlap_and_order():
    text = "\n\n".join(f"段落{i}：" + "内容" * 120 for i in range(4))
    chunks = split_document(text, target=300, overlap_paragraphs=1)
    assert len(chunks) >= 2
    for (s1, e1, _), (s2, e2, _) in zip(chunks, chunks[1:]):
        assert s2 <= e1, "consecutive chunks must overlap or abut, never skip"


# -- retrieval -------------------------------------------------------------


def test_bm25_finds_the_document_with_the_keyword(kb, corpus):
    kb.import_directory(corpus)
    hits = kb.search_bm25("灰度第一阶段放多少流量", k=2)
    assert hits and hits[0].rel_path == "a.md"
    assert hits[0].score > 0


def test_stub_embedding_changes_the_ranking(kb, corpus):
    kb.import_directory(corpus)
    # the stub only knows three words, so it can order by their presence
    emb = StubEmbedder(["回滚包", "灰度", "事故记录"])
    hits = kb.search_embedding("事故记录", emb, k=2)
    assert hits and hits[0].rel_path == "b.md"
    assert emb.kind == "stub"


def test_hybrid_reports_both_channels(kb, corpus):
    kb.import_directory(corpus)
    emb = StubEmbedder(["回滚包", "灰度", "事故记录"])
    hits = kb.search_hybrid("灰度流量", emb, k=2)
    assert hits
    assert set(hits[0].channels) == {"bm25", "stub"}
    # an item ranked by both channels should not lose to one ranked by a single one
    assert hits[0].channels["bm25"] > 0


def test_lexical_embedder_is_labelled_lexical():
    emb = build_embedder("hash")
    assert emb.kind == "lexical" and "lexical" in emb.name
    assert isinstance(emb, LexicalHashEmbedder)


def test_embedding_requires_the_model_and_says_so():
    emb = FastEmbedEmbedder()
    assert emb.kind == "semantic"
    try:
        import fastembed  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="fastembed is not installed"):
            emb.dim
    else:
        pytest.skip("fastembed is installed; the missing-dependency path is untested")


def test_cosine_rejects_mismatched_vectors():
    with pytest.raises(ValueError):
        cosine([1.0, 0.0], [1.0])


# -- answers and citations -------------------------------------------------


def test_answer_citations_point_at_retrieved_chunks(kb, corpus):
    kb.import_directory(corpus)
    record = kb.answer("回滚结束后需要留下什么记录", k=2, mode="bm25")
    retrieved = {h["chunk_id"] for h in record["hits"]}
    assert record["citations"], "an answerable question should cite something"
    assert set(record["citations"]) <= retrieved
    assert record["source"] == "self_check"
    assert record["citation_lines"]


def test_answer_abstains_when_nothing_matches(kb, corpus):
    """Only a disjoint query is refused: token overlap cannot decide answerability.

    Measured on the sample corpus, an *unanswerable* question can share more of its
    tokens with the best chunk (0.53) than an answerable one (0.37), so this
    stand-in is limited to the near-zero-overlap case. Reported, not tuned away.
    """
    kb.import_directory(corpus)
    record = kb.answer("量子纠缠的退相干时间是多少", k=2, mode="bm25")
    assert record["abstained"] is True
    assert record["citations"] == []


def test_overlap_alone_cannot_detect_a_missing_answer(kb, corpus):
    """A topically related but unanswerable question is answered anyway.

    This is the documented limit of the extractive stand-in; a model is required
    to decide that the evidence does not contain the answer.
    """
    kb.import_directory(corpus)
    record = kb.answer("回滚流程的审批人是谁", k=2, mode="bm25")
    assert record["abstained"] is False
    assert record["citations"], "it cites its best chunk, which is the honest failure"


def test_a_fabricated_marker_is_flagged(kb, corpus):
    """A model answer may cite [9]; if only 2 chunks were retrieved, that is a lie."""

    class _FibbingLLM:
        def complete(self, messages, **kwargs):
            from evomemory.llm import LLMResponse
            return LLMResponse(text="答案见 [9]。", input_tokens=1, output_tokens=1,
                               model="stub")

    kb.import_directory(corpus)
    record = kb.answer("回滚结束后需要留下什么记录", k=2, mode="bm25",
                       llm=_FibbingLLM())
    assert record["abstained"] is True
    assert "未检索到的来源" in record["answer"]


def test_query_log_records_what_was_retrieved(kb, corpus):
    kb.import_directory(corpus)
    kb.answer("灰度第一阶段放多少流量", k=2, mode="bm25")
    log = kb.query_log(limit=1)
    assert log and log[0]["retrieved_json"] and log[0]["mode"] == "bm25"


# -- boundary with the experience memory -----------------------------------


def test_knowledge_base_and_experience_store_do_not_share_tables(tmp_path):
    from evomemory.db import Database

    kb = KnowledgeBase(tmp_path / "kb.db")
    exp_db = Database(tmp_path / "exp.db")
    kb_tables = {r[0] for r in kb.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    exp_tables = {r[0] for r in exp_db.query(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    kb.close()
    exp_db.close()

    assert "experiences" not in kb_tables and "kb_chunks" not in exp_tables
    assert {"kb_documents", "kb_chunks"} <= kb_tables
    assert "experiences" in exp_tables
