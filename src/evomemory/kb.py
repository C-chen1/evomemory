"""A small document knowledge base: import, chunk, retrieve, answer with citations.

Scope is deliberately narrow — one process, one SQLite file, three retrieval modes
(BM25, real embeddings, RRF hybrid). Anything larger belongs in a service, not here.

Two things this module refuses to blur:

* **documents are facts, experiences are not.** The knowledge base never stores
  learned rules, and the experience store never stores document text. They live in
  different tables and different files.
* **lexical is not semantic.** Only retrieval produced by a neural embedding model
  is labelled ``semantic``; the feature-hashing baseline is labelled ``lexical``
  everywhere it appears, including in the evaluation output.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .embeddings import Embedder, cosine, vector_digest
from .retrieval import BM25Index, rrf, tokenize

#: ~400 Chinese characters per chunk is roughly 250-300 tokens: small enough
#: to keep the answer prompt focused, large enough to hold a full procedure.
TARGET_CHARS = 400
MIN_CHUNK_CHARS = 80
SUPPORTED_SUFFIXES = {".md", ".txt"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_documents (
    doc_id      TEXT PRIMARY KEY,
    rel_path    TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL DEFAULT '',
    sha256      TEXT NOT NULL,
    rev         INTEGER NOT NULL DEFAULT 1,
    imported_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kb_chunks (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    rev        INTEGER NOT NULL,
    seq        INTEGER NOT NULL,
    text       TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    status     TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON kb_chunks(doc_id, status);
CREATE TABLE IF NOT EXISTS kb_vectors (
    chunk_id  TEXT NOT NULL,
    embedder  TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    digest    TEXT NOT NULL,
    vector    TEXT NOT NULL,
    PRIMARY KEY (chunk_id, embedder)
);
CREATE TABLE IF NOT EXISTS kb_queries (
    query_id      TEXT PRIMARY KEY,
    question      TEXT NOT NULL,
    mode          TEXT NOT NULL,
    retrieved_json TEXT NOT NULL,
    answer        TEXT NOT NULL DEFAULT '',
    citations_json TEXT NOT NULL DEFAULT '[]',
    abstained     INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);
"""


# --------------------------------------------------------------------------
# data types
# --------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    rel_path: str
    title: str
    rev: int
    seq: int
    text: str
    sha256: str
    start_line: int
    end_line: int
    #: 1-based marker used in the answer prompt and in the citation check
    marker: int = 0
    score: float = 0.0
    channels: Dict[str, float] = field(default_factory=dict)

    def citation(self) -> str:
        return f"[{self.marker}] {self.rel_path}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> Dict[str, Any]:
        return {"chunk_id": self.chunk_id, "doc_id": self.doc_id,
                "rel_path": self.rel_path, "seq": self.seq,
                "start_line": self.start_line, "end_line": self.end_line,
                "score": round(self.score, 6),
                "marker": self.marker, "channels": self.channels}


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------


def _paragraphs(text: str) -> List[Tuple[int, int, str]]:
    """(start_line, end_line, paragraph) with 1-based inclusive line numbers."""
    out: List[Tuple[int, int, str]] = []
    buf: List[str] = []
    start = 0
    for i, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            if not buf:
                start = i
            buf.append(line)
        elif buf:
            out.append((start, i - 1, "\n".join(buf)))
            buf = []
    if buf:
        out.append((start, len(text.splitlines()), "\n".join(buf)))
    return out


def split_document(text: str, target: int = TARGET_CHARS,
                   overlap_paragraphs: int = 1) -> List[Tuple[int, int, str]]:
    """Pack paragraphs into chunks of roughly ``target`` characters.

    Overlap is done at paragraph granularity so that line ranges stay exact — an
    approximate character overlap would make ``start_line`` a lie.
    """
    paras = _paragraphs(text)
    if not paras:
        return []
    chunks: List[Tuple[int, int, str]] = []
    current: List[Tuple[int, int, str]] = []
    size = 0
    for para in paras:
        if current and size + len(para[2]) > target and size >= MIN_CHUNK_CHARS:
            chunks.append((current[0][0], current[-1][1],
                           "\n\n".join(p[2] for p in current)))
            carry = current[-overlap_paragraphs:] if overlap_paragraphs else []
            current = list(carry)
            size = sum(len(p[2]) for p in current)
        current.append(para)
        size += len(para[2])
    if current:
        chunks.append((current[0][0], current[-1][1],
                       "\n\n".join(p[2] for p in current)))
    return chunks


# --------------------------------------------------------------------------
# knowledge base
# --------------------------------------------------------------------------


class KnowledgeBase:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # -- import ----------------------------------------------------------
    def import_directory(self, root: str | Path) -> Dict[str, Any]:
        root = Path(root)
        added, updated, skipped = [], [], []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            rel = str(path.relative_to(root)).replace("\\", "/")
            status = self.import_file(path, rel_path=rel)
            {"added": added, "updated": updated, "skipped": skipped}[status].append(rel)
        return {"added": added, "updated": updated, "skipped": skipped}

    def import_file(self, path: str | Path, rel_path: Optional[str] = None) -> str:
        path = Path(path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        rel = rel_path or path.name
        text = raw.decode("utf-8", errors="replace")

        row = self.conn.execute(
            "SELECT doc_id, sha256, rev FROM kb_documents WHERE rel_path = ?",
            (rel,)).fetchone()
        if row is not None and row["sha256"] == digest:
            return "skipped"                      # same bytes, same content

        rev = (row["rev"] + 1) if row is not None else 1
        doc_id = row["doc_id"] if row is not None else hashlib.sha256(
            rel.encode("utf-8")).hexdigest()[:16]
        title = _title_of(text, rel)
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with self._transaction():
            if row is not None:
                # The old revision stays queryable for audit but never retrievable.
                self.conn.execute(
                    "UPDATE kb_chunks SET status='superseded' WHERE doc_id=?", (doc_id,))
                self.conn.execute(
                    "UPDATE kb_documents SET sha256=?, rev=?, title=?, imported_at=? "
                    "WHERE doc_id=?", (digest, rev, title, now, doc_id))
            else:
                self.conn.execute(
                    "INSERT INTO kb_documents(doc_id, rel_path, title, sha256, rev, "
                    "imported_at) VALUES (?,?,?,?,?,?)",
                    (doc_id, rel, title, digest, rev, now))

            for seq, (start, end, body) in enumerate(split_document(text)):
                chunk_id = f"{doc_id}#{rev}#{seq}"
                self.conn.execute(
                    "INSERT INTO kb_chunks(chunk_id, doc_id, rev, seq, text, sha256, "
                    "start_line, end_line, status, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,'active',?)",
                    (chunk_id, doc_id, rev, seq, body,
                     hashlib.sha256(body.encode("utf-8")).hexdigest(),
                     start, end, now))
        return "added" if row is None else "updated"

    # -- reading ---------------------------------------------------------
    def chunks(self) -> List[Chunk]:
        rows = self.conn.execute(
            "SELECT c.*, d.rel_path, d.title FROM kb_chunks c "
            "JOIN kb_documents d ON d.doc_id = c.doc_id "
            "WHERE c.status='active' ORDER BY d.rel_path, c.seq").fetchall()
        return [Chunk(chunk_id=r["chunk_id"], doc_id=r["doc_id"],
                      rel_path=r["rel_path"], title=r["title"], rev=r["rev"],
                      seq=r["seq"], text=r["text"], sha256=r["sha256"],
                      start_line=r["start_line"], end_line=r["end_line"])
                for r in rows]

    def documents(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM kb_documents ORDER BY rel_path")]

    def stats(self) -> Dict[str, Any]:
        return {
            "documents": self.conn.execute(
                "SELECT COUNT(*) FROM kb_documents").fetchone()[0],
            "chunks": self.conn.execute(
                "SELECT COUNT(*) FROM kb_chunks WHERE status='active'").fetchone()[0],
            "superseded_chunks": self.conn.execute(
                "SELECT COUNT(*) FROM kb_chunks WHERE status='superseded'").fetchone()[0],
            "embedded_chunks": self.conn.execute(
                "SELECT COUNT(DISTINCT chunk_id) FROM kb_vectors").fetchone()[0],
        }

    # -- embeddings ------------------------------------------------------
    def build_vectors(self, embedder: Embedder, force: bool = False) -> int:
        chunks = self.chunks()
        if not chunks:
            return 0
        have = {r["chunk_id"] for r in self.conn.execute(
            "SELECT chunk_id FROM kb_vectors WHERE embedder = ?", (embedder.name,))}
        todo = [c for c in chunks if force or c.chunk_id not in have]
        if not todo:
            return 0
        vectors = embedder.embed([c.text for c in todo])
        with self._transaction():
            for chunk, vec in zip(todo, vectors):
                self.conn.execute(
                    "INSERT OR REPLACE INTO kb_vectors(chunk_id, embedder, dim, digest, "
                    "vector) VALUES (?,?,?,?,?)",
                    (chunk.chunk_id, embedder.name, len(vec), vector_digest(vec),
                     json.dumps([round(v, 6) for v in vec])))
        return len(todo)

    def _vectors(self, embedder: Embedder) -> Dict[str, List[float]]:
        rows = self.conn.execute(
            "SELECT chunk_id, vector FROM kb_vectors WHERE embedder = ?",
            (embedder.name,)).fetchall()
        return {r["chunk_id"]: json.loads(r["vector"]) for r in rows}

    # -- retrieval -------------------------------------------------------
    def search_bm25(self, query: str, k: int = 5) -> List[Chunk]:
        chunks = self.chunks()
        if not chunks:
            return []
        index = BM25Index({c.chunk_id: c.text for c in chunks})
        by_id = {c.chunk_id: c for c in chunks}
        hits = []
        for chunk_id, score in index.search(query, limit=k):
            chunk = by_id[chunk_id]
            chunk.score = score
            chunk.channels = {"bm25": round(score, 6)}
            hits.append(chunk)
        return hits

    def search_embedding(self, query: str, embedder: Embedder, k: int = 5) -> List[Chunk]:
        chunks = self.chunks()
        if not chunks:
            return []
        self.build_vectors(embedder)
        vectors = self._vectors(embedder)
        qvec = embedder.embed([query])[0]
        scored = []
        for chunk in chunks:
            vec = vectors.get(chunk.chunk_id)
            if vec is None:
                continue
            scored.append((chunk, cosine(qvec, vec)))
        scored.sort(key=lambda x: (-x[1], x[0].chunk_id))
        out = []
        for chunk, score in scored[:k]:
            chunk.score = score
            chunk.channels = {embedder.kind: round(score, 6)}
            out.append(chunk)
        return out

    def search_hybrid(self, query: str, embedder: Optional[Embedder], k: int = 5,
                      per_channel: int = 20) -> List[Chunk]:
        if embedder is None:
            raise ValueError("hybrid mode requires an embedding or lexical vector channel")
        bm25 = self.search_bm25(query, k=per_channel)
        rankings = {"bm25": [(c.chunk_id, c.score) for c in bm25]}
        by_id = {c.chunk_id: c for c in bm25}
        if embedder is not None:
            dense = self.search_embedding(query, embedder, k=per_channel)
            rankings[embedder.kind] = [(c.chunk_id, c.score) for c in dense]
            by_id.update({c.chunk_id: c for c in dense})
        fused = rrf(rankings)
        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        bm25_map = dict(rankings.get("bm25", []))
        dense_map = dict(rankings.get(embedder.kind, [])) if embedder else {}
        out = []
        for chunk_id, score in ordered:
            chunk = by_id[chunk_id]
            chunk.score = score
            chunk.channels = {"bm25": round(bm25_map.get(chunk_id, 0.0), 6)}
            if embedder is not None:
                chunk.channels[embedder.kind] = round(dense_map.get(chunk_id, 0.0), 6)
            out.append(chunk)
        return out

    # -- answering -------------------------------------------------------
    def answer(self, question: str, k: int = 4, mode: str = "bm25",
               embedder: Optional[Embedder] = None,
               llm=None) -> Dict[str, Any]:
        """Retrieve, then answer with citations that are checked against the hits.

        ``llm`` must be a ``BudgetedLLM`` when a real model is used; passing None
        selects the extractive stand-in, whose output is labelled ``self_check``.
        """
        hits = self.retrieve(question, k=k, mode=mode, embedder=embedder)
        for i, chunk in enumerate(hits, start=1):
            chunk.marker = i
        if not hits:
            record = self._record_query(question, mode, hits, "根据当前知识库无法确定。", [], abstained=True)
            record["source"] = "self_check" if llm is None else "model"
            return record

        if llm is None:
            text, cited, abstained = self._extractive_answer(question, hits)
            source = "self_check"
        else:
            text, cited, abstained = self._llm_answer(question, hits, llm)
            source = "model"
        record = self._record_query(question, mode, hits, text, cited, abstained)
        record["source"] = source
        return record

    def retrieve(self, question: str, k: int = 4, mode: str = "bm25",
                 embedder: Optional[Embedder] = None) -> List[Chunk]:
        if mode == "bm25":
            return self.search_bm25(question, k=k)
        if mode == "embedding":
            if embedder is None:
                raise ValueError("embedding mode needs an embedder")
            return self.search_embedding(question, embedder, k=k)
        if mode == "hybrid":
            return self.search_hybrid(question, embedder, k=k)
        raise ValueError(f"unknown mode {mode!r}")

    # -- internals -------------------------------------------------------
    #: A lexical stand-in can only ask "does the retrieved text even talk about the
    #: question's words?". Below this share of query tokens appearing in the best
    #: hit it abstains. It cannot tell "topically related but the answer is absent"
    #: — that needs a model and is reported as a known limit, not papered over.
    ABSTAIN_TOKEN_COVERAGE = 0.20

    def _extractive_answer(self, question: str,
                           hits: Sequence[Chunk]) -> Tuple[str, List[str], bool]:
        """Deterministic stand-in: quote the best chunk, cite it, else abstain.

        This is a *plumbing* check, not question answering. It is labelled
        ``self_check`` everywhere it surfaces.
        """
        if not hits or hits[0].score <= 0.0:
            return "根据当前知识库无法确定。", [], True
        query_tokens = set(tokenize(question))
        if query_tokens:
            best_tokens = set(tokenize(hits[0].text))
            coverage = len(query_tokens & best_tokens) / len(query_tokens)
            if coverage < self.ABSTAIN_TOKEN_COVERAGE:
                return ("根据当前知识库无法确定。（检索到的片段与问题几乎无词面重合）",
                        [], True)
        best = hits[0]
        first_sentence = re.split(r"(?<=[。.!?])\s*", best.text.strip())[0][:260]
        return (f"根据文档 {best.citation()}：{first_sentence} [{best.marker}]",
                [best.chunk_id], False)

    def _llm_answer(self, question: str, hits: Sequence[Chunk],
                    llm) -> Tuple[str, List[str], bool]:
        context = "\n\n".join(f"[{c.marker}] ({c.rel_path}:{c.start_line}-{c.end_line})\n"
                              f"{c.text}" for c in hits)
        system = ("Answer the question using ONLY the numbered sources. Cite every "
                  "claim as [n]. If the sources do not support an answer, reply "
                  "exactly: INSUFFICIENT_EVIDENCE")
        resp = llm.complete(
            [{"role": "system", "content": system},
             {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}],
            purpose="kb-answer", prompt_version="kb-answer-v1", max_tokens=1200)
        text = (resp.text or "").strip()
        if text.upper().startswith("INSUFFICIENT_EVIDENCE"):
            return "根据当前知识库无法确定。", [], True
        markers = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
        cited = [c.chunk_id for c in hits if c.marker in markers]
        # A citation to a marker that is not in the retrieved set is a fabrication.
        valid = markers <= {c.marker for c in hits}
        if not valid:
            text += "  \n[警告] 回答引用了未检索到的来源，已标记为无效。"
            return text, cited, True
        if not cited:
            return text, [], True
        return text, cited, False

    def _record_query(self, question: str, mode: str, hits: Sequence[Chunk],
                      answer: str, cited: Sequence[str], abstained: bool) -> Dict[str, Any]:
        from datetime import datetime, timezone
        qid = hashlib.sha256(
            f"{question}|{mode}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]
        self.conn.execute(
            "INSERT INTO kb_queries(query_id, question, mode, retrieved_json, answer, "
            "citations_json, abstained, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (qid, question, mode,
             json.dumps([c.to_dict() for c in hits], ensure_ascii=False),
             answer, json.dumps(list(cited)), 1 if abstained else 0,
             datetime.now(timezone.utc).isoformat(timespec="seconds")))
        return {"query_id": qid, "question": question, "mode": mode,
                "hits": [c.to_dict() for c in hits], "answer": answer,
                "citations": list(cited), "abstained": abstained,
                "citation_lines": [c.citation() for c in hits if c.chunk_id in set(cited)]}

    def query_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM kb_queries ORDER BY created_at DESC LIMIT ?", (limit,))]

    def _transaction(self):
        conn = self.conn

        class _T:
            def __enter__(self_):
                conn.execute("BEGIN IMMEDIATE")

            def __exit__(self_, exc_type, exc, tb):
                conn.execute("ROLLBACK" if exc_type else "COMMIT")
                return False

        return _T()

    def close(self) -> None:
        self.conn.close()


def _title_of(text: str, rel: str) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()[:120]
        if line.strip():
            return line.strip()[:120]
    return Path(rel).stem
