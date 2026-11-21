"""Retrieval and answer evaluation for the document knowledge base.

Kept deliberately small: a fixed question file, a fixed corpus, and metrics that
can be recomputed from the stored query log. Nothing here tunes anything on the
questions it reports — the question set is fixed before the run and is not used to
adjust chunking, weights or prompts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .embeddings import Embedder
from .kb import Chunk, KnowledgeBase


def load_questions(path: str | Path) -> List[Dict[str, Any]]:
    questions = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            questions.append(json.loads(line))
    return questions


def covers(chunk: Chunk, gold: Dict[str, str]) -> bool:
    """A retrieved chunk counts for a gold item if it is from that document and
    contains the distinguishing phrase. Chunk ids are never written into the
    question file, so re-chunking does not invalidate the labels."""
    return (chunk.rel_path == gold["doc"] and gold["contains"] in chunk.text)


def evaluate_retrieval(kb: KnowledgeBase, questions: Sequence[Dict[str, Any]],
                       mode: str, embedder: Optional[Embedder], k: int = 4) -> Dict[str, Any]:
    rows = []
    hit_count = 0
    recall_sum = 0.0
    mrr_sum = 0.0
    answerable = [q for q in questions if q["answerable"]]
    for q in answerable:
        hits = kb.retrieve(q["question"], k=k, mode=mode, embedder=embedder)
        covered = []
        first_rank = 0
        for rank, chunk in enumerate(hits, start=1):
            for i, gold in enumerate(q["gold"]):
                if i not in covered and covers(chunk, gold):
                    covered.append(i)
                    if not first_rank:
                        first_rank = rank
        hit = bool(covered)
        recall = len(covered) / len(q["gold"]) if q["gold"] else 0.0
        hit_count += 1 if hit else 0
        recall_sum += recall
        mrr_sum += (1.0 / first_rank) if first_rank else 0.0
        rows.append({"qid": q["qid"], "mode": q["mode"], "question": q["question"],
                     "hits": [h.to_dict() for h in hits],
                     "covered_gold": covered, "hit": hit,
                     "recall": round(recall, 3), "first_gold_rank": first_rank})
    n = len(answerable)
    return {
        "mode": mode, "k": k, "n_answerable": n,
        "hit_at_k": round(hit_count / n, 3) if n else 0.0,
        "recall_at_k": round(recall_sum / n, 3) if n else 0.0,
        "mrr": round(mrr_sum / n, 3) if n else 0.0,
        "per_question": rows,
    }


def evaluate_answers(kb: KnowledgeBase, questions: Sequence[Dict[str, Any]],
                     mode: str, embedder: Optional[Embedder], llm=None,
                     k: int = 4) -> Dict[str, Any]:
    rows = []
    for q in questions:
        record = kb.answer(q["question"], k=k, mode=mode, embedder=embedder, llm=llm)
        retrieved_ids = {h["chunk_id"] for h in record["hits"]}
        cited = record["citations"]
        citation_valid = all(c in retrieved_ids for c in cited)
        # a citation is correct if the chunk it points at really covers a gold item
        by_id = {h["chunk_id"]: h for h in record["hits"]}
        citation_correct = False
        if q["answerable"] and cited:
            for chunk_id in cited:
                hit = by_id.get(chunk_id)
                if hit is None:
                    continue
                chunk = Chunk(chunk_id=chunk_id, doc_id=hit["doc_id"],
                              rel_path=hit["rel_path"], title="", rev=0,
                              seq=hit["seq"], text=next(
                                  (c.text for c in kb.chunks() if c.chunk_id == chunk_id), ""),
                              sha256="", start_line=hit["start_line"],
                              end_line=hit["end_line"])
                if any(covers(chunk, gold) for gold in q["gold"]):
                    citation_correct = True
        abstention_correct = (not q["answerable"]) and record["abstained"]
        rows.append({"qid": q["qid"], "mode": q["mode"], "answerable": q["answerable"],
                     "answer": record["answer"], "citations": cited,
                     "citation_valid": citation_valid,
                     "citation_correct": citation_correct,
                     "abstained": record["abstained"],
                     "abstention_correct": abstention_correct,
                     "source": record.get("source", "self_check")})
    answered = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    return {
        "mode": mode, "n": len(rows),
        "citation_valid_rate": round(
            sum(r["citation_valid"] for r in answered) / len(answered), 3) if answered else 0.0,
        "citation_correct_rate": round(
            sum(r["citation_correct"] for r in answered) / len(answered), 3) if answered else 0.0,
        "abstention_rate_on_unanswerable": round(
            sum(r["abstention_correct"] for r in unanswerable) / len(unanswerable), 3)
        if unanswerable else 0.0,
        "answer_source": rows[0]["source"] if rows else "n/a",
        "per_question": rows,
    }


def run_evaluation(kb: KnowledgeBase, questions: Sequence[Dict[str, Any]],
                   modes: Sequence[str], embedder: Optional[Embedder],
                   llm=None, k: int = 4) -> Dict[str, Any]:
    retrieval = {}
    for mode in modes:
        if mode == "embedding" or mode == "hybrid":
            if embedder is None:
                retrieval[mode] = {"mode": mode, "status": "not_run",
                                   "reason": "no embedder configured"}
                continue
        retrieval[mode] = evaluate_retrieval(kb, questions, mode, embedder, k=k)
    answers = evaluate_answers(kb, questions, "hybrid" if "hybrid" in modes else modes[0],
                               embedder, llm=llm, k=k)
    return {
        "corpus": kb.stats(),
        "embedder": getattr(embedder, "name", None),
        "embedder_kind": getattr(embedder, "kind", None),
        "retrieval": retrieval,
        "answers": answers,
        "notes": ("retrieval metrics come from the fixed question file; answer checks "
                  "verify citation validity and abstention, not factual correctness"),
    }
