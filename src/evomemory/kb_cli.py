"""Command line entry point for the document knowledge base.

    python -m evomemory.kb_cli import --corpus data/kb_corpus
    python -m evomemory.kb_cli search --query "灰度第一阶段放多少流量"
    python -m evomemory.kb_cli ask    --query "回滚结束后要留什么记录"
    python -m evomemory.kb_cli eval   --modes bm25,embedding,hybrid
    python -m evomemory.kb_cli demo

Offline by default: retrieval needs no model, and ``ask`` without ``--llm api``
uses the extractive stand-in, labelled ``self_check``. Paid answering must go
through the shared budget ledger, exactly like every other paid path.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]

from .embeddings import Embedder, build_embedder  # noqa: E402
from .kb import KnowledgeBase  # noqa: E402
from .kb_eval import load_questions, run_evaluation  # noqa: E402

DEFAULT_DB = ROOT / "data" / "kb.db"
DEFAULT_CORPUS = ROOT / "data" / "kb_corpus"
DEFAULT_QUESTIONS = ROOT / "benchmarks" / "kb_questions.jsonl"
BUDGET_DB = ROOT / "artifacts" / "stage3" / "round_budget.db"


def _embedder(kind: Optional[str]) -> Optional[Embedder]:
    return build_embedder(kind, cache_dir=ROOT / "cache" / "huggingface" / "rag") if kind else None


def _print_hits(hits: Sequence[Dict[str, Any]], show_text: int = 60) -> None:
    for hit in hits:
        print(f"  [{hit['marker']}] {hit['score']:.4f} "
              f"{hit['rel_path']}:{hit['start_line']}-{hit['end_line']}  "
              f"channels={hit['channels']}")


def cmd_import(args) -> int:
    kb = KnowledgeBase(args.db)
    result = kb.import_directory(args.corpus)
    print(json.dumps({"corpus": str(args.corpus), "result": result,
                      "stats": kb.stats()}, ensure_ascii=False, indent=1))
    if args.embedder:
        emb = _embedder(args.embedder)
        built = kb.build_vectors(emb, force=args.rebuild_vectors)
        print(f"embedded {built} chunks with {emb.name} ({emb.kind})")
    kb.close()
    return 0


def cmd_search(args) -> int:
    kb = KnowledgeBase(args.db)
    emb = _embedder(args.embedder)
    hits = kb.retrieve(args.query, k=args.k, mode=args.mode, embedder=emb)
    for i, chunk in enumerate(hits, start=1):
        chunk.marker = i
    print(f"query: {args.query}\n  mode={args.mode} k={args.k} "
          f"embedder={getattr(emb, 'name', None)}")
    _print_hits([h.to_dict() for h in hits])
    kb.close()
    return 0


def _make_llm(args):
    if args.llm != "api":
        return None
    from .config import Settings
    from .llm import BudgetedLLM, DeepSeekClient
    from .shared_budget import SharedBudget

    settings = Settings.from_env(root=ROOT)
    settings.require_api()
    shared = SharedBudget(BUDGET_DB, 450, 1_000_000)
    client = DeepSeekClient(settings.api_key, settings.model, settings.base_url)
    return BudgetedLLM(client, args.max_calls, 200_000, shared_budget=shared)


def cmd_ask(args) -> int:
    if args.llm == "api" and not args.shared_stage3_budget:
        print("refusing to run --llm api without --shared-stage3-budget: paid calls "
              "must be recorded in the round ledger.", file=sys.stderr)
        return 2
    kb = KnowledgeBase(args.db)
    llm = _make_llm(args)
    record = kb.answer(args.query, k=args.k, mode=args.mode,
                       embedder=_embedder(args.embedder), llm=llm)
    print(f"query  : {record['question']}")
    print(f"mode   : {record['mode']}  source={record.get('source')}")
    print("sources:")
    for line in record["citation_lines"] or ["(none cited)"]:
        print(f"  {line}")
    print(f"\nanswer : {record['answer']}")
    print(f"abstain: {record['abstained']}")
    kb.close()
    return 0


def _out_dir(base: Optional[str], label: str) -> Path:
    stamp = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
             + "-" + __import__("uuid").uuid4().hex[:6])
    out = (Path(base) if base else ROOT / "artifacts" / "kb") / f"{label}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def cmd_eval(args) -> int:
    out = _out_dir(args.out, "eval")
    db_path = out / "kb.db"
    kb = KnowledgeBase(db_path)
    kb.import_directory(args.corpus)
    emb = _embedder(args.embedder)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    questions = load_questions(args.questions)
    result = run_evaluation(kb, questions, modes, emb, k=args.k)
    result["corpus_dir"] = str(args.corpus)
    result["questions"] = str(args.questions)
    (out / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"corpus: {result['corpus']}")
    print(f"embedder: {result['embedder']} ({result['embedder_kind']})")
    for mode, row in result["retrieval"].items():
        if row.get("status") == "not_run":
            print(f"  {mode:9s} not run ({row['reason']})")
            continue
        print(f"  {mode:9s} hit@{row['k']}={row['hit_at_k']:.3f} "
              f"recall={row['recall_at_k']:.3f} mrr={row['mrr']:.3f}")
    a = result["answers"]
    print(f"  citations valid={a['citation_valid_rate']:.3f} "
          f"correct={a['citation_correct_rate']:.3f} "
          f"abstain_on_unanswerable={a['abstention_rate_on_unanswerable']:.3f} "
          f"(source={a['answer_source']})")
    print(f"results -> {out / 'results.json'}")
    kb.close()
    return 0


def cmd_demo(args) -> int:
    out = _out_dir(args.out, "demo")
    kb = KnowledgeBase(out / "kb.db")
    imported = kb.import_directory(args.corpus)
    print("=" * 74)
    print("EvoMemory knowledge-base demo — retrieval is real; answers are self_check")
    print("=" * 74)
    print(f"\nOutput: {out}\n")
    print(f"imported: {len(imported['added'])} documents, "
          f"{kb.stats()['chunks']} chunks")

    emb = _embedder(args.embedder)
    if emb is not None:
        built = kb.build_vectors(emb)
        print(f"embedded: {built} chunks with {emb.name} ({emb.kind})")

    queries = ["灰度第一阶段放多少流量？",
               "出错以后怎么把服务退回上一个可用版本？",
               "数据库连接池应该配置多大？"]
    for query in queries:
        print(f"\n--- {query}")
        for i, chunk in enumerate(kb.retrieve(query, k=3, mode=args.mode,
                                              embedder=emb), start=1):
            chunk.marker = i
            print(f"  [{i}] {chunk.score:.4f} {chunk.rel_path}:"
                  f"{chunk.start_line}-{chunk.end_line}")
        record = kb.answer(query, k=3, mode=args.mode, embedder=emb, llm=None)
        print(f"  answer ({record.get('source')}): {record['answer'][:120]}")
        print(f"  citations: {record['citation_lines'] or ['(none)']}")

    questions = load_questions(args.questions)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    result = run_evaluation(kb, questions, modes, emb, k=args.k)
    (out / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n--- retrieval evaluation")
    for mode, row in result["retrieval"].items():
        if row.get("status") == "not_run":
            print(f"  {mode:9s} not run ({row['reason']})")
            continue
        print(f"  {mode:9s} hit@{row['k']}={row['hit_at_k']:.3f} "
              f"recall={row['recall_at_k']:.3f} mrr={row['mrr']:.3f}")
    print(f"results -> {out / 'results.json'}")
    kb.close()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="evomemory.kb_cli")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        # accept --db after the subcommand too; people type it both ways
        p.add_argument("--db", default=argparse.SUPPRESS)
        p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
        p.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
        p.add_argument("--embedder", choices=["fastembed", "hash"], default=None)
        p.add_argument("--mode", choices=["bm25", "embedding", "hybrid"],
                       default="bm25")
        p.add_argument("--modes", default="bm25,hybrid")
        p.add_argument("--k", type=int, default=4)
        p.add_argument("--out", default=None)

    p_import = sub.add_parser("import", help="import a document directory")
    common(p_import)
    p_import.add_argument("--rebuild-vectors", action="store_true")
    p_import.set_defaults(func=cmd_import, mode="bm25", modes="bm25")

    p_search = sub.add_parser("search", help="show retrieval results only")
    common(p_search)
    p_search.add_argument("--query", required=True)
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="answer with citations")
    common(p_ask)
    p_ask.add_argument("--query", required=True)
    p_ask.add_argument("--llm", choices=["scripted", "api"], default="scripted")
    p_ask.add_argument("--shared-stage3-budget", action="store_true")
    p_ask.add_argument("--max-calls", type=int, default=5)
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="run the fixed question set")
    common(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    p_demo = sub.add_parser("demo", help="one-shot offline demonstration")
    common(p_demo)
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
