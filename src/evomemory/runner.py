"""Experiment runner: A / B / C / D over the same tasks and the same budget.

Each mode gets its own database and evidence store, so no mode can inherit
another's memory. The task list is identical across modes, generated once.

Two LLM backends:

* ``--llm scripted`` — the deterministic offline policy. Results are stamped
  ``harness_selfcheck`` and are evidence about the pipeline, not about models.
* ``--llm api``      — the configured chat model. Results are stamped
  ``live_model`` and consume the API budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .bench import BenchTask, build_dataset, dataset_summary, rules_brief
from .config import Settings
from .db import Database, finish_run, record_run
from .evidence import EvidenceStore
from .harness import Harness, MemoryGate, RunConfig
from .lifecycle import ExperienceExtractor, ExperienceReconstructor, MemoryCritic
from .llm import BudgetedLLM, DeepSeekClient, ScriptedLLM
from .memory import ExperienceStore
from .metrics import ModeMetrics, compare_modes, compute_metrics, evidence_grounding_accuracy, mde_binary
from .models import MemoryMode
from .policy import ScriptedOpsPolicy


ALL_MODES = [MemoryMode.A_NO_MEMORY, MemoryMode.B_VECTOR_MEMORY,
             MemoryMode.C_MULTIMODAL_MEMORY, MemoryMode.D_FULL_LIFECYCLE]


@dataclass
class ModeRun:
    mode: str
    metrics: ModeMetrics
    results: List[Dict[str, Any]] = field(default_factory=list)
    db_path: str = ""


def make_llm(kind: str, settings: Settings, calls: int, tokens: int,
             db: Database) -> BudgetedLLM:
    if kind == "api":
        settings.require_api()
        client = DeepSeekClient(settings.api_key, settings.model, settings.base_url)
        return BudgetedLLM(client, calls, tokens, db)
    policy = ScriptedOpsPolicy()
    return BudgetedLLM(ScriptedLLM(policy), calls, tokens, db)


def run_mode(mode: MemoryMode, tasks: Sequence[BenchTask], settings: Settings,
             llm_kind: str, out_dir: Path, max_calls: int,
             max_tokens: int) -> ModeRun:
    mode_dir = out_dir / f"mode_{mode.value}"
    mode_dir.mkdir(parents=True, exist_ok=True)
    db = Database(mode_dir / "memory.db")
    evidence = EvidenceStore(mode_dir / "evidence", db)
    run_id = f"{mode.value}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    record_run(db, run_id, mode.value, settings.describe())

    llm = make_llm(llm_kind, settings, max_calls, max_tokens, db)
    gate = MemoryGate(ExperienceStore(db), evidence)
    harness = Harness(
        db=db, evidence=evidence, llm=llm, gate=gate,
        extractor=ExperienceExtractor(evidence, llm if llm_kind == "api" else None),
        critic=MemoryCritic(llm if llm_kind == "api" else None),
        reconstructor=ExperienceReconstructor(llm if llm_kind == "api" else None),
        config=RunConfig(mode=mode),
    )

    results = []
    known_before: Dict[str, int] = {}
    for bt in tasks:
        family = bt.task.family
        known_before[bt.task.task_id] = sum(
            1 for e in gate.active_experiences() if e.family == family)
        result = harness.run_task(bt, run_id)
        results.append(result)

    finish_run(db, run_id)
    metrics = compute_metrics(db, results, mode.value,
                              {bt.task.task_id: bt.task.family for bt in tasks},
                              known_before)
    grounding = evidence_grounding_accuracy(db, evidence.exists)
    metrics.extra.update({
        "llm": llm.summary(),
        "evidence_grounding": grounding,
        "memory_counts": _memory_counts(db),
    })
    mode_run = ModeRun(
        mode=mode.value, metrics=metrics,
        results=[r.to_dict() for r in results],
        db_path=str(mode_dir / "memory.db"),
    )
    db.close()
    return mode_run


def _memory_counts(db: Database) -> Dict[str, int]:
    rows = db.query("SELECT status, COUNT(*) AS n FROM experiences GROUP BY status")
    return {r["status"]: int(r["n"]) for r in rows}


def run_experiment(settings: Settings, modes: Sequence[MemoryMode], llm_kind: str,
                   per_family: int = 4, seed: int = 7,
                   max_calls: Optional[int] = None,
                   max_tokens: Optional[int] = None,
                   out_root: Optional[Path] = None) -> Dict[str, Any]:
    tasks = build_dataset(seed=seed, per_family=per_family)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (out_root or settings.artifacts_dir) / f"run_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    budget_calls = max_calls if max_calls is not None else settings.max_calls
    budget_tokens = max_tokens if max_tokens is not None else settings.max_tokens

    mode_runs: List[ModeRun] = []
    for mode in modes:
        print(f"[run] mode {mode.value} ({llm_kind}) over {len(tasks)} tasks", flush=True)
        mode_runs.append(run_mode(mode, tasks, settings, llm_kind, out_dir,
                                  budget_calls, budget_tokens))

    payload = {
        "generated_at": stamp,
        "evidence_class": ("harness_selfcheck" if llm_kind == "scripted"
                           else "live_model"),
        "warning": ("Offline scripted policy: these numbers validate the harness, "
                    "not the effect of memory on a real model."
                    if llm_kind == "scripted" else
                    "Live model run; see llm_calls table for exact usage."),
        "settings": settings.describe(),
        "dataset": dataset_summary(tasks),
        "design": {
            "n_tasks": len(tasks),
            "cells": len(tasks),
            "mde_pp_at_this_n": mde_binary(len(tasks)),
            "note": ("With this many cells only large effects are detectable; "
                     "a null result here is not evidence of no effect."),
        },
        "modes": [mr.metrics.to_dict() for mr in mode_runs],
        "deltas_vs_A": compare_modes([mr.metrics for mr in mode_runs]),
        "per_task": {mr.mode: mr.results for mr in mode_runs},
        "rules": {f: rules_brief(next(t for t in tasks if t.family == f))
                  for f in {t.family for t in tasks}},
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(_render_summary(payload), encoding="utf-8")
    print(f"[run] wrote {out_dir / 'results.json'}")
    return payload


def _render_summary(payload: Dict[str, Any]) -> str:
    lines = [
        f"# Experiment {payload['generated_at']}",
        "",
        f"Evidence class: **{payload['evidence_class']}**",
        "",
        f"> {payload['warning']}",
        "",
        f"Tasks: {payload['design']['n_tasks']}  ·  "
        f"MDE at this n: ±{payload['design']['mde_pp_at_this_n']} pp  ·  "
        f"model: {payload['settings'].get('model') or 'scripted'}",
        "",
        "| mode | success | retrieval hit | neg. transfer | tokens/task | tool calls | admitted |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in payload["modes"]:
        lines.append(
            f"| {m['mode']} | {m['success_rate']:.2%} ({m['n_success']}/{m['n_tasks']}) "
            f"| {m['retrieval_hit_rate']:.2%} ({m['n_retrieval_hit']}/{m['n_eligible']}) "
            f"| {m['negative_transfer_rate']:.2%} | {m['mean_tokens_per_task']:.1f} "
            f"| {m['mean_tool_calls']:.2f} | {m['experiences_admitted']} |")
    lines.append("")
    if payload.get("deltas_vs_A"):
        lines.append("Deltas vs mode A:")
        lines.append("")
        for mode, d in payload["deltas_vs_A"].items():
            lines.append(f"- **{mode}**: success {d['delta_success_rate']:+.2%}, "
                         f"negative transfer {d['delta_negative_transfer']:+.2%}, "
                         f"tokens/task {d['delta_tokens_per_task']:+.1f}")
    lines.append("")
    lines.append("Rules per family (never shown to the agent):")
    lines.append("")
    for fam, rule in payload["rules"].items():
        lines.append(f"- `{fam}`: {rule}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the EvoMemory benchmark.")
    parser.add_argument("--modes", default="ABCD",
                        help="subset of ABCD (default: all)")
    parser.add_argument("--llm", choices=["scripted", "api"], default="scripted")
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    chosen = [m for m in ALL_MODES if m.value in args.modes.upper()]
    if not chosen:
        print("no valid modes selected", file=sys.stderr)
        return 2
    if args.llm == "api" and settings.dry_run:
        print("refusing to run --llm api without DEEPSEEK_API_KEY and DEEPSEEK_MODEL "
              "in .env", file=sys.stderr)
        return 2

    run_experiment(settings, chosen, args.llm, per_family=args.per_family,
                   seed=args.seed, max_calls=args.max_calls,
                   max_tokens=args.max_tokens,
                   out_root=Path(args.out) if args.out else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
