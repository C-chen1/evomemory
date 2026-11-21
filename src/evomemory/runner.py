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
import uuid
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .bench import (BenchTask, apply_release, build_dataset, dataset_summary,
                    rules_brief)
from .config import Settings
from .db import Database, finish_run, loads, record_run
from .evidence import EvidenceStore
from .harness import Harness, MemoryGate, RunConfig, TaskRunResult
from .lifecycle import ExperienceExtractor, ExperienceReconstructor, MemoryCritic
from .llm import BudgetedLLM, BudgetExceeded, DeepSeekClient, LLMError, ScriptedLLM
from .memory import ExperienceStore
from .metrics import (
    ModeMetrics, by_cluster, compare_modes, compare_paired, compute_metrics,
    effective_independence, evidence_grounding_accuracy, mde_binary,
    paired_by_cluster,
)
from .models import MemoryMode, Outcome, OutcomeStatus, VerifiedBy
from .policy import ScriptedOpsPolicy


#: Modes where the critic/reconstructor are only invoked when a structural
#: check cannot settle applicability.
SELECTIVE_CRITIQUE_MODES = {MemoryMode.E_SELECTIVE}

ALL_MODES = [MemoryMode.A_NO_MEMORY, MemoryMode.B_VECTOR_MEMORY,
             MemoryMode.C_MULTIMODAL_MEMORY, MemoryMode.D_FULL_LIFECYCLE,
             MemoryMode.S_STRUCTURED, MemoryMode.G_STRUCTURED_GATE,
             MemoryMode.E_SELECTIVE]


@dataclass
class ModeRun:
    mode: str
    metrics: ModeMetrics
    results: List[Dict[str, Any]] = field(default_factory=list)
    db_path: str = ""


def make_llm(kind: str, settings: Settings, calls: int, tokens: int,
             db: Database, shared_budget=None) -> BudgetedLLM:
    if kind == "api":
        settings.require_api()
        client = DeepSeekClient(settings.api_key, settings.model, settings.base_url)
        return BudgetedLLM(client, calls, tokens, db, shared_budget=shared_budget)
    policy = ScriptedOpsPolicy()
    return BudgetedLLM(ScriptedLLM(policy), calls, tokens, db)


def _completed_result(db: Database, task_id: str, mode: str) -> Optional[TaskRunResult]:
    """Rebuild a TaskRunResult for a task that was settled in an earlier process.

    Resuming skips re-running a task, so its numbers must come back out of the
    database — otherwise the resumed metrics would silently cover fewer tasks than
    the run actually completed.
    """
    t = db.query_one(
        "SELECT trace_id, started_at FROM traces WHERE task_id = ? AND status = "
        "'completed' ORDER BY started_at DESC LIMIT 1", (task_id,))
    if t is None:
        return None
    o = db.query_one("SELECT * FROM outcomes WHERE trace_id = ?", (t["trace_id"],))
    if o is None:
        return None

    outcome = Outcome(
        status=OutcomeStatus(o["status"]),
        verified_by=VerifiedBy(o["verified_by"]),
        score=float(o["score"]),
        reason=o["reason"],
        checks=loads(o["checks_json"], []),
    )
    mem = db.query_one(
        "SELECT injected_json, retrieved_json, rejected_json FROM memory_events "
        "WHERE task_id = ? ORDER BY id DESC LIMIT 1", (task_id,))
    injected = loads(mem["injected_json"], []) if mem else []
    retrieved = loads(mem["retrieved_json"], []) if mem else []
    rejected = loads(mem["rejected_json"], []) if mem else []
    tok = db.query_one(
        "SELECT COALESCE(SUM(input_tokens),0) AS i, COALESCE(SUM(output_tokens),0) AS o "
        "FROM llm_calls WHERE task_id = ?", (task_id,))
    steps = db.query_one(
        "SELECT COUNT(*) AS n, COALESCE(SUM(CASE WHEN tool NOT IN "
        "('', 'finish', '__parse__') THEN 1 ELSE 0 END),0) AS tc "
        "FROM trace_steps WHERE trace_id = ?", (t["trace_id"],))
    written = db.query_one(
        "SELECT COUNT(*) AS n FROM experiences WHERE task_id = ?", (task_id,))
    admitted = db.query_one(
        "SELECT COUNT(*) AS n FROM experiences WHERE task_id = ? AND status IN "
        "('active','accepted','superseded','deprecated')", (task_id,))

    return TaskRunResult(
        task_id=task_id, mode=mode, trace_id=t["trace_id"], outcome=outcome,
        injected_ids=injected, rejected=rejected, retrieved=retrieved,
        tool_calls=int(steps["tc"]), steps=int(steps["n"]),
        memory_written=int(written["n"]), memory_admitted=int(admitted["n"]),
        tokens_in=int(tok["i"]), tokens_out=int(tok["o"]),
        reused_from_memory=bool(injected),
    )


def run_mode(mode: MemoryMode, tasks: Sequence[BenchTask], settings: Settings,
             llm_kind: str, out_dir: Path, max_calls: int,
             max_tokens: int, resume: bool = True,
             evolve: bool = False, shared_budget=None) -> ModeRun:
    mode_dir = out_dir / f"mode_{mode.value}"
    mode_dir.mkdir(parents=True, exist_ok=True)
    db = Database(mode_dir / "memory.db")
    evidence = EvidenceStore(mode_dir / "evidence", db)
    # Run ids used to be second-resolution, so a resume started in the same second
    # as the original collided on runs.run_id and crashed instead of resuming.
    run_id = (f"{mode.value}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
              f"-{uuid.uuid4().hex[:6]}")
    record_run(db, run_id, mode.value, settings.describe())

    mode_writes_memory = mode is not MemoryMode.A_NO_MEMORY
    llm = make_llm(llm_kind, settings, max_calls, max_tokens, db, shared_budget)
    gate = MemoryGate(ExperienceStore(db), evidence)
    harness = Harness(
        db=db, evidence=evidence, llm=llm, gate=gate,
        extractor=ExperienceExtractor(evidence, llm if llm_kind == "api" else None,
                                      strict_errors=shared_budget is not None),
        critic=MemoryCritic(llm if llm_kind == "api" else None,
                            selective=mode in SELECTIVE_CRITIQUE_MODES),
        reconstructor=ExperienceReconstructor(llm if llm_kind == "api" else None,
                                              selective=mode in SELECTIVE_CRITIQUE_MODES),
        config=RunConfig(mode=mode, evolves=evolve, abort_on_llm_error=shared_budget is not None),
    )

    # Resume within a mode's database: a live run can be longer than one shell
    # window, and re-running must not redo (and re-pay for) settled tasks.
    done: set[str] = set()
    needs_memory: set[str] = set()
    if resume:
        rows = db.query(
            "SELECT DISTINCT t.task_id AS task_id FROM traces t "
            "JOIN outcomes o ON o.trace_id = t.trace_id WHERE t.status = 'completed'")
        scored = {r["task_id"] for r in rows}
        for task_id in scored:
            if not mode_writes_memory or harness.memory_settled(task_id):
                done.add(task_id)
            else:
                needs_memory.add(task_id)
        if done or needs_memory:
            print(f"[run] mode {mode.value}: resuming, {len(done)} settled, "
                  f"{len(needs_memory)} scored but not yet learned from", flush=True)

    results = []
    known_before: Dict[str, int] = {}
    aborted: List[str] = []
    for bt in tasks:
        family = bt.task.family
        known_before[bt.task.task_id] = sum(
            1 for e in gate.active_experiences() if e.family == family)
        if bt.task.task_id in done or bt.task.task_id in needs_memory:
            prior_trace = db.query_one(
                "SELECT params_json FROM traces WHERE task_id = ? AND status = 'completed' "
                "ORDER BY started_at DESC LIMIT 1", (bt.task.task_id,))
            recorded = loads(prior_trace["params_json"], {}) if prior_trace else {}
            if (recorded.get("policy_scope") != bt.policy_scope or
                    recorded.get("feedback_split", "train") != bt.feedback_split):
                db.close()
                raise ValueError(f"resume scope or split differs for task {bt.task.task_id}")
        if bt.task.task_id in needs_memory:
            # scoring happened in an earlier process but extraction did not; run
            # only the write-back rather than re-executing (and re-charging) the task
            try:
                recovered = harness.settle_memory_for(bt, run_id)
            except BudgetExceeded as exc:
                print(f"[run] budget exhausted settling {bt.task.task_id}: {exc}",
                      file=sys.stderr)
                aborted.append(bt.task.task_id)
                break
            prior = _completed_result(db, bt.task.task_id, mode.value)
            if prior is not None:
                results.append(prior)
            continue
        if bt.task.task_id in done:
            prior = _completed_result(db, bt.task.task_id, mode.value)
            if prior is not None:
                results.append(prior)
            continue
        try:
            results.append(harness.run_task(bt, run_id))
        except (BudgetExceeded, LLMError) as exc:
            # The cap is a hard limit, not a suggestion. Stop cleanly and keep the
            # partial results instead of losing the whole run to an exception.
            print(f"[run] interrupted at task {bt.task.task_id}: {exc}",
                  file=sys.stderr)
            if shared_budget is not None:
                db.execute("UPDATE traces SET status='aborted' WHERE run_id=? AND task_id=?",
                           (run_id, bt.task.task_id))
            aborted.append(bt.task.task_id)
            break

    finish_run(db, run_id)
    drop = set(aborted)
    metrics = compute_metrics(db, [r for r in results if r.task_id not in drop],
                              mode.value,
                              {bt.task.task_id: bt.task.family for bt in tasks},
                              known_before)
    grounding = evidence_grounding_accuracy(db, evidence.exists)
    by_purpose = {r["purpose"]: int(r["n"]) for r in db.query(
        "SELECT purpose, COUNT(*) AS n FROM llm_calls GROUP BY purpose")}
    metrics.extra.update({
        "llm": llm.summary(),
        "llm_by_purpose": by_purpose,
        "critique_skipped": harness.critic.skipped,
        "reconstruct_skipped": harness.reconstructor.skipped,
        "evidence_grounding": grounding,
        "memory_counts": _memory_counts(db),
        "budget_aborted_tasks": aborted,
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
                   out_root: Optional[Path] = None,
                   run_dir: Optional[Path] = None,
                   families: Sequence[str] = ("escalation", "closure"),
                   evolve: bool = False,
                   drift_at: Optional[int] = None,
                   policy_scopes: Optional[Sequence[Optional[str]]] = None,
                   holdout_from: Optional[int] = None,
                   shared_budget=None) -> Dict[str, Any]:
    tasks = build_dataset(seed=seed, per_family=per_family, families=families,
                          drift_at=drift_at)
    # A policy release is declared by the caller, never inferred from outcomes.
    apply_release(tasks, scopes=policy_scopes, holdout_from=holdout_from)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(run_dir) if run_dir else (out_root or settings.artifacts_dir) / f"run_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # A live-model experiment rarely fits in one process: it is slow and it must
    # survive the shell's time limit. So a run directory can be extended one mode
    # at a time, and the results file accumulates rather than being overwritten.
    results_path = out_dir / "results.json"
    previous: Dict[str, Any] = {}
    if results_path.exists():
        try:
            previous = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}

    budget_calls = max_calls if max_calls is not None else settings.max_calls
    budget_tokens = max_tokens if max_tokens is not None else settings.max_tokens

    mode_runs: List[ModeRun] = []
    for mode in modes:
        print(f"[run] mode {mode.value} ({llm_kind}) over {len(tasks)} tasks", flush=True)
        mode_runs.append(run_mode(mode, tasks, settings, llm_kind, out_dir,
                                  budget_calls, budget_tokens, evolve=evolve,
                                  shared_budget=shared_budget))
        if shared_budget is not None and mode_runs[-1].metrics.extra.get("budget_aborted_tasks"):
            break

    merged_modes = {m["mode"]: m for m in previous.get("modes", [])}
    for mr in mode_runs:
        merged_modes[mr.mode] = mr.metrics.to_dict()
    merged_tasks = dict(previous.get("per_task", {}))
    for mr in mode_runs:
        merged_tasks[mr.mode] = mr.results

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
        "modes": [merged_modes[k] for k in sorted(merged_modes)],
        "deltas_vs_A": compare_modes([_as_metrics(m) for m in merged_modes.values()]),
        "paired_vs_A": _paired_block(merged_tasks),
        "by_cluster": _cluster_block(merged_tasks, tasks),
        "paired_by_cluster_vs_A": _paired_cluster_block(merged_tasks, tasks),
        "independence": effective_independence({t.task.task_id: t.family for t in tasks}),
        "drift_at": drift_at,
        "evolve": evolve,
        "per_task": merged_tasks,
        "rules": {f: rules_brief(next(t for t in tasks if t.family == f))
                  for f in {t.family for t in tasks}},
    }
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    (out_dir / "summary.md").write_text(_render_summary(payload), encoding="utf-8")
    print(f"[run] wrote {results_path}")
    return payload


def _as_metrics(d: Dict[str, Any]) -> ModeMetrics:
    """Rebuild a ModeMetrics from its serialised form (for cross-run comparison)."""
    fields = {k: v for k, v in d.items()
              if k in ModeMetrics.__dataclass_fields__ and k != "extra"}
    return ModeMetrics(**fields)


def _outcome_maps(per_task: Dict[str, Any]) -> Dict[str, Dict[str, bool]]:
    return {mode: {r["task_id"]: r["outcome"]["status"] == "success" for r in runs}
            for mode, runs in per_task.items()}


def _cluster_block(per_task: Dict[str, Any], tasks: Sequence[BenchTask]) -> Dict[str, Any]:
    families = {t.task.task_id: t.family for t in tasks}
    return {mode: by_cluster(m, families) for mode, m in _outcome_maps(per_task).items()}


def _paired_cluster_block(per_task: Dict[str, Any],
                          tasks: Sequence[BenchTask]) -> Dict[str, Any]:
    families = {t.task.task_id: t.family for t in tasks}
    maps = _outcome_maps(per_task)
    base = maps.get("A")
    if base is None:
        return {}
    return {mode: paired_by_cluster(base, m, families)
            for mode, m in maps.items() if mode != "A"}


def _paired_block(per_task: Dict[str, Any]) -> Dict[str, Any]:
    """Paired A-vs-X comparison over the tasks both modes actually ran."""
    base = per_task.get("A")
    if not base:
        return {}
    base_map = {r["task_id"]: r["outcome"]["status"] == "success" for r in base}
    out: Dict[str, Any] = {}
    for mode, runs in per_task.items():
        if mode == "A":
            continue
        cand_map = {r["task_id"]: r["outcome"]["status"] == "success" for r in runs}
        out[mode] = compare_paired(base_map, cand_map)
    return out


def _model_label(payload: Dict[str, Any]) -> str:
    """Report the model that actually produced the numbers, not whatever .env says."""
    models = []
    for m in payload.get("modes", []):
        llm = (m.get("llm") or {}).get("model")
        if llm:
            models.append(llm)
    return models[0] if models else (payload["settings"].get("model") or "—")


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
        f"model: {_model_label(payload)}",
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
    paired = payload.get("paired_vs_A") or {}
    if paired:
        lines.append("Paired comparison vs mode A (same tasks, McNemar exact test):")
        lines.append("")
        lines.append("| vs A | pairs | gained | lost | delta | 95% CI | p |")
        lines.append("|---|---|---|---|---|---|---|")
        for mode, p in paired.items():
            if not p.get("n_pairs"):
                continue
            lo, hi = p["ci95"]
            lines.append(
                f"| {mode} | {p['n_pairs']} | {p['gained']} | {p['lost']} | "
                f"{p['delta']:+.1%} | [{lo:+.1%}, {hi:+.1%}] | {p['mcnemar_p']:.4f} |")
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
    parser.add_argument("--families", default="escalation,closure",
                        help="comma-separated task families")
    parser.add_argument("--evolve", action="store_true",
                        help="retire memories that keep coinciding with failure")
    parser.add_argument("--drift-at", type=int, default=None,
                        help="from this task index onward, the family's policy changes")
    parser.add_argument("--policy-scope", default=None,
                        help="comma-separated trusted release per task, e.g. A,A,A,B,B,B")
    parser.add_argument("--holdout-from", type=int, default=None,
                        help="tasks from this index onward are test: no memory writes")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--run-dir", default=None,
                        help="extend an existing run directory (adds/overwrites modes)")
    parser.add_argument("--shared-stage3-budget", action="store_true",
                        help="use the persistent 450-call, 1M-token stage 3 ledger")
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

    if args.shared_stage3_budget and args.llm != "api":
        parser.error("--shared-stage3-budget requires --llm api")
    # Paid calls must not escape the round's ledger by omitting a flag: that is how
    # 90 calls went unaccounted once. There is no bypass; use the scripted client
    # for offline work.
    if args.llm == "api" and not args.shared_stage3_budget:
        print("refusing to run --llm api without --shared-stage3-budget.\n"
              "Every remote call must be recorded in the round ledger. Use "
              "--llm scripted for offline runs.", file=sys.stderr)
        return 2
    shared = None
    if args.shared_stage3_budget:
        from .shared_budget import SharedBudget
        shared = SharedBudget(settings.artifacts_dir / "stage3" / "round_budget.db", 450, 1_000_000)
    try:
        run_experiment(settings, chosen, args.llm, per_family=args.per_family,
                   seed=args.seed, max_calls=args.max_calls,
                   max_tokens=args.max_tokens,
                   out_root=Path(args.out) if args.out else None,
                   run_dir=Path(args.run_dir) if args.run_dir else None,
                   families=[f.strip() for f in args.families.split(",") if f.strip()],
                       evolve=args.evolve, drift_at=args.drift_at,
                       policy_scopes=([None if s.strip() in ("", "-") else s.strip()
                                       for s in args.policy_scope.split(",")]
                                      if args.policy_scope else None),
                       holdout_from=args.holdout_from,
                       shared_budget=shared)
    finally:
        if shared:
            shared.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
