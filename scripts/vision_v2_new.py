"""Exploratory panel-v2 holdout experiment. No historical data are overwritten."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evomemory.config import Settings
from evomemory.db import Database, finish_run, record_run
from evomemory.evidence import EvidenceStore
from evomemory.llm import BudgetedLLM, DeepSeekClient, ScriptedLLM
from evomemory.metrics import compare_paired
from evomemory.shared_budget import SharedBudget
from evomemory.visual_bench import visual_condition
from evomemory.visual_trial import PanelTrainer, execute_panel, make_split, make_training_probe
from vision_experiment import learn_lesson


def propose_condition(llm, trainer, task_id, run_id):
    prompt = ("Infer an exception rule for RED zones only from these completed training "
              "observations. Return a JSON predicate using field lamp or readout, "
              "op eq/ne/lt/lte/gt/gte, or all_of/any_of/not. "
              "A match means leave the red switch off. If uncertain return {}.\n" +
              json.dumps(trainer.feedback_for_extraction(), ensure_ascii=False))
    response = llm.complete([{"role": "user", "content": prompt}], purpose="extract",
                            task_id=task_id, run_id=run_id, max_tokens=3200)
    try:
        s = response.text
        obj = json.loads(s[s.index("{"):s.rindex("}") + 1])
        return obj if isinstance(obj, dict) else {}
    except (ValueError, json.JSONDecodeError):
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--train", type=int, default=3)
    ap.add_argument("--test", type=int, default=4)
    ap.add_argument("--modes", default="V0,V1,V2-old,V2-new")
    ap.add_argument("--adaptive-boundary", action="store_true",
                    help="construct later training panels from earlier training feedback")
    args = ap.parse_args()
    if not 1 <= args.train <= 30 or not 1 <= args.test <= 30:
        raise ValueError("train/test count must be 1..30")
    modes = args.modes.split(",")
    if len(set(modes)) != len(modes) or not set(modes) <= {"V0", "V1", "V2-old", "V2-new", "V2-threshold"}:
        raise ValueError("invalid or repeated mode")
    if args.adaptive_boundary and modes[0] != "V2-threshold":
        raise ValueError("adaptive boundary requires V2-threshold to run first")
    train, test = make_split(range(1000, 1000 + args.train), range(2000, 2000 + args.test))
    settings = Settings.from_env()
    if not args.dry_run:
        settings.require_api()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = ROOT / "artifacts" / "stage3" / f"vision_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    shared = None if args.dry_run else SharedBudget(ROOT / "artifacts" / "stage3" / "round_budget.db",
                                                    450, 1_000_000)
    payload = {"evidence_class": "harness_selfcheck" if args.dry_run else "live_model",
               "dataset_version": "panel-v2", "model": "scripted" if args.dry_run else settings.model,
               "train_seeds": list(range(1000, 1000 + args.train)),
               "test_seeds": list(range(2000, 2000 + args.test)),
               "adaptive_boundary": args.adaptive_boundary, "training_probes": [],
               "modes": {}, "errors": {}}
    try:
        for mode in modes:
            db = Database(out / mode / "run.db")
            evidence = EvidenceStore(out / mode / "evidence", db)
            run_id = f"{mode}-{stamp}"
            record_run(db, run_id, mode, settings.describe())
            client = (ScriptedLLM(lambda _: '{"tool":"finish","args":{}}') if args.dry_run else
                      DeepSeekClient(settings.api_key, settings.model, settings.base_url))
            llm = BudgetedLLM(client, 450, 1_000_000, db, shared_budget=shared)
            trainer = PanelTrainer(db, evidence, run_id, {x.task.task_id for x in train})
            lessons, decisions, outcomes, checks, tool_calls = [], [], {}, {}, 0
            training_hashes = {}
            try:
                for index, bt in enumerate(train):
                    if mode == "V2-threshold" and args.adaptive_boundary and decisions:
                        thresholds = decisions[-1]["induction"]["induction"].get("coverage", {}).get("consistent_thresholds", [])
                        if len(thresholds) > 1:
                            probe = (thresholds[0] + thresholds[-1]) // 2
                            bt = make_training_probe(1000 + index, bt.task.task_id, probe)
                            if any(hashlib.sha256(bt.image).digest() == hashlib.sha256(x.image).digest() for x in test):
                                raise ValueError("training probe duplicates a holdout image")
                            train[index] = bt
                            payload["training_probes"].append({"task_id": bt.task.task_id,
                                                               "readout": probe,
                                                               "sha256": hashlib.sha256(bt.image).hexdigest()})
                    training_hashes[bt.task.task_id] = hashlib.sha256(bt.image).hexdigest()
                    result = execute_panel(llm, db, bt, run_id, lessons, image_enabled=mode != "V0")
                    tool_calls += result.env.calls
                    if mode == "V2-threshold":
                        trainer.record(bt, result)
                        induction = trainer.induce_threshold()
                        decisions.append({"decision": induction.get("admission", induction["induction"]["decision"]),
                                          "induction": induction})
                        lessons = trainer.guidance()
                    elif mode == "V2-new":
                        trainer.record(bt, result)
                        candidate = visual_condition() if args.dry_run else propose_condition(llm, trainer, bt.task.task_id, run_id)
                        proposed = trainer.propose(candidate)
                        if proposed:
                            exp, report, audit = proposed
                            decisions.append({"decision": report.decision, "audit_id": audit,
                                              "coverage": report.coverage})
                        else:
                            decisions.append({"decision": "invalid", "coverage": {}})
                        lessons = trainer.guidance()
                    elif mode == "V2-old" and result.outcome.status.value == "failure":
                        failure = "; ".join(c["name"] for c in result.outcome.checks if not c["passed"])
                        lesson = None if args.dry_run else learn_lesson(llm, bt, failure, run_id)
                        if lesson:
                            lessons.append(lesson)
                # Freeze memory; no test outcome or label is fed back to the trainer.
                for bt in test:
                    result = execute_panel(llm, db, bt, run_id, tuple(lessons), image_enabled=mode != "V0")
                    tool_calls += result.env.calls
                    outcomes[bt.task.task_id] = result.outcome.status.value == "success"
                    checks[bt.task.task_id] = {c["name"]: c["passed"] for c in result.outcome.checks}
            except Exception as exc:
                payload["errors"][mode] = f"{type(exc).__name__}: {exc}"
            finally:
                finish_run(db, run_id)
                payload["modes"][mode] = {"outcomes": outcomes, "checks": checks,
                                          "training_image_sha256": training_hashes,
                                          "lessons": lessons, "decisions": decisions,
                                          "tool_calls": tool_calls, "budget": llm.summary(),
                                          "candidate_rejection_rate": (
                                              sum(x["decision"] in ("rejected", "invalid") for x in decisions) /
                                              len(decisions) if decisions else None),
                                          "complete": len(outcomes) == len(test)}
                db.close()
            if mode in payload["errors"]:
                break
        if "V1" in payload["modes"] and "V2-new" in payload["modes"]:
            payload["paired_V2_new_vs_V1"] = compare_paired(payload["modes"]["V1"]["outcomes"],
                                                              payload["modes"]["V2-new"]["outcomes"])
            base = payload["modes"]["V1"]["outcomes"]
            candidate = payload["modes"]["V2-new"]["outcomes"]
            common = set(base) & set(candidate)
            successes = sum(base[k] for k in common)
            payload["paired_V2_new_vs_V1"]["negative_transfer_rate"] = (
                sum(base[k] and not candidate[k] for k in common) / successes if successes else None)
        if "V1" in payload["modes"] and "V2-threshold" in payload["modes"]:
            baseline = payload["modes"]["V1"]["outcomes"]
            candidate = payload["modes"]["V2-threshold"]["outcomes"]
            paired = compare_paired(baseline, candidate)
            common = set(baseline) & set(candidate)
            successes = sum(baseline[k] for k in common)
            paired["negative_transfer_rate"] = (
                sum(baseline[k] and not candidate[k] for k in common) / successes if successes else None)
            payload["paired_V2_threshold_vs_V1"] = paired
        payload["shared_budget"] = shared.summary() if shared else None
    finally:
        if shared:
            shared.close()
        (out / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(out), "errors": payload["errors"],
                      "complete": {k: v["complete"] for k, v in payload["modes"].items()}}, ensure_ascii=False))
    return 1 if payload["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
