"""panel-v4: a bounded, preregistered visual comparison (docs/PANEL_V4_PROTOCOL.md).

Two arms on the same four unseen test panels:

* V1  — image, no memory.
* V2T — image + a threshold induced *programmatically* from graded training panels.

The induction is not done by the model. Three earlier attempts at open-ended model
induction returned empty content at 900/1800/3200 output tokens (reasoning budget
exhausted), so this protocol removes that step rather than retrying it: the rule
is enumerated over a declared integer hypothesis class and the model only executes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evomemory.config import Settings  # noqa: E402
from evomemory.db import Database  # noqa: E402
from evomemory.llm import BudgetedLLM, DeepSeekClient, LLMError  # noqa: E402
from evomemory.metrics import compare_paired  # noqa: E402
from evomemory.shared_budget import SharedBudget  # noqa: E402
from evomemory.visual_bench import (  # noqa: E402
    PanelEnv, PanelTask, build_panel_task_v3, build_panel_task_with_readouts, verify,
)

BUDGET_DB = ROOT / "artifacts" / "stage3" / "round_budget.db"
MAX_CALLS = 450
MAX_TOKENS = 1_000_000
EXPERIMENT_CALL_CAP = 120
#: Bisection probes used to narrow the consistent threshold interval to one value.
#: Declared in the protocol before any test panel was run.
MAX_PROBES = 6
PROBE_SEEDS = (7000, 7001, 7002, 7003, 7004, 7005)
TRAIN_SEEDS = (5000, 5001, 5002, 5003)
TEST_SEEDS = (6000, 6001, 6002, 6003)
ACTION_TOKENS = 1600

SYSTEM = (
    "You are an operations agent controlling a panel.\n"
    "Reply with exactly one JSON object: "
    '{"thought": "<brief>", "tool": "<name>", "args": {...}}\n'
    "Lamp colours and readouts are visible ONLY in the image.\n"
    "Never claim success: the environment decides.\n"
)


def _parse(text: str) -> Optional[Dict[str, Any]]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) and "tool" in value else None


@dataclass
class PanelRun:
    task_id: str
    ok: bool = False
    checks: Dict[str, bool] = field(default_factory=dict)
    calls: int = 0
    tokens: int = 0
    error: str = ""
    zone_facts: List[Dict[str, Any]] = field(default_factory=list)


def run_panel(llm: BudgetedLLM, bt: PanelTask, guidance: str = "") -> PanelRun:
    env = PanelEnv(bt)
    system = (SYSTEM + "\nTOOLS: " + json.dumps(env.spec(), ensure_ascii=False)
              + "\nGOAL: " + bt.task.goal)
    if guidance:
        system += "\n\nLearned from earlier panels:\n- " + guidance
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Inspect the panel and act."}]

    run = PanelRun(task_id=bt.task.task_id)
    for _ in range(bt.task.max_steps):
        try:
            resp = llm.complete(messages, purpose="act", task_id=bt.task.task_id,
                                prompt_version="panel-v4-act",
                                max_tokens=ACTION_TOKENS, images=[bt.image])
        except LLMError as exc:
            run.error = str(exc)[:200]
            break
        run.calls += 1
        run.tokens += resp.input_tokens + resp.output_tokens
        action = _parse(resp.text or "")
        if action is None:
            messages.append({"role": "assistant", "content": (resp.text or "")[:200]})
            messages.append({"role": "user",
                             "content": 'Reply with {"tool": ..., "args": {...}}'})
            continue
        tool = str(action.get("tool", ""))
        if tool == "finish":
            break
        result = env.call(tool, action.get("args") or {})
        messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
        messages.append({"role": "user", "content": json.dumps(result, ensure_ascii=False)[:1200]})

    outcome = verify(bt, env)
    run.ok = outcome.status.value == "success"
    run.checks = {c["name"]: bool(c["passed"]) for c in outcome.checks}
    run.zone_facts = [
        {"zone": z.name, "lamp": z.lamp, "readout": z.readout,
         "expected": "activate" if z.is_target else "leave_off",
         "observed": "activate" if env.switches[z.name] == "on" else "leave_off"}
        for z in bt.zones]
    return run


def induce_threshold(observations: Sequence[PanelRun]) -> Dict[str, Any]:
    """Enumerate integer thresholds consistent with the graded training panels.

    Declared hypothesis class: for red zones, `readout >= T` means leave off.
    A single consistent T is required; otherwise the experiment reports no candidate.
    """
    reds = [(row["readout"], row["expected"])
            for run in observations for row in run.zone_facts if row["lamp"] == "red"]
    if not any(a == "activate" for _, a in reds) or not any(a == "leave_off" for _, a in reds):
        return {"decision": "pending", "reason": "both action classes required",
                "red_observations": len(reds)}
    consistent = [t for t in range(100)
                  if all(("leave_off" if r >= t else "activate") == a for r, a in reds)]
    if len(consistent) == 1:
        return {"decision": "candidate", "threshold": consistent[0],
                "red_observations": len(reds)}
    if consistent:
        # Report the true bounds, not a truncated sample: the bisection uses these
        # to choose its probe, and a truncated list silently probes the wrong value.
        return {"decision": "pending", "reason": "boundary underdetermined",
                "lo": consistent[0], "hi": consistent[-1],
                "n_consistent": len(consistent), "red_observations": len(reds)}
    return {"decision": "rejected", "reason": "no threshold fits",
            "red_observations": len(reds)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "stage3"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    settings = Settings.from_env()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = Path(args.out) / f"panel_v4_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = [build_panel_task_v3(seed=s, task_id=f"train-{i:03d}")
             for i, s in enumerate(TRAIN_SEEDS)]
    test = [build_panel_task_v3(seed=s, task_id=f"test-{i:03d}")
            for i, s in enumerate(TEST_SEEDS)]
    image_hashes = {bt.task.task_id: hashlib.sha256(bt.image).hexdigest()
                    for bt in train + test}

    payload: Dict[str, Any] = {
        "protocol": "docs/PANEL_V4_PROTOCOL.md",
        "started": stamp,
        "evidence_class": "harness_selfcheck" if args.dry_run else "live_model",
        "train_seeds": list(TRAIN_SEEDS), "test_seeds": list(TEST_SEEDS),
        "image_hashes": image_hashes,
    }

    if args.dry_run:
        payload["note"] = "dry run: no API calls, no model"
        (out_dir / "results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"dry run ok -> {out_dir}")
        return 0

    settings.require_api()
    budget = SharedBudget(BUDGET_DB, MAX_CALLS, MAX_TOKENS)
    before = budget.summary()
    db = Database(out_dir / "panel_v4.db")
    client = DeepSeekClient(settings.api_key, settings.model, settings.base_url)
    llm = BudgetedLLM(client, EXPERIMENT_CALL_CAP, 400_000, db, shared_budget=budget)

    # 1) V1 on the training panels -> graded labels (this arm carries no memory)
    train_runs = []
    for bt in train:
        try:
            train_runs.append(run_panel(llm, bt))
        except Exception as exc:  # noqa: BLE001 - recorded, experiment marked incomplete
            payload.setdefault("errors", []).append(f"train {bt.task.task_id}: {exc}")
            break
    payload["training"] = [r.__dict__ for r in train_runs]
    induction = induce_threshold(train_runs)

    # Bisect the consistent interval with panels whose red readouts are chosen to
    # straddle its midpoint. A probe costs one panel run and is graded the same way.
    probes: List[Dict[str, Any]] = []
    if induction.get("decision") == "pending" and "lo" in induction:
        lo, hi = induction["lo"], induction["hi"]
        for seed in PROBE_SEEDS:
            if lo >= hi:
                break
            mid = (lo + hi) // 2
            probe = build_panel_task_with_readouts(seed, f"probe-{seed}", [mid, mid + 1])
            try:
                run = run_panel(llm, probe)
            except Exception as exc:  # noqa: BLE001
                payload.setdefault("errors", []).append(f"probe {seed}: {exc}")
                break
            train_runs.append(run)
            probes.append({"seed": seed, "mid": mid, "ok": run.ok,
                           "zone_facts": run.zone_facts})
            induction = induce_threshold(train_runs)
            if induction.get("decision") == "candidate":
                break
            if induction.get("decision") == "pending" and "lo" in induction:
                lo, hi = induction["lo"], induction["hi"]
            else:
                break
    payload["probes"] = probes
    payload["induction"] = induction

    if induction.get("decision") != "candidate":
        payload["status"] = "experiment incomplete: no unique threshold"
        payload["budget"] = {"before": before, "after": budget.summary()}
        (out_dir / "results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"incomplete: {induction}")
        return 0

    threshold = induction["threshold"]
    guidance = (f"On these panels, red-lamp zones with readout >= {threshold} are subject to "
                f"an operating restriction: leave their switch OFF even though the lamp is red.")

    # 2) both arms on the same unseen test panels
    arms: Dict[str, List[PanelRun]] = {}
    for arm, text in (("V1", ""), ("V2T", guidance)):
        runs = []
        for bt in test:
            try:
                runs.append(run_panel(llm, bt, text))
            except Exception as exc:  # noqa: BLE001
                runs.append(PanelRun(task_id=bt.task.task_id, error=str(exc)[:200]))
        arms[arm] = runs

    def summarise(runs):
        return {
            "n": len(runs), "success": sum(r.ok for r in runs),
            "completed": sum(1 for r in runs if not r.error),
            "checks": {name: sum(r.checks.get(name, False) for r in runs)
                       for name in ("targets_switched_on", "lockdown_respected",
                                    "no_collateral_switching", "log_written")},
            "calls": sum(r.calls for r in runs),
            "tokens": sum(r.tokens for r in runs),
            "per_task": {r.task_id: r.ok for r in runs},
            "errors": [r.error for r in runs if r.error],
        }

    payload["threshold"] = threshold
    payload["guidance"] = guidance
    payload["arms"] = {arm: summarise(runs) for arm, runs in arms.items()}
    payload["paired_V2T_vs_V1"] = compare_paired(
        {r.task_id: r.ok for r in arms["V1"]},
        {r.task_id: r.ok for r in arms["V2T"]})
    payload["status"] = ("completed"
                         if all(not r.error for runs in arms.values() for r in runs)
                         else "experiment incomplete: an arm did not finish every panel")
    payload["budget"] = {"before": before, "after": budget.summary()}

    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "threshold": threshold,
                      "arms": payload["arms"], "paired": payload["paired_V2T_vs_V1"],
                      "budget_after": payload["budget"]["after"]},
                     ensure_ascii=False, indent=1))
    db.close()
    budget.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
