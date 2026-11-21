"""Live check: does a real model over-act on locked regions without history?

This does not test memory. It tests the *environment*: if a live model with no
history can still avoid the locked regions, the task has no discriminating power
and is no better than the panel task it replaces.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evomemory.config import Settings  # noqa: E402
from evomemory.history_bench import build_sequence, to_env, verify  # noqa: E402
from evomemory.llm import BudgetedLLM, DeepSeekClient, LLMError  # noqa: E402
from evomemory.shared_budget import SharedBudget  # noqa: E402

BUDGET_DB = ROOT / "artifacts" / "stage3" / "round_budget.db"
MAX_CALLS, MAX_TOKENS = 450, 1_000_000
CALL_CAP = 60
ACTION_TOKENS = 1600

SYSTEM = (
    "You are an operations agent controlling a panel.\n"
    "Reply with exactly one JSON object: "
    '{"thought": "<brief>", "tool": "<name>", "args": {...}}\n'
    "Lamp colours are visible ONLY in the image.\n"
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tasks", type=int, default=4)
    args = ap.parse_args(argv)

    settings = Settings.from_env()
    settings.require_api()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = ROOT / "artifacts" / "stage4" / f"history_live_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    budget = SharedBudget(BUDGET_DB, MAX_CALLS, MAX_TOKENS)
    before = budget.summary()
    client = DeepSeekClient(settings.api_key, settings.model, settings.base_url)
    llm = BudgetedLLM(client, CALL_CAP, 300_000, shared_budget=budget)

    panels = build_sequence(seed=args.seed, n_tasks=args.tasks)
    runs: List[Dict[str, Any]] = []
    for panel in panels:
        env = to_env(panel)
        system = (SYSTEM + "\nTOOLS: " + json.dumps(env.spec(), ensure_ascii=False)
                  + "\nGOAL: " + panel.task.goal)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": "Inspect the panel and act."}]
        calls = tokens = 0
        error = ""
        for _ in range(panel.task.max_steps):
            try:
                resp = llm.complete(messages, purpose="act", task_id=panel.task.task_id,
                                    prompt_version="history-v1", max_tokens=ACTION_TOKENS,
                                    images=[panel.image])
            except LLMError as exc:
                error = str(exc)[:200]
                break
            calls += 1
            tokens += resp.input_tokens + resp.output_tokens
            action = _parse(resp.text or "")
            if action is None:
                messages.append({"role": "assistant", "content": (resp.text or "")[:200]})
                messages.append({"role": "user", "content": 'Reply with a JSON action.'})
                continue
            tool = str(action.get("tool", ""))
            if tool == "finish":
                break
            result = env.call(tool, action.get("args") or {})
            messages.append({"role": "assistant",
                             "content": json.dumps(action, ensure_ascii=False)})
            messages.append({"role": "user",
                             "content": json.dumps(result, ensure_ascii=False)[:1200]})

        outcome = verify(panel, env.switches, env.log)
        runs.append({"task_id": panel.task.task_id, "ok": outcome.status.value == "success",
                     "checks": {c["name"]: bool(c["passed"]) for c in outcome.checks},
                     "locked": panel.locked, "history": panel.history,
                     "switches": dict(env.switches), "calls": calls,
                     "tokens": tokens, "error": error})

    payload = {
        "evidence_class": "live_model", "model": settings.model,
        "note": ("no memory: this checks whether the environment discriminates, "
                 "not whether memory helps"),
        "seed": args.seed, "runs": runs,
        "budget": {"before": before, "after": budget.summary()},
    }
    (out / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    for r in runs:
        print(f"{r['task_id']} ok={r['ok']} locked={r['locked']} "
              f"checks={r['checks']} calls={r['calls']} err={r['error'][:60]}")
    print("budget after:", payload["budget"]["after"])
    budget.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
