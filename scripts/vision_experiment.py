"""Visual memory experiment: does reusing image evidence help?

Three arms, same panels, same budget:

* ``V0`` no image is sent. The model can only call ``list_zones`` (names) and
  guess. This is the floor, and it proves the environment cannot be solved blind.
* ``V1`` the current panel image is sent; no memory.
* ``V2`` the current panel image is sent **plus** memories learned from earlier
  panels in the same family, as prose.

A fourth arm is available once a rule has been learned:

* ``V3`` like V2 but the learned rule is also executable: the model may call
  ``check_rule`` against the panel state it reports, so the conjunction is
  evaluated by code rather than read out of prose.

Run with ``--dry-run`` to exercise the plumbing without any API call: a scripted
policy stands in for the model and receives the ground-truth panel state. That
path only proves the harness works; it says nothing about vision.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evomemory.config import Settings  # noqa: E402
from evomemory.db import Database, finish_run, record_run  # noqa: E402
from evomemory.evidence import EvidenceStore  # noqa: E402
from evomemory.llm import BudgetedLLM, DeepSeekClient, LLMError, ScriptedLLM  # noqa: E402
from evomemory.metrics import compare_paired  # noqa: E402
from evomemory.models import EvidenceKind  # noqa: E402
from evomemory.visual_bench import (  # noqa: E402
    PanelEnv, PanelTask, build_panel_task, verify,
)

SYSTEM = (
    "You are an operations agent controlling a panel.\n"
    "Reply with exactly one JSON object: "
    '{"thought": "...", "tool": "<name>", "args": {...}}\n'
    "Tools: list_zones (names only), set_switch(zone, on|off), log_entry(text), "
    "finish().\n"
    "Lamp colours and readouts are visible ONLY in the image. If no image is "
    "provided, say so in your thought and act conservatively.\n"
    "Never claim success: the environment decides.\n"
)


@dataclass
class ModeResult:
    mode: str
    outcomes: Dict[str, bool] = field(default_factory=dict)
    checks: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    lessons: List[str] = field(default_factory=list)
    tokens: int = 0
    calls: int = 0
    images_sent: int = 0

    def check_totals(self) -> Dict[str, Any]:
        """Per-check pass counts: separates "cannot see" from "does not know the rule"."""
        totals: Dict[str, List[int]] = {}
        for row in self.checks.values():
            for name, ok in row.items():
                totals.setdefault(name, [0, 0])
                totals[name][1] += 1
                totals[name][0] += 1 if ok else 0
        return {name: {"pass": v[0], "of": v[1]} for name, v in sorted(totals.items())}

    def summary(self) -> Dict[str, Any]:
        n = len(self.outcomes)
        return {"mode": self.mode, "n": n,
                "success": sum(self.outcomes.values()),
                "rate": round(sum(self.outcomes.values()) / n, 4) if n else 0.0,
                "tokens": self.tokens, "calls": self.calls,
                "images_sent": self.images_sent}


def _roundtrip(llm: BudgetedLLM, messages: List[Dict[str, str]],
               images: Optional[List[bytes]], task_id: str, run_id: str) -> Optional[str]:
    resp = llm.complete(messages, purpose="act", task_id=task_id, run_id=run_id,
                        prompt_version="vision-v1", max_tokens=1200, images=images)
    return resp.text


def run_task(llm: BudgetedLLM, bt: PanelTask, mode: str,
             memories: Sequence[str], run_id: str) -> tuple[bool, int, int]:
    env = PanelEnv(bt)
    images = [bt.image] if mode != "V0" else None
    system = (SYSTEM + "\nTOOLS: " + json.dumps(env.spec(), ensure_ascii=False)
              + "\nGOAL: " + bt.task.goal)
    if memories:
        system += "\n\nReusable guidance from earlier panels:\n" + "\n".join(
            f"- {m}" for m in memories)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Inspect the panel and act."},
    ]

    images_sent = 1 if images else 0
    for _ in range(bt.task.max_steps):
        try:
            raw = _roundtrip(llm, messages, images, bt.task.task_id, run_id)
        except LLMError:
            break
        action = _parse(raw or "")
        if action is None:
            messages.append({"role": "assistant", "content": (raw or "")[:300]})
            messages.append({"role": "user", "content":
                             'Reply with {"tool": ..., "args": {...}}'})
            continue
        tool = str(action.get("tool", ""))
        if tool == "finish":
            break
        result = env.call(tool, action.get("args") or {})
        messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
        messages.append({"role": "user", "content": json.dumps(result, ensure_ascii=False)[:1500]})

    outcome = verify(bt, env)
    return (outcome.status.value == "success", images_sent, env.calls,
            {c["name"]: bool(c["passed"]) for c in outcome.checks})


def _parse(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) and "tool" in value else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="V1")
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--max-calls", type=int, default=120)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "vision"))
    ap.add_argument("--dry-run", action="store_true",
                    help="no API calls; scripted policy with oracle state (plumbing only)")
    args = ap.parse_args(argv)

    settings = Settings.from_env()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    tasks = [build_panel_task(seed=args.seed + i) for i in range(args.tasks)]
    results: Dict[str, ModeResult] = {}

    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        mdir = out_dir / f"mode_{mode}"
        mdir.mkdir(parents=True, exist_ok=True)
        db = Database(mdir / "vision.db")
        evidence = EvidenceStore(mdir / "evidence", db)
        run_id = f"{mode}-{stamp}"
        record_run(db, run_id, mode, settings.describe())

        if args.dry_run:
            # In V0 no image is sent, so the stand-in is deliberately blind: it
            # can list zone names and nothing else. If V0 still passed, the
            # environment would be leaking state through text and the experiment
            # would be measuring nothing.
            llm = BudgetedLLM(ScriptedLLM(_oracle_responder(tasks, blind=(mode == "V0"))),
                              args.max_calls, 400_000, db)
        else:
            settings.require_api()
            llm = BudgetedLLM(DeepSeekClient(settings.api_key, settings.model,
                                             settings.base_url), args.max_calls,
                              400_000, db)

        mr = ModeResult(mode=mode)
        learned: List[str] = []
        env_failure = ""
        for bt in tasks:
            try:
                ok, images_sent, calls, checks = run_task(llm, bt, mode, learned, run_id)
                env_failure = "; ".join(k for k, v in checks.items() if not v)
            except Exception as exc:  # noqa: BLE001 - recorded, then stop
                print(f"[{mode}] stopped at {bt.task.task_id}: {exc}", file=sys.stderr)
                break
            mr.outcomes[bt.task.task_id] = ok
            mr.checks[bt.task.task_id] = checks
            mr.images_sent += images_sent
            # store the panel as content-addressed evidence, every time
            evidence.put(bt.image, EvidenceKind.IMAGE,
                         metadata={"task_id": bt.task.task_id, "arm": mode})
            if mode in ("V2", "V3"):
                # learn from failure, not from success: with no rule in hand the
                # first tasks all fail, so a success-only learner never starts.
                if not ok and env_failure:
                    lesson = learn_lesson(llm, bt, env_failure, run_id)
                    if lesson:
                        learned.append(lesson)
                        mr.lessons.append(lesson)
            elif ok and mode != "V0":
                learned.append(_lesson_from(bt))
        finish_run(db, run_id)
        mr.tokens = llm.summary()["total_tokens"]
        mr.calls = llm.summary()["calls"]
        results[mode] = mr
        db.close()
        print(f"[{mode}] {mr.summary()} checks={mr.check_totals()}")

    payload = {
        "generated_at": stamp,
        "evidence_class": ("harness_selfcheck" if args.dry_run else "live_model"),
        "model": settings.model if not args.dry_run else "scripted-dry-run",
        "modes": [r.summary() for r in results.values()],
        "check_totals": {m: r.check_totals() for m, r in results.items()},
        "lessons": {m: r.lessons for m, r in results.items()},
        "per_task": {m: r.outcomes for m, r in results.items()},
    }
    if "V0" in results and "V1" in results:
        payload["paired_V1_vs_V0"] = compare_paired(results["V0"].outcomes,
                                                    results["V1"].outcomes)
    (out_dir / f"results_{stamp}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["modes"], ensure_ascii=False, indent=1))
    return 0


EXTRACT_SYSTEM = (
    "You just attempted a panel task and it failed. The panel image is attached.\n"
    "Write ONE sentence stating the rule you should have followed, so that a future "
    "task on a similar panel succeeds. Be concrete about what distinguishes the "
    "records you should not touch. Reply with the sentence only."
)


def _lesson_from(bt: PanelTask) -> str:
    """Used by the dry run only; the live V2 arm asks the model instead."""
    return ("On these panels, zones whose lamp is red AND whose readout is 50 or "
            "higher are in lockdown: leave their switch off even though the lamp is red.")


def learn_lesson(llm: BudgetedLLM, bt: PanelTask, failure: str,
                 run_id: str) -> Optional[str]:
    """Ask the model, with the image in context, what rule it should have followed.

    The lesson is the model's own induction from what it saw and what the grader
    said — not ground truth injected by the harness.
    """
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": (
            "GOAL: " + bt.task.goal + "\nGRADER SAID: " + failure
            + "\nWhat rule should I have applied?")},
    ]
    try:
        resp = llm.complete(messages, purpose="extract", task_id=bt.task.task_id,
                            run_id=run_id, prompt_version="vision-extract-v1",
                            max_tokens=600, images=[bt.image])
    except LLMError:
        return None
    text = (resp.text or "").strip()
    return text[:400] if text else None


def _oracle_responder(tasks: Sequence[PanelTask], blind: bool = False):
    """Dry-run stand-in: reads the ground truth. Plumbing test only."""
    by_goal = {bt.task.goal: bt for bt in tasks}

    def responder(messages: List[Dict[str, str]]) -> str:
        system = messages[0]["content"]
        bt = next((b for g, b in by_goal.items() if g in system), None)
        if bt is None:
            return json.dumps({"thought": "no task", "tool": "finish", "args": {}})
        if not any(m["role"] == "user" and m["content"].startswith('{"ok"')
                   for m in messages[1:]):
            return json.dumps({"thought": "list", "tool": "list_zones", "args": {}})
        if blind:
            return json.dumps({"thought": "no image was provided; I cannot see the "
                                          "lamp states, so I will not switch anything",
                               "tool": "finish", "args": {}})
        done = {json.loads(m["content"]).get("args", {}).get("zone")
                for m in messages if m["role"] == "assistant"}
        for zone in bt.expected_on:
            if zone not in done:
                return json.dumps({"thought": "switch", "tool": "set_switch",
                                   "args": {"zone": zone, "state": "on"}})
        if not any("log_entry" in m["content"] for m in messages if m["role"] == "assistant"):
            return json.dumps({"thought": "log", "tool": "log_entry",
                               "args": {"text": "done"}})
        return json.dumps({"thought": "finish", "tool": "finish", "args": {}})

    return responder


if __name__ == "__main__":
    raise SystemExit(main())
