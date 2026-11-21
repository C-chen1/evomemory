"""End-to-end demonstration of the experience-memory mainline. Offline, no API.

    PYTHONPATH=src python scripts/demo.py

What it shows, in order:

1. a task fails because there is no experience yet;
2. a candidate experience is extracted, validated and activated;
3. the next task in the family retrieves it and succeeds;
4. the policy changes underneath the memory, and the stale rule now causes failures;
5. an explicit, cited skip gives the harness enough evidence to retire that rule,
   and the family recovers — while the retired version stays readable.

Everything here runs the scripted policy, so results are labelled
``harness_selfcheck``. They validate the harness, not a model.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evomemory.config import Settings  # noqa: E402
from evomemory.runner import run_experiment  # noqa: E402
from evomemory.models import MemoryMode  # noqa: E402

OUT = ROOT / "artifacts" / "demo"
PER_FAMILY = 8
DRIFT_AT = 3


def _db(run_dir: Path):
    import sqlite3
    return sqlite3.connect(run_dir / "mode_E" / "memory.db")


def _run(name: str, evolve: bool) -> Path:
    out = OUT / name
    if out.exists():
        shutil.rmtree(out)
    settings = Settings.from_env(root=ROOT)
    settings.api_key = ""                     # force the offline path
    run_experiment(settings, [MemoryMode.E_SELECTIVE], "scripted",
                   per_family=PER_FAMILY, families=("escalation",),
                   drift_at=DRIFT_AT, evolve=evolve, out_root=out)
    return next(out.glob("run_*"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("EvoMemory demo — evidence class: harness_selfcheck / scripted policy")
    print("=" * 72)
    print("\nStory: 8 tasks in one family. Tasks 0-2 follow policy A; from task 3 the")
    print("policy changes to B while the family name stays the same, so the memory")
    print("learned under A keeps being offered. Two runs differ only in whether the")
    print("store is allowed to retire a memory that has stopped working.\n")

    noevo = _run("without_evolution", evolve=False)
    evo = _run("with_evolution", evolve=True)

    for label, run_dir in (("WITHOUT evolution", noevo), ("WITH evolution", evo)):
        db = _db(run_dir)
        rows = db.execute(
            "SELECT t.task_id, o.status, o.reason FROM traces t "
            "JOIN outcomes o ON o.trace_id = t.trace_id ORDER BY t.task_id").fetchall()
        print(f"--- {label} ---")
        for task_id, status, reason in rows:
            mark = "OK  " if status == "success" else "fail"
            note = "" if status == "success" else f"   {reason[:60]}"
            print(f"  {task_id:22s} {mark}{note}")
        strikes = db.execute("SELECT COUNT(*) FROM experience_strikes").fetchone()[0]
        skips = db.execute("SELECT COUNT(*) FROM skip_decisions").fetchone()[0]
        weak = db.execute("SELECT COUNT(*) FROM attribution_weak").fetchone()[0]
        retired = db.execute(
            "SELECT COUNT(*) FROM experiences WHERE status='deprecated'").fetchone()[0]
        print(f"  explicit skips={skips}  cited strikes={strikes}  "
              f"weak evidence={weak}  retired memories={retired}")
        db.close()
        print()

    db = _db(evo)
    print("--- what the store ended up with (with_evolution) ---")
    for status, version, lesson in db.execute(
            "SELECT status, version, substr(lesson, 1, 62) FROM experiences "
            "ORDER BY version"):
        print(f"  v{version} {status:10s} {lesson}")
    print("\n--- one recorded strike (why a memory was retired) ---")
    row = db.execute(
        "SELECT s.experience_id, s.task_id, s.rid, s.reason FROM experience_strikes s "
        "LIMIT 1").fetchone()
    if row:
        print(f"  experience {row[0]}  task={row[1]}  record={row[2]}")
        print(f"  reason: {row[3]}")
    else:
        print("  (none recorded in this run)")
    db.close()

    print("\n--- where to look ---")
    print(f"  runs       : {noevo.relative_to(ROOT)}  |  {evo.relative_to(ROOT)}")
    print("  memory db  : <run>/mode_E/memory.db")
    print("               tables: experiences, experience_events, rule_decisions,")
    print("                       task_actions, skip_decisions, experience_strikes,")
    print("                       attribution_weak, attribution_unknown")
    print("  evidence   : <run>/mode_E/evidence/  (content addressed by sha256)")
    print("  results    : <run>/results.json  and  summary.md")
    print("\nNext: the history-dependent visual demo")
    print("  PYTHONPATH=src python scripts/demo_history.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
