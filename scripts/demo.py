"""End-to-end demonstration of the experience-memory mainline. Offline, no API.

    PYTHONPATH=src python scripts/demo.py [--out DIR]

Writes to a **new** run directory every time (never deletes an existing one) and
prints where it went.

What it shows, in order:

1. a task fails because there is no experience yet;
2. a candidate is extracted, then replayed against the graded record tables of
   the tasks already completed in that family, and only then activated;
3. the next task in the family retrieves it and succeeds;
4. the policy changes underneath the memory, and the stale rule now causes failures;
5. an explicit, cited skip gives the harness enough evidence to retire that rule,
   and the family recovers, while the retired version stays readable.

Everything runs the scripted policy, so results are labelled ``harness_selfcheck``.
They validate the harness, not a model.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evomemory.config import Settings  # noqa: E402
from evomemory.models import MemoryMode  # noqa: E402
from evomemory.runner import run_experiment  # noqa: E402

PER_FAMILY = 8
DRIFT_AT = 3


def _run(base: Path, name: str, evolve: bool) -> Path:
    settings = Settings.from_env(root=ROOT)
    settings.api_key = ""                     # force the offline path
    out = base / name
    run_experiment(settings, [MemoryMode.E_SELECTIVE], "scripted",
                   per_family=PER_FAMILY, families=("escalation",),
                   drift_at=DRIFT_AT, evolve=evolve, out_root=out)
    return next(out.glob("run_*"))


def _connect(run_dir: Path):
    import sqlite3
    return sqlite3.connect(run_dir / "mode_E" / "memory.db")


def _show(path: Path) -> str:
    """Path for display: relative when it is inside the project, absolute otherwise."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="parent directory for this demo run (a new subdirectory "
                         "is created inside it; nothing is deleted)")
    args = ap.parse_args(argv)

    stamp = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
             + "-" + uuid.uuid4().hex[:6])
    base = (Path(args.out) if args.out else ROOT / "artifacts" / "demo") / stamp
    base.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("EvoMemory demo — evidence class: harness_selfcheck / scripted policy")
    print("=" * 74)
    print(f"\nOutput directory: {_show(base)}  (new; nothing overwritten)")
    print("\nStory: 8 tasks in one family. Tasks 0-2 follow policy A; from task 3 the")
    print("policy changes to B while the family name stays the same, so the memory")
    print("learned under A keeps being offered. The two runs differ only in whether")
    print("the store may retire a memory that has stopped working.\n")

    noevo = _run(base, "without_evolution", evolve=False)
    evo = _run(base, "with_evolution", evolve=True)

    for label, run_dir in (("WITHOUT evolution", noevo), ("WITH evolution", evo)):
        db = _connect(run_dir)
        rows = db.execute(
            "SELECT t.task_id, o.status, o.reason FROM traces t "
            "JOIN outcomes o ON o.trace_id = t.trace_id ORDER BY t.task_id").fetchall()
        print(f"--- {label} ---")
        for task_id, status, reason in rows:
            mark = "OK  " if status == "success" else "fail"
            note = "" if status == "success" else f"   {reason[:56]}"
            print(f"  {task_id:22s} {mark}{note}")
        counts = {t: db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("skip_decisions", "experience_strikes",
                            "attribution_weak", "training_examples")}
        retired = db.execute(
            "SELECT COUNT(*) FROM experiences WHERE status='deprecated'").fetchone()[0]
        print(f"  explicit skips={counts['skip_decisions']}  "
              f"cited strikes={counts['experience_strikes']}  "
              f"weak evidence={counts['attribution_weak']}  "
              f"retired={retired}  training samples={counts['training_examples']}")
        db.close()
        print()

    db = _connect(evo)
    print("--- replay: candidates must explain the records already graded ---")
    n_examples = db.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0]
    print(f"  {n_examples} record-level training samples were replayed against")
    events = db.execute(
        "SELECT version, reason FROM experience_events WHERE reason LIKE 'replay:%' "
        "ORDER BY id").fetchall()
    accepted = sum(1 for _, r in events if "replay:accepted" in r)
    rejected = sum(1 for _, r in events if "replay:rejected" in r)
    pending = sum(1 for _, r in events if "replay:pending" in r)
    print(f"  decisions: accepted={accepted} rejected={rejected} pending={pending}")
    for version, reason in events:
        if "replay:rejected" in reason:
            print(f"  first rejection (v{version}): {reason[:120]}")
            break

    print("\n--- what the store ended up with (with_evolution) ---")
    for status, version, lesson in db.execute(
            "SELECT status, version, substr(lesson, 1, 60) FROM experiences "
            "ORDER BY version"):
        print(f"  v{version} {status:10s} {lesson}")

    print("\n--- one recorded strike (why a memory was retired) ---")
    row = db.execute(
        "SELECT experience_id, task_id, rid, reason FROM experience_strikes "
        "LIMIT 1").fetchone()
    if row:
        print(f"  experience {row[0]}  task={row[1]}  record={row[2]}")
        print(f"  reason: {row[3]}")
    else:
        print("  (none recorded in this run)")
    db.close()

    print("\n--- where to look ---")
    print(f"  runs      : {_show(noevo)}  |  {_show(evo)}")
    print("  memory db : <run>/mode_E/memory.db")
    print("              tables: experiences, experience_events, training_examples,")
    print("                      rule_decisions, task_actions, skip_decisions,")
    print("                      experience_strikes, attribution_weak/unknown")
    print("  evidence  : <run>/mode_E/evidence/  (content addressed by sha256)")
    print("  results   : <run>/results.json  and  summary.md")
    print("\nTwo things to read carefully:")
    print("  * retirement creates a NEW experience; it does not revise one in place.")
    print("    In-place revision (v1 -> v2) is implemented and tested")
    print("    (ExperienceStore.revise + MemoryGate.admit_with_replay) but this demo")
    print("    does not exercise it.")
    print("  * after the policy changes, the candidate that would encode the new rule")
    print("    is rejected by replay: it contradicts records graded under the old")
    print("    policy. That is the gate doing its job, and it is also why recovery")
    print("    stops here. Retiring stale *rules* is not enough; the evidence they")
    print("    produced is still in the replay set. Windowed or evidence-retiring")
    print("    replay is future work, not part of this demo.")
    print("\nNext: PYTHONPATH=src python scripts/demo_history.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
