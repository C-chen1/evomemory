"""Does a trusted policy scope unblock a correct new rule that all-history replay
would refuse — without letting a wrong one through?

    PYTHONPATH=src python scripts/demo_policy_scope.py [--out DIR]

Same task sequence in both arms (8 escalation tasks; the family's exemption set
changes from task 3 onward). Tasks 6-7 are held out: they are graded but write no
evidence. The arms differ only in whether the caller declares which trusted policy
release each task belongs to.

Everything here runs the scripted policy; the candidate rules come from the real
extractor over the grader's feedback. This is an offline check of the harness, not
a model result.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evomemory.config import Settings  # noqa: E402
from evomemory.models import MemoryMode  # noqa: E402
from evomemory.runner import run_experiment  # noqa: E402

N_TASKS = 8
DRIFT_AT = 3
HOLDOUT_FROM = 6
SCOPES = ["A", "A", "A", "B", "B", "B", "B", "B"]


def _run(base: Path, label: str, scopes, holdout) -> Path:
    settings = Settings.from_env(root=ROOT)
    settings.api_key = ""
    out = base / label
    run_experiment(settings, [MemoryMode.E_SELECTIVE], "scripted",
                   per_family=N_TASKS, families=("escalation",),
                   drift_at=DRIFT_AT, evolve=True, out_root=out,
                   policy_scopes=scopes, holdout_from=holdout)
    return next(out.glob("run_*"))


def _report(run_dir: Path, title: str) -> dict:
    db = sqlite3.connect(run_dir / "mode_E" / "memory.db")
    rows = db.execute(
        "SELECT t.task_id, o.status, t.params_json FROM traces t "
        "JOIN outcomes o ON o.trace_id = t.trace_id ORDER BY t.task_id").fetchall()
    decisions = [r[0].split(":", 1)[1].split(" ")[0]
                 for r in db.execute(
                     "SELECT reason FROM experience_events WHERE reason LIKE 'replay:%'")]
    rules = [r[0] for r in db.execute(
        "SELECT condition_json FROM experiences WHERE status='active'")]
    trained = {r[0] for r in db.execute("SELECT DISTINCT task_id FROM training_examples")}

    print(f"\n--- {title} ---")
    for task_id, status, params in rows:
        import json
        split = json.loads(params or "{}").get("feedback_split", "train")
        mark = "OK  " if status == "success" else "fail"
        print(f"  {task_id:22s} {mark} [{split}]")
    print(f"  replay decisions : {decisions}")
    print(f"  active rules     : {len(rules)}")
    for rule in rules:
        print(f"    {rule}")
    print(f"  tasks that fed training evidence: {len(trained)}")
    db.close()
    return {"outcomes": {t: s for t, s, _ in rows}, "decisions": decisions,
            "rules": rules, "n_training_tasks": len(trained)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="parent directory (a new subdirectory is created)")
    args = ap.parse_args(argv)

    stamp = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
             + "-" + uuid.uuid4().hex[:6])
    base = (Path(args.out) if args.out else ROOT / "artifacts" / "demo") / f"scope-{stamp}"
    base.mkdir(parents=True, exist_ok=True)

    try:
        shown = base.relative_to(ROOT)
    except ValueError:
        shown = base
    print("=" * 76)
    print("EvoMemory policy-scope demo — harness_selfcheck / scripted policy")
    print("=" * 76)
    print(f"\nOutput: {shown}  (new; nothing overwritten)")
    print(f"\nSequence: {N_TASKS} tasks, the family's exemption set changes at task {DRIFT_AT}.")
    print(f"Tasks {HOLDOUT_FROM}+ are held out: graded, but they write no evidence.")
    print("The only difference between the arms is whether the caller declares which")
    print("trusted policy release each task belongs to.\n")

    plain = _report(_run(base, "unscoped", None, HOLDOUT_FROM),
                    "ARM 1 — no policy release declared (replay uses every task)")
    scoped = _report(_run(base, "scoped", SCOPES, HOLDOUT_FROM),
                     "ARM 2 — trusted release declared per task")

    print("\n--- what the difference means ---")
    print("  The candidate that encodes the NEW rule is the same in both arms, and in")
    print("  both it is consistent with the records graded after the change.")
    print("  Unscoped replay also re-checks it against records graded under the OLD")
    print("  rule, where it conflicts, so it can never be admitted. Declaring the")
    print("  release lets replay use the evidence that is actually about that release.")
    print("\n  This is NOT drift detection: the release has to come from outside.")
    print("  With no declared release the system stays conservative on purpose.")

    print("\n--- where to look ---")
    print(f"  {shown}/unscoped/  and  {shown}/scoped/")
    print("  <run>/mode_E/memory.db: training_examples carries policy_scope per task,")
    print("  traces.params_json records the release and split actually used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
