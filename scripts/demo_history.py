"""History-dependent visual demo. Offline, no API, no model.

    PYTHONPATH=src python scripts/demo_history.py [--out DIR]

What this DOES demonstrate: the current image shows the current lamp colours, but
whether a red region may be switched on depends on how many times it was red in
*earlier* panels. That count is not in the image, not in the goal text and not
reachable from any tool. So a policy that only looks at the panel must get it
wrong, and a policy that consults the stored history can get it right.

How the history is obtained: each panel's image and its region facts go into the
evidence store, and the counts for panel N are rebuilt by reading back panels
0..N-1 from that store — not from an in-memory object.

What this does NOT demonstrate: a real model retrieving this history. Both
policies below are deterministic stand-ins (``harness_selfcheck``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evomemory.db import Database  # noqa: E402
from evomemory.evidence import EvidenceStore  # noqa: E402
from evomemory.history_bench import (  # noqa: E402
    LOCK_AFTER, act_image_only, act_with_history, build_sequence, verify,
)
from evomemory.models import EvidenceKind  # noqa: E402


def _show(path: Path) -> str:
    """Path for display: relative when it is inside the project, absolute otherwise."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="parent directory (a new subdirectory is created; nothing "
                         "is deleted)")
    args = ap.parse_args(argv)

    stamp = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
             + "-" + uuid.uuid4().hex[:6])
    out = (Path(args.out) if args.out else ROOT / "artifacts" / "demo" / "history") / stamp
    out.mkdir(parents=True, exist_ok=True)

    db = Database(out / "demo.db")
    store = EvidenceStore(out / "evidence", db)
    panels = build_sequence(seed=1234, n_tasks=6)

    print("=" * 74)
    print("EvoMemory history-dependent visual demo — harness_selfcheck / scripted")
    print("=" * 74)
    print(f"\nOutput: {_show(out)}  (new; nothing overwritten)")
    print(f"\nRule: a region red {LOCK_AFTER}+ times in earlier panels is locked and must")
    print("stay off. The count exists only in stored history, never in the panel.\n")

    rows = []
    stored_facts: list[dict] = []
    for i, panel in enumerate(panels):
        # this panel's evidence: the image (content addressed) plus its region facts
        image_ev = store.put(panel.image, EvidenceKind.IMAGE,
                             metadata={"task_id": panel.task.task_id, "panel": i})
        (out / f"panel_{i}.png").write_bytes(panel.image)
        facts = {z.name: z.lamp for z in panel.zones}
        facts_ev = store.put_json(
            {"panel": i, "task_id": panel.task.task_id, "lamps": facts},
            metadata={"kind": "panel_facts", "panel": i},
            derived_from=image_ev.evidence_id)

        # history rebuilt from what is already stored, not from the live object
        history = {r: 0 for r in facts}
        for prior in stored_facts:
            for region, lamp in prior["lamps"].items():
                if lamp == "red":
                    history[region] += 1
        replay_panel = replace(panel, history=history)

        image_sw, image_log = act_image_only(panel)
        hist_sw, hist_log = act_with_history(replay_panel)
        image_out = verify(panel, image_sw, image_log)
        hist_out = verify(panel, hist_sw, hist_log)

        reds = [z.name for z in panel.zones if z.lamp == "red"]
        print(f"[{i}] {panel.task.task_id}  red={reds}  "
              f"history_from_store={history}  locked={panel.locked}")
        print(f"    image only  : {image_out.status.value:8s} "
              f"{image_out.reason[:52]}")
        print(f"    with history: {hist_out.status.value:8s} "
              f"{hist_out.reason[:52]}")
        print(f"    evidence: image={image_ev.evidence_id}  facts={facts_ev.evidence_id}")
        rows.append({"panel": i, "task_id": panel.task.task_id, "red": reds,
                     "history_from_store": history, "locked": panel.locked,
                     "image_only": image_out.status.value,
                     "with_history": hist_out.status.value,
                     "image_evidence_id": image_ev.evidence_id,
                     "facts_evidence_id": facts_ev.evidence_id,
                     "image_sha256": hashlib.sha256(panel.image).hexdigest()})
        stored_facts.append({"lamps": facts})

    image_score = sum(r["image_only"] == "success" for r in rows)
    hist_score = sum(r["with_history"] == "success" for r in rows)
    print(f"\nsummary: image-only {image_score}/{len(rows)}   "
          f"with history {hist_score}/{len(rows)}")
    print("The gap appears exactly on the panels where a region is locked.\n")

    sample = panels[2]
    stale = replace(sample, history={r: 0 for r in sample.history})
    a, _ = act_with_history(sample)
    b, _ = act_with_history(stale)
    same_pixels = hashlib.sha256(sample.image).hexdigest() == \
        hashlib.sha256(stale.image).hexdigest()
    print("--- same pixels, different history ---")
    print(f"  identical image bytes : {same_pixels}")
    print(f"  true history  {sample.history} -> {a}")
    print(f"  stale history {stale.history} -> {b}")
    print("  A stale memory produces a different, wrong action on identical pixels.\n")

    (out / "demo.json").write_text(json.dumps({
        "evidence_class": "harness_selfcheck",
        "policy": "deterministic stand-in, not a model",
        "history_source": "reconstructed from stored panel_facts evidence",
        "lock_after": LOCK_AFTER, "panels": rows,
        "image_only_score": f"{image_score}/{len(rows)}",
        "with_history_score": f"{hist_score}/{len(rows)}",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    db.close()

    print(f"--- where to look ---\n  {_show(out)}/demo.json")
    print(f"  {_show(out)}/evidence/  (images and panel facts, sha256 named)")
    print("\nBoth policies above are deterministic stand-ins. No model was called, and")
    print("this is not evidence that a model can use visual history.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
