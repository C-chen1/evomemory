"""History-dependent visual demo. Offline, no API, no model.

    PYTHONPATH=src python scripts/demo_history.py

The point of this task family: the current image shows the current lamp colours,
but whether a red region may be switched on depends on how often that region was
red in *earlier* panels. That count is not in the image, not in the goal text and
not reachable from any tool — so an agent that only looks at the panel must get it
wrong, and one that carries traced history can get it right.

Both policies below are deterministic stand-ins (``harness_selfcheck``). They
demonstrate that the task discriminates; they are not model results.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evomemory.history_bench import (  # noqa: E402
    LOCK_AFTER, act_image_only, act_with_history, build_sequence, verify,
)

OUT = ROOT / "artifacts" / "demo" / "history"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panels = build_sequence(seed=1234, n_tasks=6)

    print("=" * 72)
    print("EvoMemory history-dependent visual demo — harness_selfcheck / scripted")
    print("=" * 72)
    print(f"\nRule: a region that has been red {LOCK_AFTER}+ times in earlier panels is")
    print("locked and must stay off. The count is only available from the run")
    print("history, never from the current panel.\n")

    rows = []
    for i, panel in enumerate(panels):
        image_path = OUT / f"panel_{i}.png"
        image_path.write_bytes(panel.image)
        digest = hashlib.sha256(panel.image).hexdigest()[:12]

        image_sw, image_log = act_image_only(panel)
        hist_sw, hist_log = act_with_history(panel)
        image_out = verify(panel, image_sw, image_log)
        hist_out = verify(panel, hist_sw, hist_log)

        reds = [z.name for z in panel.zones if z.lamp == "red"]
        print(f"[{i}] {panel.task.task_id}  red={reds}  "
              f"seen_before={panel.history}  locked={panel.locked}")
        print(f"    image only : {image_out.status.value:8s} "
              f"({image_out.reason[:56]})")
        print(f"    with history: {hist_out.status.value:8s} "
              f"({hist_out.reason[:56]})")
        print(f"    image sha256[:12] = {digest}")
        rows.append({"panel": i, "task_id": panel.task.task_id, "red": reds,
                     "history": panel.history, "locked": panel.locked,
                     "image_only": image_out.status.value,
                     "with_history": hist_out.status.value,
                     "image_sha256": hashlib.sha256(panel.image).hexdigest(),
                     "image_path": str(image_path.relative_to(ROOT))})

    image_score = sum(r["image_only"] == "success" for r in rows)
    hist_score = sum(r["with_history"] == "success" for r in rows)
    print(f"\nsummary: image-only {image_score}/{len(rows)}   "
          f"with history {hist_score}/{len(rows)}")
    print("The gap appears exactly on the panels where a region is locked.\n")

    # the same picture can require different actions under different history
    sample = panels[2]
    twin = type(sample)(task=sample.task, zones=list(sample.zones),
                        image=sample.image, history={r: 0 for r in sample.history},
                        locked=[], expected_on=[])
    same_pixels = hashlib.sha256(twin.image).hexdigest() == \
        hashlib.sha256(sample.image).hexdigest()
    a, _ = act_with_history(sample)
    b, _ = act_with_history(twin)
    print("--- same pixels, different history ---")
    print(f"  identical image bytes: {same_pixels}")
    print(f"  real history  {sample.history} -> {a}")
    print(f"  stale history {twin.history} -> {b}")
    print("  A memory that is stale produces a different, wrong action on identical")
    print("  pixels — which is exactly the negative transfer this project studies.\n")

    (OUT / "demo.json").write_text(json.dumps({
        "evidence_class": "harness_selfcheck", "policy": "scripted, deterministic",
        "lock_after": LOCK_AFTER, "panels": rows,
        "image_only_score": f"{image_score}/{len(rows)}",
        "with_history_score": f"{hist_score}/{len(rows)}",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("--- where to look ---")
    print(f"  panels : {OUT.relative_to(ROOT)}/panel_*.png  (content-addressed in a real run)")
    print(f"  result : {(OUT / 'demo.json').relative_to(ROOT)}")
    print("\nIn a real run the panel images enter the evidence store as IMAGE rows,")
    print("and an experience references them by evidence id rather than by path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
