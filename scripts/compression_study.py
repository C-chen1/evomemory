"""Multimodal memory compression study.

Generates synthetic evidence images, then measures — for each thumbnail
configuration — what it costs and what verifiable information survives.

Honest scoping:
* visual-token numbers are *estimates* from a patch/merge convention, not
  provider-reported usage;
* fidelity numbers are computed from pixels, so they are measurements;
* nothing here claims a retrieval-accuracy benefit. That experiment needs a
  vision-language model and is listed in NEXT_STEPS.md as not run.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evomemory.multimodal import (  # noqa: E402
    compression_study, render_study, representation_costs,
)


def synthetic_images():
    """Deterministic images with distinct geometry and colour.

    Written with plain PNG encoding so the study needs no fixture files.
    """
    from PIL import Image, ImageDraw, ImageFilter

    out = []
    specs = [
        ("wide_banner", 1200, 300, (198, 40, 40)),
        ("tall_screenshot", 360, 960, (40, 120, 198)),
        ("square_chart", 640, 640, (40, 198, 90)),
        ("small_card", 180, 140, (230, 200, 40)),
    ]
    for name, w, h, colour in specs:
        img = Image.new("RGB", (w, h), colour)
        draw = ImageDraw.Draw(img)
        for i in range(6):
            x = int(w * (0.1 + 0.13 * i))
            draw.rectangle([x, int(h * 0.2), x + max(4, w // 20), int(h * 0.8)],
                           fill=(20, 20, 20 + i * 20))
        img = img.filter(ImageFilter.SMOOTH)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append((name, buf.getvalue()))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "multimodal"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = synthetic_images()
    rows = compression_study(images, sizes=(96, 224, 448, 896),
                             modes=("squash", "letterbox", "crop"))
    table = render_study(rows)

    costs = {}
    for name, data in images:
        costs[name] = [c.to_dict() for c in
                       representation_costs(data, f"synthetic evidence: {name}")]
    (out_dir / "compression_study.json").write_text(
        json.dumps({"rows": rows, "costs": costs}, indent=2), encoding="utf-8")

    lines = [
        "# Multimodal memory compression study",
        "",
        f"Images: {len(images)} synthetic, deterministic. "
        f"Grid: 4 sizes x 3 resize modes = {len(rows)} measured cells.",
        "",
        "Visual-token figures are **estimates** (16px patches, 2x2 merge). "
        "Fidelity figures are measured from pixels.",
        "",
        "## Cost of the three representations",
        "",
        "| image | caption only | caption + 448 thumb | caption + original |",
        "|---|---|---|---|",
    ]
    for name, reps in costs.items():
        by = {c["representation"]: c["tokens"] for c in reps}
        lines.append(f"| {name} | {by['caption_only']} | "
                     f"{by['caption_plus_thumbnail']} | {by['caption_plus_original']} |")
    lines += ["", "## Fidelity and size by thumbnail configuration", "", table, ""]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:20]))
    print(f"\nwrote {out_dir/'REPORT.md'} and compression_study.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
