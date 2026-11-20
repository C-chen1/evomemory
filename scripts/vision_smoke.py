"""Smoke test: can the configured model actually read pixels?

Everything in P3 depends on this answer, so it is checked empirically rather
than assumed from documentation. The image is generated here, so the ground
truth is known by construction — no labels leak into the request.

Costs one API call. Prints no image content beyond the answers being scored.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evomemory.config import Settings  # noqa: E402
from evomemory.llm import DeepSeekClient, LLMError  # noqa: E402


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# The ground truth is fixed here and rendered, never read back from the image.
PANEL_TITLE = "PANEL-7"
READOUT = "42"
RED_LIGHTS = 1
GREEN_LIGHTS = 2


def build_panel() -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (640, 400), (248, 248, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 620, 380], outline=(40, 40, 60), width=3)

    title_font = ImageFont.truetype(FONT, 44)
    small = ImageFont.truetype(FONT, 30)

    draw.text((40, 40), PANEL_TITLE, font=title_font, fill=(20, 20, 40))

    # status lights: one red, two green
    draw.ellipse([60, 130, 120, 190], fill=(200, 30, 30))
    draw.ellipse([150, 130, 210, 190], fill=(30, 170, 60))
    draw.ellipse([240, 130, 300, 190], fill=(30, 170, 60))

    # numeric readout in its own box
    draw.rectangle([380, 120, 580, 200], outline=(40, 40, 60), width=2)
    draw.text((410, 130), READOUT, font=ImageFont.truetype(FONT, 52), fill=(10, 10, 10))

    draw.text((60, 280), "STATUS", font=small, fill=(90, 90, 110))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


QUESTIONS = [
    ("title", f"What text is shown at the top of the panel? Reply with only that text."),
    ("lights", "How many red lights and how many green lights are visible? "
               "Reply exactly as: red=N green=M"),
    ("readout", "What number is displayed in the readout box? Reply with only the number."),
]


def ask(client: DeepSeekClient, data_url: str, question: str) -> tuple[str, int, int]:
    """One vision request. Returns (answer, input_tokens, output_tokens)."""
    body = {
        "model": client.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        "max_tokens": 400,
        "temperature": 0,
    }
    req = urllib.request.Request(
        client.chat_url, data=json.dumps(body).encode(), method="POST",
        headers={**client._headers(), "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        payload = json.loads(resp.read().decode())
    usage = payload.get("usage") or {}
    text = (payload["choices"][0]["message"].get("content") or "").strip()
    return (text, int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)))


def score(key: str, answer: str) -> bool:
    low = answer.lower().replace("\n", " ")
    if key == "title":
        return PANEL_TITLE.lower() in low
    if key == "lights":
        return (f"red={RED_LIGHTS}" in low.replace(" ", "")
                and f"green={GREEN_LIGHTS}" in low.replace(" ", ""))
    if key == "readout":
        return READOUT in low
    return False


def main() -> int:
    settings = Settings.from_env()
    settings.require_api()
    image = build_panel()
    out_dir = ROOT / "artifacts" / "vision_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "panel.png").write_bytes(image)
    data_url = "data:image/png;base64," + base64.b64encode(image).decode()

    client = DeepSeekClient(settings.api_key, settings.model, settings.base_url)
    results = []
    for key, question in QUESTIONS:
        try:
            answer, tin, tout = ask(client, data_url, question)
        except (LLMError, urllib.error.HTTPError, KeyError) as exc:
            detail = getattr(exc, "read", lambda: b"")()
            results.append({"key": key, "ok": False, "error": str(exc)[:300],
                            "body": detail.decode("utf-8", "replace")[:300]})
            continue
        results.append({"key": key, "ok": score(key, answer), "answer": answer[:120],
                        "in": tin, "out": tout})

    payload = {
        "model": settings.model,
        "image": str(out_dir / "panel.png"),
        "ground_truth": {"title": PANEL_TITLE, "readout": READOUT,
                         "red": RED_LIGHTS, "green": GREEN_LIGHTS},
        "results": results,
        "vision_supported": all(r["ok"] for r in results),
    }
    (out_dir / "smoke.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for r in results:
        if "error" in r:
            print(f"[{r['key']}] ERROR {r['error'][:200]}")
        else:
            print(f"[{r['key']}] {'PASS' if r['ok'] else 'FAIL'} "
                  f"answer={r['answer'][:60]!r} tokens={r['in']}+{r['out']}")
    print(f"\nvision_supported = {payload['vision_supported']}")
    return 0 if payload["vision_supported"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
