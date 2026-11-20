"""Multimodal evidence: attachment, perception, and measured compression.

Three things live here, in increasing order of ambition:

1. **Attachment** — an image becomes a content-addressed ``IMAGE`` evidence row,
   and any derived representation (thumbnail) is its own row with a
   ``derived_from`` pointer. The original is never overwritten. This is the
   direct opposite of the reference repo that resizes images in place.

2. **Perception** — a ``Perceiver`` turns bytes into a caption. The shipped
   perceiver is deliberately *not* a vision-language model: it extracts facts
   that are true by construction (dimensions, dominant colours, colour-count),
   and declares ``is_semantic=False``. A real VLM plugs into the same protocol.
   No captions are invented, so nothing downstream can cite a hallucination as
   evidence.

3. **Compression study** — a measured table of representation size vs. how much
   verifiable information survives. Token costs for images are *estimates* and
   are labelled as such; the fidelity numbers are measured by decoding the
   derived image and comparing it with the original.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from .evidence import EvidenceStore
from .models import Evidence, EvidenceKind, Experience
from .llm import estimate_tokens


class PillowMissing(RuntimeError):
    pass


def _pil():
    try:
        from PIL import Image  # noqa: PLC0415
        return Image
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise PillowMissing(
            "Pillow is required for image handling: pip install 'evomemory[images]'"
        ) from exc


# --------------------------------------------------------------------------
# perception
# --------------------------------------------------------------------------


@dataclass
class VisualDescription:
    caption: str
    tags: List[str] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)
    perceiver: str = ""
    is_semantic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Perceiver(Protocol):
    name: str

    def describe(self, image_bytes: bytes) -> VisualDescription: ...


def pixel_facts(image_bytes: bytes) -> Dict[str, Any]:
    """Facts that survive any lossless pipeline: geometry and colour statistics."""
    Image = _pil()
    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB")
        small = rgb.resize((32, 32))
        raw = small.tobytes()
        pixels = [tuple(raw[i:i + 3]) for i in range(0, len(raw), 3)]
        n = len(pixels)
        mean = tuple(round(sum(p[i] for p in pixels) / n) for i in range(3))
        distinct = len(set(pixels))
        # coarse hue bucket from the dominant channel
        dominant = ("red", "green", "blue")[mean.index(max(mean))]
    return {
        "width": img.width,
        "height": img.height,
        "aspect": round(img.width / img.height, 4),
        "mean_rgb": list(mean),
        "distinct_colors_32x32": distinct,
        "dominant_channel": dominant,
    }


class PixelFactsPerceiver:
    """Produces a caption that contains only measured facts.

    ``is_semantic = False`` is load-bearing: downstream reports must not treat
    this as a vision model's understanding.
    """

    name = "pixel-facts-v1"

    def describe(self, image_bytes: bytes) -> VisualDescription:
        facts = pixel_facts(image_bytes)
        caption = (
            f"image {facts['width']}x{facts['height']} "
            f"(aspect {facts['aspect']}), dominant channel {facts['dominant_channel']}, "
            f"mean rgb {tuple(facts['mean_rgb'])}, "
            f"{facts['distinct_colors_32x32']} distinct colours in a 32x32 sample"
        )
        return VisualDescription(
            caption=caption,
            tags=[facts["dominant_channel"], f"{facts['width']}x{facts['height']}"],
            facts=facts,
            perceiver=self.name,
            is_semantic=False,
        )


class SidecarCaptionPerceiver:
    """Reads a caption stored in PNG text metadata.

    Exists so tests can exercise the *semantic* path (a caption that says what
    the image means) without pretending a model produced it.
    """

    name = "sidecar-caption-v1"

    def describe(self, image_bytes: bytes) -> VisualDescription:
        Image = _pil()
        with Image.open(io.BytesIO(image_bytes)) as img:
            caption = (img.info or {}).get("Description", "")
            facts = pixel_facts(image_bytes)
        return VisualDescription(
            caption=caption or PixelFactsPerceiver().describe(image_bytes).caption,
            tags=[],
            facts=facts,
            perceiver=self.name,
            is_semantic=bool(caption),
        )


# --------------------------------------------------------------------------
# thumbnails
# --------------------------------------------------------------------------


def make_thumbnail(image_bytes: bytes, size: int = 448,
                   mode: str = "letterbox", quality: int = 85) -> bytes:
    """Resize to a ``size``x``size`` square.

    ``mode="squash"`` reproduces what the reference implementation does (a plain
    resize that distorts aspect ratio) and is kept **only** so the distortion can
    be measured. ``mode="letterbox"`` preserves geometry.
    """
    Image = _pil()
    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB")
        if mode == "squash":
            out = rgb.resize((size, size))
        elif mode == "letterbox":
            rgb.thumbnail((size, size))
            out = Image.new("RGB", (size, size), (0, 0, 0))
            out.paste(rgb, ((size - rgb.width) // 2, (size - rgb.height) // 2))
        elif mode == "crop":
            side = min(rgb.width, rgb.height)
            left = (rgb.width - side) // 2
            top = (rgb.height - side) // 2
            out = rgb.crop((left, top, left + side, top + side)).resize((size, size))
        else:
            raise ValueError(f"unknown thumbnail mode {mode!r}")
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()


def estimate_image_tokens(width: int, height: int,
                          patch: int = 16, merge: int = 2) -> int:
    """Visual-token estimate for a patch-based VLM (Qwen-VL family convention).

    This is an **estimate**, not a measurement: the exact count depends on the
    model's processor. It is used to compare representations with each other,
    never reported as a provider-measured number.
    """
    side_patches = math.ceil(width / patch) * math.ceil(height / patch)
    return max(1, side_patches // (merge * merge))


# --------------------------------------------------------------------------
# attachment
# --------------------------------------------------------------------------


@dataclass
class AttachedImage:
    original: Evidence
    thumbnail: Evidence
    caption_evidence: Evidence
    description: VisualDescription


class MultimodalMemory:
    """Attaches images to experiences without ever mutating the originals."""

    def __init__(self, evidence: EvidenceStore,
                 perceiver: Optional[Perceiver] = None,
                 thumb_size: int = 448,
                 thumb_mode: str = "letterbox"):
        self.evidence = evidence
        self.perceiver = perceiver or PixelFactsPerceiver()
        self.thumb_size = thumb_size
        self.thumb_mode = thumb_mode

    def attach(self, image_bytes: bytes, *, mime: str = "image/png",
               metadata: Optional[Dict[str, Any]] = None) -> AttachedImage:
        original = self.evidence.put(image_bytes, EvidenceKind.IMAGE, mime=mime,
                                     metadata=metadata or {})
        thumb_bytes = make_thumbnail(image_bytes, self.thumb_size, self.thumb_mode)
        description = self.perceiver.describe(image_bytes)
        thumbnail = self.evidence.put(
            thumb_bytes, EvidenceKind.THUMBNAIL, mime="image/png",
            metadata={"size": self.thumb_size, "mode": self.thumb_mode,
                      "caption": description.caption},
            derived_from=original.evidence_id)
        caption_ev = self.evidence.put_text(
            description.caption, kind=EvidenceKind.TEXT,
            metadata={"perceiver": description.perceiver,
                      "is_semantic": description.is_semantic,
                      "tags": description.tags,
                      "facts": description.facts},
            derived_from=original.evidence_id)
        return AttachedImage(original, thumbnail, caption_ev, description)

    def attach_to(self, exp: Experience, image_bytes: bytes, **kwargs: Any) -> AttachedImage:
        attached = self.attach(image_bytes, **kwargs)
        for ev in (attached.original, attached.thumbnail, attached.caption_evidence):
            if ev.evidence_id not in exp.evidence_refs:
                exp.evidence_refs.append(ev.evidence_id)
        return attached

    def caption_for(self, experience: Experience) -> str:
        parts: List[str] = []
        for ref in experience.evidence_refs:
            try:
                ev = self.evidence.get(ref)
            except KeyError:
                continue
            caption = ev.metadata.get("caption", "")
            if caption:
                parts.append(caption)
        return " ".join(parts)


# --------------------------------------------------------------------------
# compression study
# --------------------------------------------------------------------------


@dataclass
class RepresentationCost:
    representation: str
    tokens: int
    bytes_len: int
    token_is_estimate: bool = True
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def representation_costs(image_bytes: bytes, caption: str,
                         thumb_size: int = 448) -> List[RepresentationCost]:
    """Token cost of the three representations the brief asks us to compare."""
    facts = pixel_facts(image_bytes)
    full = estimate_image_tokens(facts["width"], facts["height"])
    thumb = estimate_image_tokens(thumb_size, thumb_size)
    caption_tokens = estimate_tokens(caption)
    return [
        RepresentationCost(
            "caption_only", caption_tokens, len(caption.encode("utf-8")),
            notes="text only; geometry and colour facts are lost unless written out"),
        RepresentationCost(
            "caption_plus_thumbnail", caption_tokens + thumb,
            len(caption.encode("utf-8")) + len(make_thumbnail(image_bytes, thumb_size)),
            notes=f"{thumb} visual tokens at {thumb_size}px (estimated)"),
        RepresentationCost(
            "caption_plus_original", caption_tokens + full,
            len(caption.encode("utf-8")) + len(image_bytes),
            notes=f"{full} visual tokens at native size (estimated)"),
    ]


def content_box_aspect(image_bytes: bytes, padding: Tuple[int, int, int] = (0, 0, 0),
                       tolerance: int = 24) -> Optional[float]:
    """Aspect ratio of the region that is not padding colour.

    Canvas aspect is the wrong measure for a letterboxed thumbnail: the file is
    square either way. What matters is whether the *content* was squashed, so the
    bounding box of pixels that differ from the padding colour is what gets
    compared.

    The padding colour is explicit rather than inferred. Inferring it (e.g. from
    the corner pixel) makes the metric inconsistent between representations: the
    corner of the original is the image's own background, while the corner of a
    letterboxed thumbnail is the pad.
    """
    Image = _pil()
    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        px = rgb.load()
        minx, miny, maxx, maxy = w, h, -1, -1
        step = max(1, min(w, h) // 96)
        for y in range(0, h, step):
            for x in range(0, w, step):
                p = px[x, y]
                if (abs(p[0] - padding[0]) + abs(p[1] - padding[1])
                        + abs(p[2] - padding[2])) > tolerance:
                    if x < minx:
                        minx = x
                    if x > maxx:
                        maxx = x
                    if y < miny:
                        miny = y
                    if y > maxy:
                        maxy = y
        if maxx < 0 or maxy < 0:
            # nothing differs from the padding colour: the frame is uniform, so its
            # content aspect is the canvas aspect
            return round(w / max(h, 1), 4)
        return round((maxx - minx + 1) / max(1, (maxy - miny + 1)), 4)


def fidelity(original_bytes: bytes, derived_bytes: bytes) -> Dict[str, Any]:
    """Measure what a derived representation preserves. No model involved."""
    a = pixel_facts(original_bytes)
    b = pixel_facts(derived_bytes)
    colour_a = a["mean_rgb"]
    colour_b = b["mean_rgb"]
    distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(colour_a, colour_b)))

    content_a = content_box_aspect(original_bytes)
    content_b = content_box_aspect(derived_bytes)
    content_error = (round(abs(content_a - content_b) / max(content_a, 1e-9), 4)
                     if content_a and content_b else None)
    return {
        "aspect_original": a["aspect"],
        "aspect_derived": b["aspect"],
        "content_aspect_original": content_a,
        "content_aspect_derived": content_b,
        "aspect_error": content_error,
        "mean_colour_distance": round(distance, 2),
        "dominant_channel_preserved": a["dominant_channel"] == b["dominant_channel"],
        "distinct_colour_ratio": round(
            b["distinct_colors_32x32"] / max(a["distinct_colors_32x32"], 1), 4),
    }


def compression_study(images: Sequence[Tuple[str, bytes]], caption: str = "",
                      sizes: Sequence[int] = (96, 224, 448, 896),
                      modes: Sequence[str] = ("squash", "letterbox", "crop"),
                      perceiver: Optional[Perceiver] = None) -> List[Dict[str, Any]]:
    """Grid over (thumbnail size x resize mode), measuring cost and fidelity.

    Returns one row per (image, size, mode). Every fidelity number is computed
    from pixels, so this is evidence, not a claim.
    """
    perceiver = perceiver or PixelFactsPerceiver()
    rows: List[Dict[str, Any]] = []
    for name, data in images:
        desc = perceiver.describe(data)
        cap = caption or desc.caption
        base = representation_costs(data, cap, thumb_size=448)
        for size in sizes:
            for mode in modes:
                thumb = make_thumbnail(data, size, mode)
                f = fidelity(data, thumb)
                rows.append({
                    "image": name,
                    "size": size,
                    "mode": mode,
                    "approx_tokens": estimate_image_tokens(size, size),
                    "bytes": len(thumb),
                    **f,                    "caption_tokens": estimate_tokens(cap),
                    "cost_caption_only": base[0].tokens,
                    "cost_caption_plus_thumb": base[1].tokens,
                    "cost_caption_plus_original": base[2].tokens,
                })
    return rows


def render_study(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| image | size | mode | est. visual tokens | bytes | aspect err | "
        "colour dist | dominant kept |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        err = r["aspect_error"]
        err_text = f"{err:.4f}" if err is not None else "n/a"
        lines.append(
            f"| {r['image']} | {r['size']} | {r['mode']} | {r['approx_tokens']} "
            f"| {r['bytes']} | {err_text} | {r['mean_colour_distance']:.2f} "
            f"| {'yes' if r['dominant_channel_preserved'] else 'NO'} |")
    return "\n".join(lines)
