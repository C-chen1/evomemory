"""Multimodal evidence: attachment lineage, perception honesty, measured compression."""

import io

import pytest

PIL = pytest.importorskip("PIL", reason="multimodal tests need Pillow")
from PIL import Image, ImageDraw  # noqa: E402

from evomemory.evidence import EvidenceStore  # noqa: E402
from evomemory.models import EvidenceKind, Experience  # noqa: E402
from evomemory.multimodal import (  # noqa: E402
    MultimodalMemory,
    PixelFactsPerceiver,
    SidecarCaptionPerceiver,
    compression_study,
    estimate_image_tokens,
    fidelity,
    make_thumbnail,
    pixel_facts,
    representation_costs,
)


def make_image(width=640, height=480, color=(200, 40, 40), text=None) -> bytes:
    """A synthetic image with a solid inner rectangle.

    The inner block matters: a flat fill gives a degenerate content bounding box,
    and the aspect-fidelity measurement becomes meaningless.
    """
    img = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([width // 4, height // 6, width * 3 // 4, height * 5 // 6],
                   fill=(20, 20, 20))
    buf = io.BytesIO()
    if text:
        img.save(buf, format="PNG", pnginfo=_pnginfo(text))
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def _pnginfo(text):
    from PIL import PngImagePlugin
    info = PngImagePlugin.PngInfo()
    info.add_text("Description", text)
    return info


# -- perception -----------------------------------------------------------


def test_pixel_facts_are_true_by_construction():
    data = make_image(640, 480, (200, 40, 40))
    facts = pixel_facts(data)
    assert facts["width"] == 640 and facts["height"] == 480
    assert facts["aspect"] == pytest.approx(640 / 480, rel=1e-3)
    assert facts["dominant_channel"] == "red"


def test_pixel_perceiver_declares_it_is_not_semantic():
    desc = PixelFactsPerceiver().describe(make_image())
    assert desc.is_semantic is False
    assert desc.perceiver == "pixel-facts-v1"
    assert "640x480" in desc.caption


def test_sidecar_perceiver_reads_a_caption_and_marks_it_semantic():
    data = make_image(text="a crimson alert banner above the table")
    desc = SidecarCaptionPerceiver().describe(data)
    assert desc.is_semantic is True
    assert "crimson" in desc.caption


# -- thumbnails -----------------------------------------------------------


def test_letterbox_preserves_aspect_ratio():
    data = make_image(300, 900, (10, 200, 10))
    thumb = make_thumbnail(data, 224, "letterbox")
    with Image.open(io.BytesIO(thumb)) as img:
        assert img.size == (224, 224)


def test_squash_distorts_and_letterbox_does_not():
    data = make_image(300, 900, (10, 200, 10))
    squash = fidelity(data, make_thumbnail(data, 224, "squash"))
    letter = fidelity(data, make_thumbnail(data, 224, "letterbox"))
    assert squash["aspect_error"] > 1.0, squash
    assert letter["aspect_error"] < 0.1, letter
    assert squash["aspect_error"] > letter["aspect_error"]


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        make_thumbnail(make_image(), 64, "warp")


# -- attachment -----------------------------------------------------------


def test_attach_creates_lineage_and_preserves_the_original(tmp_env):
    _, evidence = tmp_env
    mm = MultimodalMemory(evidence)
    data = make_image()
    attached = mm.attach(data, metadata={"source": "unit-test"})

    assert attached.original.kind is EvidenceKind.IMAGE
    assert attached.thumbnail.derived_from == attached.original.evidence_id
    assert attached.caption_evidence.derived_from == attached.original.evidence_id
    # the original bytes are byte-identical to what went in
    assert evidence.read_bytes(attached.original.evidence_id) == data
    assert len(evidence.children(attached.original.evidence_id)) == 2


def test_attach_to_appends_refs_without_duplicating(tmp_env):
    _, evidence = tmp_env
    mm = MultimodalMemory(evidence)
    exp = Experience(lesson="a lesson that is long enough to be plausible")
    mm.attach_to(exp, make_image(640, 480))
    first = len(exp.evidence_refs)
    mm.attach_to(exp, make_image(641, 480))   # a *different* image
    assert len(exp.evidence_refs) == first * 2
    # identical bytes twice -> same evidence ids, no growth
    same = make_image(100, 100)
    exp2 = Experience(lesson="another sufficiently long lesson entry")
    mm.attach_to(exp2, same)
    mm.attach_to(exp2, same)
    assert len(exp2.evidence_refs) == 3


def test_caption_for_joins_evidence_captions(tmp_env):
    _, evidence = tmp_env
    mm = MultimodalMemory(evidence)
    exp = Experience(lesson="x" * 30)
    mm.attach_to(exp, make_image(640, 480))
    caption = mm.caption_for(exp)
    assert "640x480" in caption


# -- cost -------------------------------------------------------------


def test_estimate_image_tokens_scales_with_area():
    assert estimate_image_tokens(448, 448) < estimate_image_tokens(896, 896)
    assert estimate_image_tokens(16, 16) >= 1


def test_representation_costs_are_ordered():
    data = make_image(1024, 1024)
    cap = "a wide banner image"
    costs = {c.representation: c for c in representation_costs(data, cap, 448)}
    assert costs["caption_only"].tokens < costs["caption_plus_thumbnail"].tokens
    assert costs["caption_plus_thumbnail"].tokens < costs["caption_plus_original"].tokens
    assert all(c.token_is_estimate for c in costs.values())


def test_compression_study_produces_measured_rows():
    data = make_image(400, 800, (30, 30, 200))
    rows = compression_study([("tall", data)], sizes=(96, 224), modes=("squash", "letterbox"))
    assert len(rows) == 4
    for row in rows:
        assert row["aspect_error"] >= 0
        assert row["approx_tokens"] > 0
        assert "dominant_channel_preserved" in row
    squash_errors = [r["aspect_error"] for r in rows if r["mode"] == "squash"]
    letter_errors = [r["aspect_error"] for r in rows if r["mode"] == "letterbox"]
    assert min(squash_errors) > max(letter_errors)
