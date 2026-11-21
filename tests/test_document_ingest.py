"""PDF, image and Markdown ingestion with page and asset provenance."""
import io
import os
from pathlib import Path

import pytest

from evomemory.document_parser import markdown_images, ocr_image
from evomemory.embeddings import LexicalHashEmbedder
from evomemory.kb import KnowledgeBase, split_document


def test_markdown_image_revision_is_reindexed_and_fenced_example_ignored(tmp_path, monkeypatch):
    corpus = tmp_path / "docs"
    corpus.mkdir()
    image = corpus / "alarm.png"
    image.write_bytes(b"old image")
    md = corpus / "guide.md"
    md.write_text("# Guide\n\n![alert](alarm.png)\n\n```md\n![example](missing.png)\n```\n", encoding="utf-8")
    monkeypatch.setattr("evomemory.kb.ocr_image", lambda data: "ERROR-OLD" if data == b"old image" else "ERROR-NEW")
    kb = KnowledgeBase(tmp_path / "kb.db")
    assert kb.import_file(md, corpus_root=corpus) == "added"
    hit = kb.search_bm25("ERROR-OLD")[0]
    assert hit.asset_path == "alarm.png" and hit.start_line == 3
    assert "alarm.png" in hit.citation() and "OCR" in hit.citation()
    assert kb.import_file(md, corpus_root=corpus) == "skipped"
    image.write_bytes(b"new image")
    assert kb.import_file(md, corpus_root=corpus) == "updated"
    assert all("ERROR-OLD" not in hit.text for hit in kb.search_bm25("ERROR-OLD"))
    assert kb.search_bm25("ERROR-NEW")
    assert kb.stats()["superseded_chunks"] > 0
    kb.close()


def test_markdown_long_text_and_code_fence_preserve_content_and_lines():
    text = "# Guide\n\n" + "x" * 950 + "ENDMARKER\n\n```python\nfirst()\n\nsecond()\n```\n"
    chunks = split_document(text, target=300)
    assert any("ENDMARKER" in body for _, _, body in chunks)
    assert any("```python\nfirst()\n\nsecond()\n```" in body for _, _, body in chunks)
    assert all(1 <= start <= end <= len(text.splitlines()) for start, end, _ in chunks)


def test_failed_ocr_update_preserves_previous_searchable_revision(tmp_path, monkeypatch):
    image = tmp_path / "alarm.png"
    image.write_bytes(b"first")
    monkeypatch.setattr("evomemory.kb.ocr_image", lambda _: "ORIGINAL-418")
    kb = KnowledgeBase(tmp_path / "kb.db")
    assert kb.import_file(image) == "added"
    image.write_bytes(b"second")
    def unavailable(_):
        raise RuntimeError("OCR unavailable")
    monkeypatch.setattr("evomemory.kb.ocr_image", unavailable)
    with pytest.raises(RuntimeError, match="OCR unavailable"):
        kb.import_file(image)
    assert any("ORIGINAL-418" in hit.text for hit in kb.search_bm25("ORIGINAL-418"))
    kb.close()


def test_markdown_rejects_image_outside_corpus(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"not needed")
    doc = root / "guide.md"
    text = "![x](../secret.png)"
    with pytest.raises(ValueError, match="escapes"):
        markdown_images(text, doc, root)


def test_markdown_reference_and_html_images_keep_source_lines(tmp_path):
    image = tmp_path / "error capture.png"
    image.write_bytes(b"image")
    doc = tmp_path / "guide.md"
    text = ('# Guide\n![screen][capture]\n<img src="error%20capture.png">\n'
            '[capture]: <error capture.png>\n')
    resolved = markdown_images(text, doc, tmp_path)
    assert [(line, path.name) for line, path in resolved] == [(2, image.name), (3, image.name)]


def test_real_pdf_mixed_and_scanned_pages_are_searchable(tmp_path, monkeypatch):
    pytest.importorskip("pypdfium2")
    reportlab = pytest.importorskip("reportlab.pdfgen.canvas")
    from PIL import Image, ImageDraw
    from reportlab.lib.utils import ImageReader

    img = Image.new("RGB", (600, 140), "white")
    ImageDraw.Draw(img).text((20, 30), "ALARM-731", fill="black")
    pdf = tmp_path / "mixed.pdf"
    c = reportlab.Canvas(str(pdf), pagesize=(600, 800))
    c.drawString(40, 740, "NATIVE-942 release reference")
    c.drawImage(ImageReader(img), 40, 450, width=500, height=120)
    c.showPage()
    c.drawImage(ImageReader(img), 40, 450, width=500, height=120)
    c.save()
    # OCR stub isolates PDF rendering/page routing; actual OCR is checked separately.
    monkeypatch.setattr("evomemory.document_parser.ocr_image", lambda _: "ALARM-731")
    kb = KnowledgeBase(tmp_path / "kb.db")
    assert kb.import_file(pdf) == "added"
    sources = {(x.page_number, x.source_kind) for x in kb.chunks()}
    assert (1, "pdf_text") in sources and (1, "pdf_ocr") in sources and (2, "pdf_ocr") in sources
    hits = kb.search_bm25("ALARM-731", k=10)
    assert {h.page_number for h in hits} == {1, 2}
    assert all("p." in h.citation() for h in hits)
    assert kb.build_vectors(LexicalHashEmbedder()) == len(kb.chunks())
    kb.close()


def test_real_chinese_ocr_when_locally_available():
    from evomemory.document_parser import TESSDATA, _tesseract
    try:
        _tesseract()
    except RuntimeError:
        pytest.skip("Tesseract is not installed")
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if not font_path.exists() or not (TESSDATA / "chi_sim.traineddata").exists():
        pytest.skip("Chinese test font or traineddata is unavailable")
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGB", (1100, 240), "white")
    font = ImageFont.truetype(str(font_path), 52)
    ImageDraw.Draw(image).text((20, 40), "故障代号 ALPHA-731 立即回滚", font=font, fill="black")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    text = ocr_image(buf.getvalue())
    assert "ALPHA-731" in text and "回滚" in text
