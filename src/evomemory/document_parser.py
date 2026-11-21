"""Extract searchable text from PDFs and images, preserving source locations.

OCR is a text transcription, not visual embedding or diagram understanding.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[2]
TESSDATA = ROOT / "cache" / "tessdata"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class TextSource:
    text: str
    kind: str
    page: Optional[int] = None
    asset: Optional[str] = None
    line: Optional[int] = None


def _tesseract() -> str:
    configured = os.environ.get("EVOMEMORY_TESSERACT")
    candidate = configured or shutil.which("tesseract")
    if not candidate and os.name == "nt":
        default = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Tesseract-OCR" / "tesseract.exe"
        if default.is_file():
            candidate = str(default)
    if not candidate or not Path(candidate).is_file():
        raise RuntimeError("OCR requires Tesseract; install it or set EVOMEMORY_TESSERACT")
    return str(candidate)


def ocr_image(image_bytes: bytes) -> str:
    """Run local OCR over image bytes; no remote model or temporary source file."""
    lang = os.environ.get("EVOMEMORY_OCR_LANG", "chi_sim+eng")
    data_dir = Path(os.environ.get("EVOMEMORY_TESSDATA", str(TESSDATA)))
    args = [_tesseract(), "stdin", "stdout", "-l", lang]
    if data_dir.is_dir():
        args.extend(["--tessdata-dir", str(data_dir)])
    try:
        result = subprocess.run(args, input=image_bytes, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=120, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("OCR timed out") from exc
    if result.returncode:
        raise RuntimeError("OCR failed: " + result.stderr.decode("utf-8", "replace")[:500])
    return result.stdout.decode("utf-8", "replace").strip()


def parse_pdf(raw: bytes) -> list[TextSource]:
    try:
        import pypdfium2 as pdfium
        from pypdfium2 import raw as pdfium_raw
    except ImportError as exc:
        raise RuntimeError("PDF import requires pypdfium2; install evomemory[documents]") from exc
    doc = pdfium.PdfDocument(raw)
    sources = []
    try:
        for i in range(len(doc)):
            page = doc[i]
            try:
                textpage = page.get_textpage()
                try:
                    native = textpage.get_text_range().strip()
                finally:
                    textpage.close()
                if native:
                    sources.append(TextSource(native, "pdf_text", page=i + 1))
                images = list(page.get_objects(filter=[pdfium_raw.FPDF_PAGEOBJ_IMAGE]))
                if images or not native:
                    bitmap = page.render(scale=2)
                    try:
                        img = bitmap.to_pil()
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                    finally:
                        bitmap.close()
                    scanned = ocr_image(buf.getvalue())
                    # Native text is authoritative where available; retain OCR lines
                    # that add image content and avoid exact duplicate text lines.
                    normalized = re.sub(r"\s+", "", native).casefold()
                    extra = [line for line in scanned.splitlines() if line.strip() and
                             re.sub(r"\s+", "", line).casefold() not in normalized]
                    if extra:
                        sources.append(TextSource("\n".join(extra), "pdf_ocr", page=i + 1))
                    elif not native:
                        warnings.warn(f"PDF page {i + 1} produced no OCR text", RuntimeWarning)
            finally:
                page.close()
    finally:
        doc.close()
    if not sources:
        raise ValueError("PDF contains no extractable text; no searchable content was imported")
    return sources


def markdown_images(text: str, doc_path: Path, root: Path) -> list[tuple[int, Path]]:
    """Resolve local Markdown image references, without fetching remote URLs."""
    out = []
    fence = None
    active_lines = []
    for line_number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker.group(1)[0]
            elif marker.group(1)[0] == fence:
                fence = None
            continue
        if fence:
            continue
        active_lines.append((line_number, line))
    references = {}
    for _, line in active_lines:
        definition = re.match(r"^\s*\[([^]]+)\]:\s*(?:<([^>]+)>|(\S+))", line)
        if definition:
            references[definition.group(1).strip().casefold()] = definition.group(2) or definition.group(3)
    for line_number, line in active_lines:
        targets = [(m.group(1) or m.group(2)) for m in re.finditer(
            r"!\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))(?:\s+[^)]*)?\)", line)]
        targets.extend(references[m.group(2).strip().casefold() or m.group(1).strip().casefold()]
                       for m in re.finditer(r"!\[([^]]*)\]\[([^]]*)\]", line)
                       if (m.group(2).strip().casefold() or m.group(1).strip().casefold()) in references)
        targets.extend(m.group(2) for m in re.finditer(
            r"<img\b[^>]*\bsrc\s*=\s*(['\"])(.*?)\1", line, flags=re.IGNORECASE))
        for target in targets:
            if "://" in target or target.startswith("data:"):
                warnings.warn(f"remote or embedded Markdown image was not imported at line {line_number}", RuntimeWarning)
                continue
            from urllib.parse import unquote
            asset = (doc_path.parent / unquote(target)).resolve()
            if not asset.is_relative_to(root.resolve()):
                raise ValueError("Markdown image reference escapes the corpus directory")
            if not asset.is_file() or asset.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Markdown image not found or unsupported: {target}")
            out.append((line_number, asset))
    return out
