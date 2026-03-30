"""
pdf_loader.py — loads the local PDF catalog files that already exist in the
workspace (Course Pages/, Program Pages/, Academic Policy Page/).

This is an alternative ingestion path to the web scraper.  Since the PDFs
are already present, this is the fastest way to build the index — no
internet connection or URL verification needed.

Usage (called by build_index.py when PDFs are detected):
    from src.ingestion.pdf_loader import load_local_pdfs
    docs = load_local_pdfs()
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np  # noqa: F401 — imported here so the error is clear if missing

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

logger = logging.getLogger(__name__)

# Map folder names → doc_type
# NOTE: "Additional Pages" excluded — those are massive admissions/registration
# documents not relevant to course prerequisite planning.
_FOLDER_TYPE: Dict[str, str] = {
    "Course Pages":         "course",
    "Program Pages":        "program",
    "Academic Policy Page": "policy",
}

def _extract_text_pymupdf(path: Path) -> str:
    """
    Primary extractor: PyMuPDF text layer.
    Returns non-empty string only for text-based PDFs.
    """
    try:
        import fitz
        doc = fitz.open(str(path))
        texts = [page.get_text() for page in doc]
        return "\n\n".join(t for t in texts if t.strip())
    except Exception:
        return ""


def _extract_text_ocr(path: Path) -> str:
    """
    OCR extractor for image-based PDFs.

    Priority:
      1. pytesseract  — 5-8x faster than EasyOCR; auto-detected if Tesseract is
                        installed on the OS. Windows installer:
                        https://github.com/UB-Mannheim/tesseract/wiki
                        Then: pip install pytesseract pillow
      2. EasyOCR      — pure-Python fallback (no OS install needed, already
                        installed). Slower but works out-of-the-box.

    Both backends render pages at 120 DPI via PyMuPDF (2x faster than 150 DPI
    with negligible quality loss for single-page printed catalog PDFs).
    """
    import fitz
    import numpy as np

    # ── Detect best available OCR backend (once per process) ──────────────────
    _tesseract_ok = False
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        _tesseract_ok = True
        logger.info("  OCR backend: pytesseract (fast)")
    except Exception:
        logger.info("  OCR backend: EasyOCR (pytesseract not found)")

    doc = fitz.open(str(path))
    page_texts: List[str] = []

    for page_num, page in enumerate(doc):
        # 120 DPI: ~2x faster than 150 DPI, still sharp for printed catalog text
        mat = fitz.Matrix(120 / 72, 120 / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

        if _tesseract_ok:
            # pytesseract path — render to PNG bytes, pass to PIL
            import io
            from PIL import Image
            img_pil = Image.open(io.BytesIO(pix.tobytes("png")))
            page_text = pytesseract.image_to_string(
                img_pil, lang="eng", config="--psm 6"
            ).strip()
        else:
            # EasyOCR path — render to numpy array
            global _ocr_reader
            if _ocr_reader is None:
                logger.info(
                    "  Initialising EasyOCR (English) — first call downloads ~100 MB …"
                )
                import easyocr
                _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                logger.info("  EasyOCR ready.")
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            results = _ocr_reader.readtext(img, detail=0, paragraph=True)
            page_text = " ".join(results).strip()

        if page_text:
            page_texts.append(page_text)
        logger.debug("    Page %d: %d chars", page_num + 1, len(page_text))

    return "\n\n".join(page_texts)


# Module-level EasyOCR reader (loaded lazily, reused across all PDFs)
_ocr_reader = None


def _extract_pdf_text(path: Path) -> str:
    """
    Try text extraction first; fall back to OCR if the PDF is image-based.
    """
    # Step 1: fast text-layer extraction
    text = _extract_text_pymupdf(path)
    if len(text.strip()) > 100:          # threshold: 100 chars = real content
        return text

    # Step 2: OCR for image-based / scanned PDFs
    logger.info("  Text layer empty — running OCR on %s …", path.name)
    return _extract_text_ocr(path)


def _stem_to_doc_id(stem: str) -> str:
    """Convert a filename stem to a doc_id without spaces."""
    return re.sub(r"[^A-Za-z0-9_]", "_", stem).strip("_")


def _normalise_text(text: str) -> str:
    """Light normalisation: collapse blank lines, expand abbreviations."""
    abbrevs = {
        r"\bPrereq\.": "Prerequisite:",
        r"\bCo-req\.": "Co-requisite:",
    }
    for pat, rep in abbrevs.items():
        text = re.sub(pat, rep, text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_local_pdfs(base_dir: Optional[Path] = None) -> List[Dict]:
    """
    Scan the workspace for PDF files in known catalog folders and extract
    their text into structured document dicts.

    Returns a list of document dicts compatible with the chunker.
    """
    if base_dir is None:
        base_dir = config.BASE_DIR

    documents: List[Dict] = []
    metadata_updates: Dict = {}

    # Load existing metadata if present
    existing_meta: Dict = {}
    if config.METADATA_FILE.exists():
        existing_meta = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))

    for folder_name, doc_type in _FOLDER_TYPE.items():
        folder = base_dir / folder_name
        if not folder.exists():
            continue

        pdf_files = sorted(folder.glob("*.pdf"))
        if not pdf_files:
            continue

        logger.info("  Found %d PDFs in '%s'", len(pdf_files), folder_name)

        for pdf_path in pdf_files:
            doc_id = _stem_to_doc_id(pdf_path.stem)

            # Skip if already processed (avoids re-running slow OCR)
            out_path_check = config.PROCESSED_DIR / f"{doc_id}.json"
            if out_path_check.exists():
                logger.info("  ↷ %s already processed — skipping.", pdf_path.name)
                existing = json.loads(out_path_check.read_text(encoding="utf-8"))
                documents.append(existing)
                metadata_updates[doc_id] = existing
                continue

            try:
                raw_text = _extract_pdf_text(pdf_path)
            except Exception as exc:
                logger.warning("  ✗ %s  PDF extraction failed: %s", pdf_path.name, exc)
                continue

            if not raw_text.strip():
                logger.warning("  ✗ %s  Empty text — skipping.", pdf_path.name)
                continue

            clean_text = _normalise_text(raw_text)

            doc = {
                "doc_id":        doc_id,
                "title":         pdf_path.stem,
                "url":           str(pdf_path.relative_to(base_dir)).replace("\\", "/"),
                "doc_type":      doc_type,
                "date_accessed": config.DATE_ACCESSED,
                "text":          clean_text,
            }
            documents.append(doc)

            # Save processed JSON alongside existing processed docs
            out_path = config.PROCESSED_DIR / f"{doc_id}.json"
            out_path.write_text(
                json.dumps(doc, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            metadata_updates[doc_id] = {
                "doc_id":       doc_id,
                "title":        pdf_path.stem,
                "url":          doc["url"],
                "doc_type":     doc_type,
                "date_accessed": config.DATE_ACCESSED,
            }
            logger.info("  ✓ %s  (%d chars)", pdf_path.name, len(clean_text))

    # Merge metadata
    existing_meta.update(metadata_updates)
    config.METADATA_FILE.write_text(
        json.dumps(existing_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("PDF loading complete — %d documents extracted.", len(documents))
    return documents


def has_local_pdfs(base_dir: Optional[Path] = None) -> bool:
    """Return True if any local PDF files are found in the catalog folders."""
    if base_dir is None:
        base_dir = config.BASE_DIR
    for folder_name in _FOLDER_TYPE:
        folder = base_dir / folder_name
        if folder.exists() and any(folder.glob("*.pdf")):
            return True
    return False
