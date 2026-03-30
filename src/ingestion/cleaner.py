"""
HTML cleaner — converts raw catalog HTML files into structured plain-text.

Strategy
--------
1. Parse with BeautifulSoup (lxml parser).
2. Remove boilerplate tags (nav, header, footer, script, style, aside).
3. Convert <table> to aligned plain text.
4. Preserve semantic structure: headings become UPPERCASE lines with a
   separator, which the chunker later uses to extract section_heading.
5. Normalise whitespace and expand common abbreviations.
6. Emit one JSON file per document to data/processed/.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

from bs4 import BeautifulSoup, Tag

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

logger = logging.getLogger(__name__)

# ── Abbreviation expansions ───────────────────────────────────────────────────
_ABBREV = {
    r"\bPrereq\.":       "Prerequisite:",
    r"\bPrereqs\.":      "Prerequisites:",
    r"\bCo-req\.":       "Co-requisite:",
    r"\bCo-reqs\.":      "Co-requisites:",
    r"\bCoreq\.":        "Co-requisite:",
    r"\bCoreqs\.":       "Co-requisites:",
    r"\bReq\.":          "Required:",
    r"\bCredit[s]?\.:":  "Credits:",
    r"\bhr[s]?\.":       "hours",
}

# Tags whose entire subtree should be stripped
_STRIP_TAGS = {
    "script", "style", "nav", "header", "footer",
    "aside", "noscript", "iframe", "form", "button",
    "meta", "link", "svg", "img",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_boilerplate(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()


def _table_to_text(table: Tag) -> str:
    """Convert an HTML table to a simple pipe-separated text representation."""
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["td", "th"])]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def _extract_text(soup: BeautifulSoup) -> str:
    """
    Walk the parse tree and produce clean plain text.
    Headings are capitalised and underlined with dashes.
    """
    parts: list[str] = []

    # Replace tables with text before get_text processing
    for table in soup.find_all("table"):
        table.replace_with(BeautifulSoup(_table_to_text(table), "html.parser"))

    # Walk remaining elements
    for element in soup.descendants:
        if not hasattr(element, "name"):
            continue  # NavigableString — skip; content captured by parent

        if element.name in {"h1", "h2", "h3", "h4"}:
            heading = element.get_text(separator=" ", strip=True)
            if heading:
                parts.append(f"\n{'='*60}\n{heading.upper()}\n{'='*60}")

        elif element.name in {"p", "li", "dd", "dt"}:
            text = element.get_text(separator=" ", strip=True)
            if text:
                parts.append(text)

        elif element.name in {"ul", "ol"}:
            pass  # individual <li> items handled above

        elif element.name == "br":
            parts.append("")

    return "\n".join(parts)


def _normalise(text: str) -> str:
    """Apply abbreviation expansions and whitespace normalisation."""
    for pattern, replacement in _ABBREV.items():
        text = re.sub(pattern, replacement, text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Core clean function ───────────────────────────────────────────────────────

def clean_html(
    html: str,
    doc_id: str,
    title: str,
    url: str,
    doc_type: str,
    date_accessed: str,
) -> Dict:
    """
    Clean a raw HTML string and return a structured document dict.

    The ``text`` field is what gets chunked and embedded.
    """
    soup = BeautifulSoup(html, "lxml")
    _strip_boilerplate(soup)

    # Try to find the main content area
    main = (
        soup.find("main")
        or soup.find("div", {"id": re.compile(r"content|main|catalog", re.I)})
        or soup.find("div", {"class": re.compile(r"content|main|catalog", re.I)})
        or soup.body
        or soup
    )

    raw_text = _extract_text(main)
    clean_text = _normalise(raw_text)

    return {
        "doc_id":        doc_id,
        "title":         title,
        "url":           url,
        "doc_type":      doc_type,
        "date_accessed": date_accessed,
        "text":          clean_text,
    }


# ── File-level driver ─────────────────────────────────────────────────────────

def clean_file(doc_id: str, metadata: Dict) -> Optional[Dict]:
    """
    Read ``data/raw/<doc_id>.html``, clean it, write ``data/processed/<doc_id>.json``.
    Returns the document dict or None on failure.
    """
    raw_path  = config.RAW_DIR / f"{doc_id}.html"
    out_path  = config.PROCESSED_DIR / f"{doc_id}.json"
    meta_entry = metadata.get(doc_id, {})

    if not raw_path.exists():
        logger.warning("  ✗ %s — raw HTML file not found, skipping.", doc_id)
        return None

    try:
        html = raw_path.read_text(encoding="utf-8")
        doc  = clean_html(
            html          = html,
            doc_id        = doc_id,
            title         = meta_entry.get("title", doc_id),
            url           = meta_entry.get("url", ""),
            doc_type      = meta_entry.get("doc_type", "course"),
            date_accessed = meta_entry.get("date_accessed", config.DATE_ACCESSED),
        )
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("  ✓ %s  →  %s", doc_id, out_path.name)
        return doc

    except Exception as exc:  # noqa: BLE001
        logger.warning("  ✗ %s  Cleaning failed: %s", doc_id, exc)
        return None


# ── Public entrypoint ─────────────────────────────────────────────────────────

def run_cleaner() -> Dict[str, int]:
    """
    Clean all raw HTML files that have a metadata entry.

    Returns summary {"success": N, "failed": N}.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not config.METADATA_FILE.exists():
        logger.error("metadata.json not found — run scraper first.")
        return {"success": 0, "failed": 0}

    metadata = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))
    logger.info("Cleaning %d documents …", len(metadata))

    success = failed = 0
    for doc_id in metadata:
        result = clean_file(doc_id, metadata)
        if result:
            success += 1
        else:
            failed += 1

    logger.info("Cleaning complete — %d succeeded, %d failed.", success, failed)
    return {"success": success, "failed": failed}


if __name__ == "__main__":
    run_cleaner()
