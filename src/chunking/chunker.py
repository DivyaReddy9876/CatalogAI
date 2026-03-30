"""
Document chunker — converts cleaned JSON documents into LangChain Documents
with rich metadata for precise citation and retrieval.

Chunking strategy
-----------------
- Splitter : RecursiveCharacterTextSplitter
- Chunk size   : 500 tokens  (approx chars: 500 × 4 ≈ 2000)
  Rationale    : Large enough to capture a full prerequisite statement plus
                 surrounding context, without losing the course code or grade
                 requirement that often appears in the same paragraph.
- Overlap      : 100 tokens  (~400 chars)
  Rationale    : A prerequisite sentence cut at a chunk boundary still appears
                 in the following chunk, preventing silent information loss.
- Split order  : ["\\n\\n", "\\n", ". ", " "] — respects paragraph > sentence > word.
- Section heading detection: extracted from the nearest `=====` separator
  inserted by cleaner.py, so every chunk carries the section it came from.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

logger = logging.getLogger(__name__)

# Approximate chars-per-token ratio for the MiniLM tokenizer
_CHARS_PER_TOKEN = 4

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE * _CHARS_PER_TOKEN,
    chunk_overlap=config.CHUNK_OVERLAP * _CHARS_PER_TOKEN,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
    is_separator_regex=False,
)

# Regex to detect the heading marker inserted by cleaner.py
_HEADING_RE = re.compile(r"={60}\n(.+?)\n={60}", re.MULTILINE)


def _extract_section_heading(text_before: str) -> str:
    """Return the last section heading that appears before the chunk text."""
    matches = list(_HEADING_RE.finditer(text_before))
    if matches:
        return matches[-1].group(1).strip().title()
    return "General"


def chunk_document(doc: dict) -> List[Document]:
    """
    Split one cleaned document dict into a list of LangChain Documents,
    each carrying full citation metadata.
    """
    full_text: str = doc.get("text", "").strip()
    if not full_text:
        logger.warning("  ⚠ %s has empty text — skipped.", doc.get("doc_id", "?"))
        return []

    raw_chunks = _SPLITTER.split_text(full_text)
    documents: List[Document] = []

    for idx, chunk_text in enumerate(raw_chunks):
        # Find where this chunk starts in the full text to detect its heading
        pos = full_text.find(chunk_text[:80])  # use first 80 chars as anchor
        text_before = full_text[:pos] if pos != -1 else full_text

        section = _extract_section_heading(text_before)

        documents.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "chunk_id":        f"{doc['doc_id']}_chunk_{idx}",
                    "doc_id":          doc["doc_id"],
                    "title":           doc["title"],
                    "url":             doc["url"],
                    "section_heading": section,
                    "doc_type":        doc["doc_type"],
                    "date_accessed":   doc["date_accessed"],
                    "chunk_index":     idx,
                },
            )
        )

    logger.debug("  %s → %d chunks", doc["doc_id"], len(documents))
    return documents


def chunk_all_documents() -> List[Document]:
    """
    Load every processed JSON file from ``data/processed/`` and chunk them.

    Returns the full list of LangChain Documents ready for embedding.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    json_files = sorted(config.PROCESSED_DIR.glob("*.json"))
    if not json_files:
        logger.error("No processed JSON files found — run cleaner first.")
        return []

    logger.info("Chunking %d documents …", len(json_files))

    all_docs: List[Document] = []
    for path in json_files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            chunks = chunk_document(doc)
            all_docs.extend(chunks)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  ✗ %s  Chunking failed: %s", path.name, exc)

    logger.info(
        "Chunking complete — %d chunks from %d documents.",
        len(all_docs), len(json_files),
    )
    return all_docs


if __name__ == "__main__":
    docs = chunk_all_documents()
    print(f"\nTotal chunks: {len(docs)}")
    if docs:
        print("\nSample chunk metadata:")
        print(json.dumps(docs[0].metadata, indent=2))
        print("\nSample chunk text (first 300 chars):")
        print(docs[0].page_content[:300])
