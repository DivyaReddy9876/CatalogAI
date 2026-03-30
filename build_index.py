"""
build_index.py — one-shot pipeline that takes you from zero to a queryable
FAISS index in five steps.  Run this ONCE before starting the assistant.

    python build_index.py

Steps
-----
1  Scrape   → download 25+ catalog HTML pages to data/raw/
2  Clean    → strip boilerplate, normalise → data/processed/
3  Chunk    → split into 500-token overlapping chunks with metadata
4  Embed    → encode chunks with all-MiniLM-L6-v2 (downloads ~90 MB first run)
5  Index    → build FAISS index, save to vector_store/catalog_index/

Re-running is safe — each step is idempotent.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _step(n: int, label: str) -> None:
    logger.info("")
    logger.info("━" * 60)
    logger.info("  STEP %d/5 — %s", n, label)
    logger.info("━" * 60)


def main() -> None:
    start = time.time()

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   CATALOG RAG — INDEX BUILDER                          ║")
    logger.info("╚" + "═" * 58 + "╝")

    import config  # noqa: PLC0415 — imported here to keep top of file clean
    # ── Steps 1 & 2: Ingest ──────────────────────────────────────────────────
    from src.ingestion.pdf_loader import has_local_pdfs, load_local_pdfs

    # If processed/ already has files, skip PDF loading entirely
    existing_processed = list(config.PROCESSED_DIR.glob("*.json"))
    if existing_processed:
        _step(1, f"Skipped — {len(existing_processed)} documents already in data/processed/")
        _step(2, "Skipped — using existing processed files")
        doc_count = len(existing_processed)
        logger.info("  %d processed documents found — jumping to Step 3.", doc_count)
    elif has_local_pdfs():
        # Fast path — local PDFs already present (no internet needed)
        _step(1, "Loading local PDF catalog files")
        pdf_docs = load_local_pdfs()
        logger.info("  %d PDF documents loaded.", len(pdf_docs))
        if not pdf_docs:
            logger.error("No text extracted from PDFs — check pdf_loader.py.")
            sys.exit(1)
        _step(2, "Skipped — local PDFs used directly (no HTML cleaning needed)")
        doc_count = len(pdf_docs)
    else:
        # Fallback — scrape from the web
        _step(1, "Scraping catalog pages")
        from src.ingestion.scraper import run_scraper
        scrape_stats = run_scraper()
        logger.info(
            "  Scraping done: %d succeeded, %d failed.",
            scrape_stats["success"], scrape_stats["failed"],
        )
        if scrape_stats["success"] == 0:
            logger.error(
                "No pages scraped successfully.\n"
                "  → Check your internet connection.\n"
                "  → Verify URLs in data/urls.py against the live catalog.\n"
                "  → Update any 404 URLs and re-run."
            )
            sys.exit(1)

        _step(2, "Cleaning HTML files")
        from src.ingestion.cleaner import run_cleaner
        clean_stats = run_cleaner()
        logger.info(
            "  Cleaning done: %d succeeded, %d failed.",
            clean_stats["success"], clean_stats["failed"],
        )
        if clean_stats["success"] == 0:
            logger.error("No documents cleaned — cannot continue.")
            sys.exit(1)
        doc_count = clean_stats["success"]

    # ── Step 3: Chunk ─────────────────────────────────────────────────────────
    _step(3, "Chunking documents")
    from src.chunking.chunker import chunk_all_documents
    documents = chunk_all_documents()
    if not documents:
        logger.error("No chunks produced — cannot continue.")
        sys.exit(1)
    logger.info("  %d chunks created from %d documents.", len(documents), doc_count)

    # ── Step 4: Embed ─────────────────────────────────────────────────────────
    _step(4, "Loading embedding model")
    from src.embeddings.embedder import get_embeddings
    embeddings = get_embeddings()
    logger.info("  Embedding model ready.")

    # ── Step 5: Index ─────────────────────────────────────────────────────────
    _step(5, "Building FAISS index")
    from src.vector_store.faiss_store import build_and_save
    vectorstore = build_and_save(documents, embeddings)
    logger.info(
        "  FAISS index saved  (%d vectors, dim=%d).",
        vectorstore.index.ntotal, vectorstore.index.d,
    )

    elapsed = round(time.time() - start, 1)
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   BUILD COMPLETE  (%ss)                                ║", str(elapsed).ljust(6))
    logger.info("║                                                          ║")
    logger.info("║   Next steps:                                            ║")
    logger.info("║     python main.py          ← CLI chat                  ║")
    logger.info("║     python app.py           ← Gradio web demo           ║")
    logger.info("║     python src/evaluation/evaluator.py  ← run eval      ║")
    logger.info("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    main()
