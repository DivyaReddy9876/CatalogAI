"""
Web scraper — downloads catalog HTML pages and persists them to disk.

Design decisions
----------------
- One raw HTML file per catalog entry (named by doc_id).
- Polite: configurable inter-request delay + proper User-Agent.
- Resilient: exponential-backoff retries; failures are logged and skipped,
  not raised, so a single dead URL doesn't abort the whole pipeline.
- metadata.json is updated incrementally — safe to re-run.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from data.urls import ALL_URLS, CatalogEntry

logger = logging.getLogger(__name__)


# ── HTTP session with built-in retry ─────────────────────────────────────────

def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=config.SCRAPER_MAX_RETRIES,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": config.SCRAPER_USER_AGENT})
    return session


# ── Core scrape function ──────────────────────────────────────────────────────

def scrape_page(session: requests.Session, entry: CatalogEntry) -> bool:
    """
    Fetch one catalog URL and save its raw HTML.

    Returns True on success, False on any failure.
    """
    out_path: Path = config.RAW_DIR / f"{entry['doc_id']}.html"

    try:
        response = session.get(entry["url"], timeout=config.SCRAPER_TIMEOUT)
        response.raise_for_status()
        out_path.write_text(response.text, encoding="utf-8")
        logger.info("  ✓ %s  →  %s", entry["doc_id"], out_path.name)
        return True

    except requests.exceptions.HTTPError as exc:
        logger.warning("  ✗ %s  HTTP %s  (%s)", entry["doc_id"], exc.response.status_code, entry["url"])
    except requests.exceptions.ConnectionError:
        logger.warning("  ✗ %s  Connection error  (%s)", entry["doc_id"], entry["url"])
    except requests.exceptions.Timeout:
        logger.warning("  ✗ %s  Timeout after %ss", entry["doc_id"], config.SCRAPER_TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        logger.warning("  ✗ %s  Unexpected error: %s", entry["doc_id"], exc)

    return False


# ── Metadata writer ───────────────────────────────────────────────────────────

def _load_metadata() -> Dict:
    if config.METADATA_FILE.exists():
        return json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))
    return {}


def _save_metadata(metadata: Dict) -> None:
    config.METADATA_FILE.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ── Public entrypoint ─────────────────────────────────────────────────────────

def run_scraper(urls: List[CatalogEntry] | None = None) -> Dict[str, int]:
    """
    Scrape all catalog URLs (or a custom subset) to ``data/raw/``.

    Returns a summary dict: {"success": N, "failed": N}.
    """
    if urls is None:
        urls = ALL_URLS

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Scraping %d catalog pages …", len(urls))

    session  = _build_session()
    metadata = _load_metadata()
    success  = 0
    failed   = 0

    for entry in urls:
        ok = scrape_page(session, entry)
        if ok:
            metadata[entry["doc_id"]] = {
                "doc_id":       entry["doc_id"],
                "title":        entry["title"],
                "url":          entry["url"],
                "doc_type":     entry["doc_type"],
                "date_accessed": config.DATE_ACCESSED,
            }
            success += 1
        else:
            failed += 1

        time.sleep(config.SCRAPER_DELAY)

    _save_metadata(metadata)
    logger.info(
        "Scraping complete — %d succeeded, %d failed.  Metadata → %s",
        success, failed, config.METADATA_FILE,
    )
    return {"success": success, "failed": failed}


if __name__ == "__main__":
    run_scraper()
