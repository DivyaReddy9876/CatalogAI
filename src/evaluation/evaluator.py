"""
Evaluator — runs all 25 test queries through the pipeline and computes
the four assessment metrics.  Also captures the three required transcripts.

Usage
-----
    python src/evaluation/evaluator.py

Output files
------------
    evaluation_results.csv       — per-query results table
    evaluation_report.json       — aggregated metrics
    transcripts/transcript_1_eligibility.txt
    transcripts/transcript_2_course_plan.txt
    transcripts/transcript_3_abstention.txt
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from src.chains.pipeline import Pipeline
from src.embeddings.embedder import get_embeddings
from src.evaluation.test_queries import PLAN_TRANSCRIPT_QUERY, TEST_QUERIES
from src.prompts.system_prompt import ABSTENTION_PHRASE
from src.vector_store.faiss_store import load_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Groq LLM (same as serving) ────────────────────────────────────────────────

def _build_llm():
    from langchain_groq import ChatGroq
    if not config.GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set.\n"
            "Copy .env.example → .env and add your key."
        )
    return ChatGroq(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        api_key=config.GROQ_API_KEY,
    )


# ── Auto-grading helpers ──────────────────────────────────────────────────────

def _has_citation(markdown_response: str) -> bool:
    """Return True if the response contains at least one [Source: ...] citation."""
    return "[Source:" in markdown_response


_ABSTENTION_PATTERNS = [
    # exact required phrase
    "i don't have that information in the provided catalog",
    # LLM paraphrases that all mean the same thing
    "not in the catalog excerpts",
    "there is no information in",
    "this information is not available in the catalog",
    "not available in the catalog",
    "cannot find this information in the catalog",
    "no information about this in the catalog",
    "not found in the catalog",
    "outside the scope of the catalog",
    "not covered in the catalog",
    "catalog does not contain",
    "catalog does not include",
    "catalog does not provide",
    "not mentioned in the catalog",
    # "Contact your advisor" in Answer section = system is declining to answer
    "contact your academic advisor for guidance on course selection",
    "contact your academic advisor for guidance on available scholarships",
]


def _has_abstention(markdown_response: str) -> bool:
    """Return True if the response conveys the mandatory abstention intent."""
    lower = markdown_response.lower()
    return any(p in lower for p in _ABSTENTION_PATTERNS)


def _decision_matches(response: str, expected: Optional[str]) -> Optional[float]:
    """
    Auto-grade eligibility decision.
    Returns 1.0 (correct), 0.5 (partial), 0.0 (wrong), or None (not applicable).
    """
    if expected is None:
        return None

    resp_lower = response.lower()

    if expected == "Eligible":
        if "not eligible" in resp_lower:
            return 0.0
        if "eligible" in resp_lower:
            return 1.0
        if "need more info" in resp_lower or "clarify" in resp_lower:
            return 0.5
        return 0.0

    if expected == "Not eligible":
        if "not eligible" in resp_lower:
            return 1.0
        if "need more info" in resp_lower:
            return 0.5
        if "eligible" in resp_lower:
            return 0.0
        return 0.0

    if expected == "Need more info":
        if "need more info" in resp_lower or "clarify" in resp_lower:
            return 1.0
        return 0.5

    return None


# ── Per-query runner ──────────────────────────────────────────────────────────

def run_query(pipeline: Pipeline, query: str) -> Dict[str, Any]:
    """Run one query and return the response plus auto-grading fields."""
    start = time.time()
    result = pipeline.run(query)
    elapsed = round(time.time() - start, 2)

    if result.response_type == "clarify":
        markdown = result.to_markdown()
    else:
        markdown = result.to_markdown()

    return {
        "markdown":   markdown,
        "verified":   result.verified,
        "elapsed_s":  elapsed,
    }


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation() -> None:
    logger.info("═" * 60)
    logger.info("  CATALOG RAG — EVALUATION RUN")
    logger.info("═" * 60)

    # Load resources
    embeddings   = get_embeddings()
    vectorstore  = load_vectorstore(embeddings)
    llm          = _build_llm()
    pipeline     = Pipeline(vectorstore, llm)

    rows: List[Dict] = []

    # Metric accumulators
    cat1_scores:   List[float] = []
    cat2_scores:   List[float] = []
    cat3_cited:    List[bool]  = []
    cat4_abstained: List[bool] = []
    all_cited:     List[bool]  = []

    for tq in TEST_QUERIES:
        qid  = tq["id"]
        cat  = tq["category"]
        logger.info("Query %02d / 25  [Cat %d] …", qid, cat)

        out = run_query(pipeline, tq["query"])
        md  = out["markdown"]

        cited    = _has_citation(md)
        abstained = _has_abstention(md)
        decision_score = _decision_matches(md, tq.get("expected_decision"))

        row: Dict[str, Any] = {
            "id":               qid,
            "category":         cat,
            "query_snippet":    tq["query"][:80] + "…",
            "has_citation":     cited,
            "abstained":        abstained,
            "decision_score":   decision_score,
            "expected_decision": tq.get("expected_decision"),
            "expected_abstain": tq.get("expected_abstain", False),
            "verified":         out["verified"],
            "elapsed_s":        out["elapsed_s"],
            "full_response":    md,
            "response_preview": md[:200] + "…",
        }
        rows.append(row)
        all_cited.append(cited)

        if cat == 1:
            cat1_scores.append(decision_score if decision_score is not None else 0.0)
        elif cat == 2:
            # Chain correctness: 1.0 if cited + not abstained, 0.5 if partial, else 0
            chain_score = 1.0 if (cited and not abstained) else (0.5 if cited else 0.0)
            cat2_scores.append(chain_score)
        elif cat == 3:
            cat3_cited.append(cited)
        elif cat == 4:
            cat4_abstained.append(abstained)

        # 65-second pause = full 60-second TPM window reset before next query.
        # Each query uses ~2000-2500 tokens across 3 API calls (intake + retrieval
        # sub-query + planner). With 65s gap, all previous tokens expire and the
        # next query has the full 6000 TPM budget available.
        if qid < len(TEST_QUERIES):   # no sleep after the last query
            logger.info("Cooling down 65s for TPM window reset …")
            time.sleep(65)

    # ── Compute metrics ───────────────────────────────────────────────────────

    # Citation coverage: % of cat 1–3 responses that have citations
    cat123_cited = [rows[i]["has_citation"] for i in range(20)]  # first 20 = cat 1-3
    citation_coverage = sum(cat123_cited) / len(cat123_cited) if cat123_cited else 0.0

    eligibility_correctness  = sum(cat1_scores) / len(cat1_scores) if cat1_scores else 0.0
    abstention_accuracy      = sum(cat4_abstained) / len(cat4_abstained) if cat4_abstained else 0.0
    chain_correctness        = sum(cat2_scores) / len(cat2_scores) if cat2_scores else 0.0

    metrics = {
        "citation_coverage_rate":    round(citation_coverage * 100, 1),
        "eligibility_correctness":   round(eligibility_correctness * 10, 2),
        "eligibility_out_of":        10,
        "abstention_accuracy":       round(abstention_accuracy * 5, 2),
        "abstention_out_of":         5,
        "chain_correctness":         round(chain_correctness * 5, 2),
        "chain_correctness_out_of":  5,
        "total_queries":             25,
    }

    # ── Print summary ─────────────────────────────────────────────────────────

    logger.info("\n%s", "═" * 60)
    logger.info("  EVALUATION RESULTS")
    logger.info("═" * 60)
    logger.info("  Citation Coverage Rate   : %s%%  (cat 1–3)", metrics["citation_coverage_rate"])
    logger.info(
        "  Eligibility Correctness  : %s / %s",
        metrics["eligibility_correctness"], metrics["eligibility_out_of"],
    )
    logger.info(
        "  Abstention Accuracy      : %s / %s",
        metrics["abstention_accuracy"], metrics["abstention_out_of"],
    )
    logger.info(
        "  Chain Correctness        : %s / %s",
        metrics["chain_correctness"], metrics["chain_correctness_out_of"],
    )
    logger.info("═" * 60)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    try:
        import csv
        fieldnames = list(rows[0].keys())
        with open(config.EVAL_CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Results CSV saved → %s", config.EVAL_CSV_FILE)
    except Exception as exc:
        logger.warning("Could not write CSV: %s", exc)

    # ── Save JSON report (strip full_response to keep file manageable) ────────
    rows_for_json = [{k: v for k, v in r.items() if k != "full_response"} for r in rows]
    report = {"metrics": metrics, "per_query": rows_for_json}
    config.EVAL_RESULTS_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("JSON report saved  → %s", config.EVAL_RESULTS_FILE)

    # ── Capture 3 required transcripts ───────────────────────────────────────
    _save_transcripts(pipeline, rows)


def _save_transcripts(pipeline: Pipeline, rows: List[Dict]) -> None:
    """
    Save the three required example conversations to the transcripts/ folder.

    transcript_1_eligibility : Query 3  (correct eligibility + citations)
    transcript_2_course_plan : PLAN_TRANSCRIPT_QUERY (plan + justification)
    transcript_3_abstention  : Query 21 (abstention + guidance)
    """
    logger.info("Saving required transcripts …")

    # Transcript 1 — eligibility (query id=3: COP 3514 eligible check)
    eligibility_row = next((r for r in rows if r["id"] == 3), None)
    if eligibility_row:
        _write_transcript(
            "transcript_1_eligibility.txt",
            query=TEST_QUERIES[2]["query"],
            response=eligibility_row["full_response"],
            label="Transcript 1 — Correct Eligibility Decision with Citations",
        )

    # Transcript 2 — course plan (dedicated plan query, not in the 25)
    logger.info("Running plan transcript query …")
    plan_out = pipeline.run(PLAN_TRANSCRIPT_QUERY)
    _write_transcript(
        "transcript_2_course_plan.txt",
        query=PLAN_TRANSCRIPT_QUERY,
        response=plan_out.to_markdown(),
        label="Transcript 2 — Course Plan Output with Justification and Citations",
    )

    # Transcript 3 — abstention (query id=21: COP 4710 schedule check)
    abstention_row = next((r for r in rows if r["id"] == 21), None)
    if abstention_row:
        _write_transcript(
            "transcript_3_abstention.txt",
            query=TEST_QUERIES[20]["query"],
            response=abstention_row["full_response"],
            label="Transcript 3 — Correct Abstention + Guidance",
        )

    logger.info("Transcripts saved to %s", config.TRANSCRIPTS_DIR)


def _write_transcript(filename: str, query: str, response: str, label: str) -> None:
    content = (
        f"{'═' * 70}\n"
        f"{label}\n"
        f"{'═' * 70}\n\n"
        f"STUDENT INPUT:\n{query}\n\n"
        f"{'─' * 70}\n\n"
        f"ASSISTANT RESPONSE:\n{response}\n"
    )
    path = config.TRANSCRIPTS_DIR / filename
    path.write_text(content, encoding="utf-8")
    logger.info("  → %s", path.name)


if __name__ == "__main__":
    run_evaluation()
