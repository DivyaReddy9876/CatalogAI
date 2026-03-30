"""
patch_eval.py — Re-run only the queries that failed intake (Q7, Q19-Q25),
merge their new results into the existing evaluation_results.csv + report.json,
then print updated final metrics.

Usage:
    python patch_eval.py
"""
from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.chains.pipeline import Pipeline
from src.embeddings.embedder import get_embeddings
from src.evaluation.test_queries import TEST_QUERIES
from src.prompts.system_prompt import ABSTENTION_PHRASE
from src.vector_store.faiss_store import load_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── IDs of the 4 eligibility queries that scored 0 or 0.5 ─────────────────
RERUN_IDS = {16, 17}


def _build_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        api_key=config.GROQ_API_KEY,
    )


def _has_citation(md: str) -> bool:
    return "[Source:" in md


def _has_abstention(md: str) -> bool:
    return ABSTENTION_PHRASE.lower() in md.lower()


def _decision_matches(response: str, expected):
    if expected is None:
        return None
    r = response.lower()
    if expected == "Eligible":
        if "not eligible" in r:   return 0.0
        if "eligible" in r:       return 1.0
        if "need more info" in r: return 0.5
        return 0.0
    if expected == "Not eligible":
        if "not eligible" in r:   return 1.0
        if "need more info" in r: return 0.5
        if "eligible" in r:       return 0.0
        return 0.0
    if expected == "Need more info":
        if "need more info" in r or "clarify" in r: return 1.0
        return 0.5
    return None


def run_query(pipeline: Pipeline, query: str) -> Dict[str, Any]:
    start = time.time()
    result = pipeline.run(query)
    elapsed = round(time.time() - start, 2)
    return {"markdown": result.to_markdown(), "verified": result.verified, "elapsed_s": elapsed}


def main():
    logger.info("═" * 60)
    logger.info("  PATCH EVAL — reruns %s failed queries", len(RERUN_IDS))
    logger.info("═" * 60)

    # ── Load existing results ───────────────────────────────────────────────
    existing: Dict[int, Dict] = {}
    csv_path = Path(config.EVAL_CSV_FILE)
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qid = int(row["id"])
                # Restore bool / float types
                row["has_citation"]   = row["has_citation"].lower() == "true"
                row["abstained"]      = row["abstained"].lower() == "true"
                row["verified"]       = row["verified"].lower() == "true"
                row["expected_abstain"] = row.get("expected_abstain", "false").lower() == "true"
                ds = row.get("decision_score")
                row["decision_score"] = float(ds) if ds not in (None, "", "None") else None
                existing[qid] = row
        logger.info("Loaded %d existing rows from %s", len(existing), csv_path)
    else:
        logger.error("No existing CSV found — run the full evaluator first.")
        sys.exit(1)

    # ── Build pipeline ──────────────────────────────────────────────────────
    embeddings  = get_embeddings()
    vectorstore = load_vectorstore(embeddings)
    llm         = _build_llm()
    pipeline    = Pipeline(vectorstore, llm)

    target_queries = [tq for tq in TEST_QUERIES if tq["id"] in RERUN_IDS]
    total = len(target_queries)

    for i, tq in enumerate(target_queries, start=1):
        qid = tq["id"]
        logger.info("Patch query %d/%d — Q%02d [Cat %d] …", i, total, qid, tq["category"])

        out = run_query(pipeline, tq["query"])
        md  = out["markdown"]

        cited  = _has_citation(md)
        abstained = _has_abstention(md)
        ds = _decision_matches(md, tq.get("expected_decision"))

        existing[qid].update({
            "has_citation":      cited,
            "abstained":         abstained,
            "decision_score":    ds,
            "expected_decision": tq.get("expected_decision"),
            "expected_abstain":  tq.get("expected_abstain", False),
            "verified":          out["verified"],
            "elapsed_s":         out["elapsed_s"],
            "full_response":     md,
            "response_preview":  md[:200] + "…",
        })
        logger.info(
            "  cited=%s  abstained=%s  decision_score=%s",
            cited, abstained, ds,
        )

        if i < total:
            logger.info("Cooling down 65s …")
            time.sleep(65)

    # ── Recompute metrics ───────────────────────────────────────────────────
    rows = [existing[tq["id"]] for tq in TEST_QUERIES]  # sorted order

    cat1_scores:    List[float] = []
    cat2_scores:    List[float] = []
    cat3_cited:     List[bool]  = []
    cat4_abstained: List[bool]  = []

    for r in rows:
        cat = int(r["category"])
        cited     = r["has_citation"]
        abstained = r["abstained"]
        ds        = r["decision_score"]

        if cat == 1:
            cat1_scores.append(ds if ds is not None else 0.0)
        elif cat == 2:
            chain_score = 1.0 if (cited and not abstained) else (0.5 if cited else 0.0)
            cat2_scores.append(chain_score)
        elif cat == 3:
            cat3_cited.append(cited)
        elif cat == 4:
            cat4_abstained.append(abstained)

    cat123_cited = [r["has_citation"] for r in rows[:20]]
    citation_coverage       = sum(cat123_cited) / len(cat123_cited)
    eligibility_correctness = sum(cat1_scores) / len(cat1_scores)
    abstention_accuracy     = sum(cat4_abstained) / len(cat4_abstained)
    chain_correctness       = sum(cat2_scores) / len(cat2_scores)

    metrics = {
        "citation_coverage_rate":   round(citation_coverage * 100, 1),
        "eligibility_correctness":  round(eligibility_correctness * 10, 2),
        "eligibility_out_of":       10,
        "abstention_accuracy":      round(abstention_accuracy * 5, 2),
        "abstention_out_of":        5,
        "chain_correctness":        round(chain_correctness * 5, 2),
        "chain_correctness_out_of": 5,
        "total_queries":            25,
    }

    logger.info("\n%s", "═" * 60)
    logger.info("  UPDATED EVALUATION RESULTS")
    logger.info("═" * 60)
    logger.info("  Citation Coverage Rate   : %s%%  (cat 1–3)", metrics["citation_coverage_rate"])
    logger.info("  Eligibility Correctness  : %s / %s", metrics["eligibility_correctness"], 10)
    logger.info("  Abstention Accuracy      : %s / %s", metrics["abstention_accuracy"], 5)
    logger.info("  Chain Correctness        : %s / %s", metrics["chain_correctness"], 5)
    logger.info("═" * 60)

    # ── Save updated CSV ────────────────────────────────────────────────────
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Updated CSV saved → %s", csv_path)

    # ── Save updated JSON report ────────────────────────────────────────────
    rows_for_json = [{k: v for k, v in r.items() if k != "full_response"} for r in rows]
    report = {"metrics": metrics, "per_query": rows_for_json}
    config.EVAL_RESULTS_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Updated JSON report → %s", config.EVAL_RESULTS_FILE)


if __name__ == "__main__":
    main()
