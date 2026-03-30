"""Rescore existing CSV with updated abstention detector. Zero API calls."""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.evaluation.evaluator import _has_abstention, _has_citation
from src.evaluation.test_queries import TEST_QUERIES

tq_map = {t["id"]: t for t in TEST_QUERIES}

with open("evaluation_results.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

cat1, cat2, cat3, cat4 = [], [], [], []
for r in rows:
    qid = int(r["id"])
    cat = int(r["category"])
    md  = r.get("full_response") or r.get("response_preview", "")

    cited     = _has_citation(md)
    abstained = _has_abstention(md)
    ds_raw    = r.get("decision_score")
    ds        = float(ds_raw) if ds_raw not in (None, "", "None") else None

    r["has_citation"] = cited
    r["abstained"]    = abstained

    if cat == 1:
        cat1.append(ds if ds is not None else 0.0)
    elif cat == 2:
        cat2.append(1.0 if (cited and not abstained) else (0.5 if cited else 0.0))
    elif cat == 3:
        cat3.append(cited)
    elif cat == 4:
        cat4.append(abstained)

cat123 = [r["has_citation"] for r in rows[:20]]
cit   = round(sum(cat123) / len(cat123) * 100, 1)
elig  = round(sum(cat1) / len(cat1) * 10, 2)
abst  = round(sum(cat4) / len(cat4) * 5, 2)
chain = round(sum(cat2) / len(cat2) * 5, 2)

print("=== RESCORED RESULTS ===")
print(f"Citation Coverage     : {cit}%")
print(f"Eligibility Correct.  : {elig} / 10")
print(f"Abstention Accuracy   : {abst} / 5")
print(f"Chain Correctness     : {chain} / 5")
print()
print("Cat 4 per-query abstention:")
for r in rows:
    if int(r["category"]) == 4:
        qid = r["id"]
        ab  = r["abstained"]
        prev = (r.get("full_response") or r.get("response_preview",""))[:120]
        print(f"  Q{qid}: abstained={ab}  |  {prev!r}")

# Save updated report
metrics = {
    "citation_coverage_rate":   cit,
    "eligibility_correctness":  elig,
    "eligibility_out_of":       10,
    "abstention_accuracy":      abst,
    "abstention_out_of":        5,
    "chain_correctness":        chain,
    "chain_correctness_out_of": 5,
    "total_queries":            25,
}
rows_for_json = [{k: v for k, v in r.items() if k != "full_response"} for r in rows]
report = {"metrics": metrics, "per_query": rows_for_json}
config.EVAL_RESULTS_FILE.write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"\nJSON report updated -> {config.EVAL_RESULTS_FILE}")

# Save updated CSV
fieldnames = list(rows[0].keys())
with open("evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"CSV updated -> evaluation_results.csv")
