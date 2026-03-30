# UML Course Planning Assistant — Agentic RAG

A LangChain-based Retrieval-Augmented Generation (RAG) assistant that answers
student course-planning questions **strictly grounded in the UML academic catalog**.
Every claim is cited. Unavailable information is acknowledged honestly.

---

## Architecture

```
BUILD (once)
  python build_index.py
    ┌─────────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐   ┌───────────┐
    │  Scraper    │→  │ Cleaner  │→  │ Chunker │→  │ Embedder │→  │   FAISS   │
    │ (requests + │   │(BS4/lxml)│   │500 tok/ │   │MiniLM-L6 │   │  Index    │
    │  BS4)       │   │          │   │100 ovlp │   │-v2 (384d)│   │(saved)    │
    └─────────────┘   └──────────┘   └─────────┘   └──────────┘   └───────────┘

SERVE (every query)
  Student message
      │
      ▼
  ┌──────────────┐     missing info?    ┌─────────────────────────┐
  │ Intake Chain │ ──────────────────→  │  Clarifying questions   │
  └──────────────┘                      └─────────────────────────┘
      │ complete profile
      ▼
  ┌──────────────────┐
  │ Retrieval Chain  │  MultiQueryRetriever (3 sub-queries) → MMR (k=6)
  └──────────────────┘
      │ top-8 chunks
      ▼
  ┌───────────────┐
  │ Planner Chain │  LLM → PlannerOutput (cited, structured)
  └───────────────┘
      │ draft answer
      ▼
  ┌─────────────────┐   verified?   ┌──────────────────────────┐
  │ Verifier Chain  │ ────────────→ │  Final answer (markdown)  │
  └─────────────────┘               └──────────────────────────┘
```

**Stack**

| Component        | Choice                              | Cost  |
|------------------|-------------------------------------|-------|
| Agent framework  | LangChain ≥ 0.2                     | Free  |
| Vector store     | FAISS (local)                       | Free  |
| Embeddings       | `all-MiniLM-L6-v2` (HuggingFace)   | Free  |
| LLM              | Groq `llama-3.1-8b-instant`         | Free  |
| Demo UI          | Gradio                              | Free  |

---

## Setup

### 1. Clone & install

```bash
git clone <your-repo-url>
cd catalog-rag
pip install -r requirements.txt
```

### 2. Set your Groq API key

```bash
cp .env.example .env
# Edit .env — add your key from https://console.groq.com (free, no credit card)
```

### 3. Build the index  *(run once)*

```bash
python build_index.py
```

This will:
- Scrape 25 UML catalog pages → `data/raw/`
- Clean HTML → `data/processed/`
- Chunk into 500-token segments → in memory
- Embed with MiniLM → `vector_store/catalog_index/`

> **If some URLs return 404:** UML periodically updates their catalog URLs.
> Open `data/urls.py`, find the failing `doc_id`, visit `https://www.uml.edu/catalog/`
> to get the current URL, update the entry, and re-run `python build_index.py`.

---

## Run

### CLI (interactive chat)

```bash
python main.py
```

### Gradio web demo

```bash
python app.py
# Opens at http://localhost:7860
```

### Evaluation (25-query test set)

```bash
python src/evaluation/evaluator.py
```

Produces:
- `evaluation_results.csv` — per-query results
- `evaluation_report.json` — aggregated metrics
- `transcripts/transcript_1_eligibility.txt`
- `transcripts/transcript_2_course_plan.txt`
- `transcripts/transcript_3_abstention.txt`

---

## Output Format

Every response follows the mandatory structure:

```
## Answer / Plan:
## Why (requirements/prereqs satisfied):
## Citations:
## Clarifying questions (if needed):
## Assumptions / Not in catalog:
```

Citation format:
```
[Source: COMP 4110 – Database Systems, https://www.uml.edu/catalog/courses/comp/4110/, Section: Prerequisites]
```

---

## Data Sources

| # | Document | URL | Type | Accessed |
|---|----------|-----|------|----------|
| 1 | COMP 1010 – Computing I | https://www.uml.edu/catalog/courses/comp/1010/ | course | 2026-03-29 |
| 2 | COMP 1020 – Computing II | https://www.uml.edu/catalog/courses/comp/1020/ | course | 2026-03-29 |
| 3 | COMP 2010 – Computing III | https://www.uml.edu/catalog/courses/comp/2010/ | course | 2026-03-29 |
| 4 | COMP 2030 – Discrete Structures | https://www.uml.edu/catalog/courses/comp/2030/ | course | 2026-03-29 |
| 5 | COMP 3010 – Org. of Programming Languages | https://www.uml.edu/catalog/courses/comp/3010/ | course | 2026-03-29 |
| 6 | COMP 3040 – Algorithms | https://www.uml.edu/catalog/courses/comp/3040/ | course | 2026-03-29 |
| 7 | COMP 3050 – Computer Architecture | https://www.uml.edu/catalog/courses/comp/3050/ | course | 2026-03-29 |
| 8 | COMP 3080 – Operating Systems | https://www.uml.edu/catalog/courses/comp/3080/ | course | 2026-03-29 |
| 9 | COMP 4040 – Theory of Computation | https://www.uml.edu/catalog/courses/comp/4040/ | course | 2026-03-29 |
| 10 | COMP 4080 – Computer Networks | https://www.uml.edu/catalog/courses/comp/4080/ | course | 2026-03-29 |
| 11 | COMP 4110 – Database Systems | https://www.uml.edu/catalog/courses/comp/4110/ | course | 2026-03-29 |
| 12 | COMP 4130 – Machine Learning | https://www.uml.edu/catalog/courses/comp/4130/ | course | 2026-03-29 |
| 13 | COMP 4350 – Software Engineering | https://www.uml.edu/catalog/courses/comp/4350/ | course | 2026-03-29 |
| 14 | COMP 4610 – GUI Programming | https://www.uml.edu/catalog/courses/comp/4610/ | course | 2026-03-29 |
| 15 | COMP 4960 – Senior Capstone | https://www.uml.edu/catalog/courses/comp/4960/ | course | 2026-03-29 |
| 16 | MATH 1310 – Calculus I | https://www.uml.edu/catalog/courses/math/1310/ | course | 2026-03-29 |
| 17 | MATH 1320 – Calculus II | https://www.uml.edu/catalog/courses/math/1320/ | course | 2026-03-29 |
| 18 | MATH 2310 – Calculus III | https://www.uml.edu/catalog/courses/math/2310/ | course | 2026-03-29 |
| 19 | MATH 2340 – Differential Equations | https://www.uml.edu/catalog/courses/math/2340/ | course | 2026-03-29 |
| 20 | MATH 3220 – Statistics | https://www.uml.edu/catalog/courses/math/3220/ | course | 2026-03-29 |
| 21 | BS CS – Degree Requirements | https://www.uml.edu/catalog/undergraduate/sciences/computer-science/bs-computer-science/ | program | 2026-03-29 |
| 22 | CS Minor – Requirements | https://www.uml.edu/catalog/undergraduate/sciences/computer-science/cs-minor/ | program | 2026-03-29 |
| 23 | CS Concentrations / Tracks | https://www.uml.edu/catalog/undergraduate/sciences/computer-science/ | program | 2026-03-29 |
| 24 | Academic Grading Policy | https://www.uml.edu/catalog/policies/academic/grading-system/ | policy | 2026-03-29 |
| 25 | Credit Limits & Transfer Policy | https://www.uml.edu/catalog/policies/academic/credit-hour/ | policy | 2026-03-29 |

---

## Evaluation Metrics

After running `python src/evaluation/evaluator.py`, fill in the actuals:

| Metric | Target | Actual |
|--------|--------|--------|
| Citation Coverage Rate | 100% (cat 1–3) | \_\_\_ % |
| Eligibility Correctness | ≥ 9 / 10 | \_\_\_ / 10 |
| Abstention Accuracy | 5 / 5 | \_\_\_ / 5 |
| Chain Correctness | ≥ 4 / 5 | \_\_\_ / 5 |

> **Do not fill these in speculatively — run the evaluator to get real numbers.**

---

## Project Structure

```
catalog-rag/
├── build_index.py          ← Run once
├── main.py                 ← CLI chat
├── app.py                  ← Gradio demo
├── config.py               ← All tuneable parameters
├── requirements.txt
├── .env.example
├── data/
│   ├── urls.py             ← All 25 catalog URLs
│   ├── raw/                ← Scraped HTML (auto-created)
│   ├── processed/          ← Cleaned JSON (auto-created)
│   └── metadata.json       ← URL + date accessed (auto-created)
├── src/
│   ├── models.py           ← Shared Pydantic models
│   ├── ingestion/          ← scraper.py, cleaner.py
│   ├── chunking/           ← chunker.py
│   ├── embeddings/         ← embedder.py
│   ├── vector_store/       ← faiss_store.py
│   ├── retrieval/          ← retriever.py
│   ├── prompts/            ← system_prompt.py, planner_prompt.py, verifier_prompt.py
│   ├── chains/             ← intake, retrieval, planner, verifier, pipeline
│   └── evaluation/         ← test_queries.py (25 queries), evaluator.py
├── vector_store/           ← FAISS index files (auto-created)
└── transcripts/            ← 3 example conversations (auto-created by evaluator)
```

---

## Screen Recording Checklist

When recording the 1–2 minute Gradio demo, cover these 3 scenes:

1. **(30s)** Eligibility check — type `"Can I take COMP3040? I've done COMP2010 and COMP2030."` — show decision badge + citations
2. **(45s)** Course plan — fill sidebar profile → ask `"Plan my Fall 2026 semester"` — show multi-course plan with justifications
3. **(20s)** Abstention — ask `"Is COMP4110 offered in Spring 2026?"` — show abstention phrase + advisor suggestion

Record with **OBS Studio** (free) or **Windows Xbox Game Bar** (`Win + G`).
Save to `demo/` or upload to YouTube (unlisted).
