# CATALOG_AI - USF Course Planning Assistant — Agentic RAG

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

| Component        | Choice                              | 
|------------------|-------------------------------------|
| Agent framework  | LangChain ≥ 0.2                     | 
| Vector store     | FAISS (local)                       | 
| Embeddings       | `all-MiniLM-L6-v2` (HuggingFace)   | 
| LLM              | Groq `llama-3.1-8b-instant`         |
| Demo UI          | Gradio                              | 

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
- **If PDFs exist** under `data/raw/` (see layout below) → extract text → `data/processed/`
- **Else** scrape catalog HTML → `data/raw/` → clean → `data/processed/`
- Chunk into 500-token segments → in memory
- Embed with MiniLM → `vector_store/catalog_index/`

> **If some URLs return 404:** UML periodically updates their catalog URLs.
> Open `data/urls.py`, find the failing `doc_id`, visit `https://www.uml.edu/catalog/`
> to get the current URL, update the entry, and re-run `python build_index.py`.

### Catalog PDF layout (USF / local PDFs)

Put exports here (then run `python build_index.py`):

```
data/raw/
├── Course Pages/
├── Program Pages/
├── Academic Policy Page/
└── Additional Pages/
```

> **Rebuild from scratch:** delete `data/processed/*.json` (and optionally `vector_store/`) before re-running if you moved or replaced PDFs.
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

Dataset Link : https://docs.google.com/document/d/1jhMVEtiyf2gjpzOZVIkl-BCqafx6dJU2EuO5V_YE_6E/edit?usp=sharing

---

## Evaluation Metrics

After running `python src/evaluation/evaluator.py`, fill in the actuals:

| Metric | Target | 
|--------|--------|--------|
| Citation Coverage Rate | 100% (cat 1–3) | 
| Eligibility Correctness | ≥ 9 / 10 | 
| Abstention Accuracy | 5 / 5 |
| Chain Correctness | ≥ 4 / 5 |

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
## Transcript Sampling 

Documentation Link : https://docs.google.com/document/d/1RwwT8WBrgtoVS8ozQTdAxmDz4ptilKMneFuMLDZrdrE/edit?usp=sharing

---

## Screen Recording Checklist

When recording the 1–2 minute Gradio demo, cover these 3 scenes:

1. **(30s)** Eligibility check — type `"Can I take COMP3040? I've done COMP2010 and COMP2030."` — show decision badge + citations
2. **(45s)** Course plan — fill sidebar profile → ask `"Plan my Fall 2026 semester"` — show multi-course plan with justifications
3. **(20s)** Abstention — ask `"Is COMP4110 offered in Spring 2026?"` — show abstention phrase + advisor suggestion

Demo Link : https://drive.google.com/file/d/1py4YBqEZhNZYTiOlTXKHd8LHdGaeecxf/view?usp=sharing
Save to `demo/` or upload to YouTube (unlisted).
