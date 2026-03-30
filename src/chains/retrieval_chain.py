"""
Retrieval chain — fetches the most relevant catalog chunks for a query.

Uses MultiQueryRetriever (3 LLM-generated sub-queries) over an MMR base
retriever.  Results are deduplicated and returned as a list of Documents
alongside a formatted context string ready for the Planner.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models import StudentProfile
from src.retrieval.retriever import build_multi_query_retriever

logger = logging.getLogger(__name__)

_SEP = "─" * 60


def _format_docs(docs: List[Document]) -> str:
    """
    Convert retrieved documents into a numbered context string.

    Each chunk includes its citation metadata so the Planner can copy
    citation fields directly without re-looking them up.
    """
    parts: List[str] = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        header = (
            f"[Excerpt {i}]\n"
            f"Title   : {m.get('title', 'Unknown')}\n"
            f"URL     : {m.get('url', 'N/A')}\n"
            f"Section : {m.get('section_heading', 'General')}\n"
            f"Doc type: {m.get('doc_type', 'course')}\n"
            f"{_SEP}"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def _deduplicate(docs: List[Document]) -> List[Document]:
    """Remove duplicate chunks (same chunk_id)."""
    seen: set[str] = set()
    unique: List[Document] = []
    for doc in docs:
        cid = doc.metadata.get("chunk_id", doc.page_content[:40])
        if cid not in seen:
            seen.add(cid)
            unique.append(doc)
    return unique


def create_retrieval_chain(vectorstore, llm: BaseLanguageModel):
    """
    Return a callable that accepts a query + StudentProfile and returns
    (docs, context_string).

    Parameters
    ----------
    vectorstore : Loaded FAISS vectorstore.
    llm         : LLM used by MultiQueryRetriever to generate sub-queries.
    """
    multi_retriever = build_multi_query_retriever(vectorstore, llm)

    def retrieve(query: str, profile: StudentProfile) -> Tuple[List[Document], str]:
        """
        Enrich the query with key profile terms, then retrieve.

        Enrichment ensures sub-queries reference the student's specific
        courses and program, improving multi-hop chain recall.
        """
        enriched_query = (
            f"{query}\n"
            f"Program: {profile.target_program or 'BS Computer Science'}\n"
            f"Completed courses: {', '.join(profile.completed_courses) or 'none'}"
        )
        try:
            docs = multi_retriever.invoke(enriched_query)
        except Exception as exc:
            logger.warning("MultiQueryRetriever failed (%s), falling back to base retriever.", exc)
            docs = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7},
            ).invoke(enriched_query)

        unique_docs = _deduplicate(docs)

        # Cap at 7 chunks: 7 × ~350 + ~700 prompt + 1024 response ≈ 4174 tokens,
        # well under 6000 TPM with the 65s inter-query sleep. 7 is needed to
        # cover 3-hop prereq chains (e.g. COP 2513 → COP 3515 → COP 4538)
        # where each hop lives in a separate chunk.
        unique_docs = unique_docs[:7]

        logger.info("Retrieved %d unique chunks for query.", len(unique_docs))
        context = _format_docs(unique_docs)
        return unique_docs, context

    return retrieve
