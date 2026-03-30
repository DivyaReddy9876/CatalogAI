"""
Retriever factory — wraps the FAISS vectorstore in a two-layer retriever:

Layer 1 — MMR (Maximal Marginal Relevance)
  Fetches fetch_k=20 candidates from FAISS then re-ranks to k=6 by balancing
  relevance AND diversity.  Prevents returning 6 near-identical copies of the
  same prerequisite sentence.

Layer 2 — CustomMultiQueryRetriever
  LangChain 1.x removed the built-in MultiQueryRetriever, so this module
  provides a lightweight drop-in replacement with the same interface.
  The LLM generates 3 semantically distinct sub-queries; each hits the MMR
  retriever independently; results are deduplicated by page_content hash and
  merged.  This dramatically improves recall for multi-hop chains.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

logger = logging.getLogger(__name__)

# Prompt that asks the LLM to produce N alternative phrasings
_MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question", "n"],
    template=(
        "You are a university catalog assistant. "
        "Generate {n} distinct alternative phrasings of the following student question "
        "to improve document retrieval coverage. "
        "Output ONLY the {n} questions, one per line, with no numbering or extra text.\n\n"
        "Original question: {question}"
    ),
)


class CustomMultiQueryRetriever(BaseRetriever):
    """
    Drop-in replacement for the removed langchain MultiQueryRetriever.

    Generates ``num_queries`` LLM sub-queries, retrieves from the base
    MMR retriever for each, and deduplicates by page_content hash.
    """

    base_retriever: VectorStoreRetriever = Field(...)
    llm: BaseLanguageModel = Field(...)
    num_queries: int = Field(default=3)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Generate alternative sub-queries
        try:
            prompt_text = _MULTI_QUERY_PROMPT.format(question=query, n=self.num_queries)
            response = self.llm.invoke(prompt_text)
            sub_queries_raw = getattr(response, "content", str(response)).strip()
            sub_queries = [q.strip() for q in sub_queries_raw.splitlines() if q.strip()]
        except Exception as exc:
            logger.warning("Sub-query generation failed (%s) — using original query only.", exc)
            sub_queries = []

        all_queries = [query] + sub_queries[:self.num_queries]
        logger.debug("MultiQuery sub-queries: %s", all_queries)

        # Retrieve for each sub-query and deduplicate
        seen: set = set()
        docs: List[Document] = []
        for q in all_queries:
            try:
                results = self.base_retriever.invoke(q)
                for doc in results:
                    key = hash(doc.page_content)
                    if key not in seen:
                        seen.add(key)
                        docs.append(doc)
            except Exception as exc:
                logger.warning("Retrieval failed for sub-query '%s…': %s", q[:40], exc)

        logger.debug("MultiQuery retrieved %d unique chunks.", len(docs))
        return docs


def build_base_retriever(vectorstore: FAISS) -> VectorStoreRetriever:
    """
    MMR retriever (k=6, fetch_k=20, lambda_mult=0.7).

    lambda_mult of 0.7 slightly favours relevance over diversity — the right
    balance for catalog chunks that are all thematically related.
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":           config.RETRIEVER_K,
            "fetch_k":     config.RETRIEVER_FETCH_K,
            "lambda_mult": config.RETRIEVER_LAMBDA_MULT,
        },
    )


def build_multi_query_retriever(
    vectorstore: FAISS,
    llm: BaseLanguageModel,
) -> CustomMultiQueryRetriever:
    """
    CustomMultiQueryRetriever wrapping the MMR base retriever.

    The LLM generates 3 alternative phrasings of the student's question.
    Each independently queries the MMR retriever, and the union of results
    is returned deduplicated by content hash.
    """
    base = build_base_retriever(vectorstore)
    retriever = CustomMultiQueryRetriever(
        base_retriever=base,
        llm=llm,
        num_queries=3,
    )
    logger.info("CustomMultiQueryRetriever ready (base: MMR k=%d).", config.RETRIEVER_K)
    return retriever
