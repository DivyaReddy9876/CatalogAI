"""
Embedding model loader — wraps HuggingFaceEmbeddings with a module-level
singleton so the 90 MB model is downloaded and loaded only once per process.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

logger = logging.getLogger(__name__)

_embeddings_instance: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFaceEmbeddings instance.

    The model (``all-MiniLM-L6-v2``) is downloaded from HuggingFace Hub on
    the first call and cached locally.  No API key required.

    Specs
    -----
    - Embedding dimension : 384
    - Max sequence length : 256 tokens
    - Similarity metric   : Cosine (FAISS uses L2; we normalise vectors so
                            L2 distance ≡ cosine distance on unit sphere)
    """
    global _embeddings_instance

    if _embeddings_instance is None:
        logger.info("Loading embedding model: %s …", config.EMBEDDING_MODEL)
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": config.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},  # unit-length → cosine ≡ L2
        )
        logger.info("Embedding model loaded.")

    return _embeddings_instance
