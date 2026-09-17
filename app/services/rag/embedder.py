"""
Dense Vector Embedder
=====================
Uses FastEmbed (BAAI/bge-small-en-v1.5) with ONNX Runtime to compute
normalized 384-dimensional dense vector embeddings for text chunks and queries.
Runs 100% locally with low CPU latency and zero API cost/rate limits.
"""
from __future__ import annotations

import logging
from typing import Sequence
import numpy as np
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

# Singleton embedder instance
_embedder: TextEmbedding | None = None
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_DIMENSION = 384


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        logger.info("Loading FastEmbed model '%s'...", MODEL_NAME)
        _embedder = TextEmbedding(model_name=MODEL_NAME)
    return _embedder


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """
    Generate dense embeddings for a list of text strings.

    Returns
    -------
    list of list of float, each of length 384.
    """
    if not texts:
        return []
    model = get_embedder()
    generator = model.embed(list(texts))
    return [arr.tolist() if isinstance(arr, np.ndarray) else list(arr) for arr in generator]


def embed_query(query: str) -> list[float]:
    """Generate dense embedding for a single search/retrieval query."""
    results = embed_texts([query])
    return results[0] if results else [0.0] * VECTOR_DIMENSION
