"""
RAG Vector Retriever
====================
Searches the PostgreSQL `rag_chunks` table for chunks semantically most similar
to a given search query using cosine similarity over dense vector embeddings.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models import Document, RAGChunk
from app.services.rag.embedder import embed_query

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_title: str
    act_number: str | None
    chunk_index: int
    section_title: str | None
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def retrieve_similar_chunks(
    db: Session,
    query: str,
    top_k: int = 5,
    document_ids: list[uuid.UUID] | None = None,
    min_score: float = 0.0,
) -> list[RetrievedChunk]:
    """
    Find top-k most semantically relevant chunks for a search query.

    Parameters
    ----------
    db:
        Database session.
    query:
        User search query or question.
    top_k:
        Number of top matches to return.
    document_ids:
        Optional list of document UUIDs to scope the search to.
    min_score:
        Minimum cosine similarity threshold (0.0 - 1.0).
    """
    if not query or not query.strip():
        return []

    # 1. Compute query embedding vector
    q_vec = np.array(embed_query(query), dtype=np.float32)
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec = q_vec / q_norm

    # 2. Build SQL query
    stmt = (
        select(
            RAGChunk.id,
            RAGChunk.document_id,
            RAGChunk.chunk_index,
            RAGChunk.section_title,
            RAGChunk.content,
            RAGChunk.chunk_metadata,
            RAGChunk.embedding,
            Document.title.label("document_title"),
            Document.act_number.label("act_number"),
        )
        .join(Document, RAGChunk.document_id == Document.id)
        .where(RAGChunk.embedding.isnot(None))
    )

    if document_ids:
        stmt = stmt.where(RAGChunk.document_id.in_(document_ids))

    rows = db.execute(stmt).fetchall()
    if not rows:
        return []

    # 3. Compute hybrid scores (dense vector similarity + lexical/section boosting)
    scored_chunks: list[RetrievedChunk] = []
    q_lower = query.lower()
    is_asking_for_toc = any(w in q_lower for w in ["table of contents", "arrangement of sections", "list of chapters", "index of sections"])

    # Extract section numbers mentioned in query (e.g. 'section 73', 'sec. 2', 'section 3(1)')
    import re
    sec_query_match = re.search(r'\b(?:section|sec\.?|article|art\.?)\s*([0-9]+[a-z]*)', q_lower)
    target_sec_num = sec_query_match.group(1).lower() if sec_query_match else None

    # Extract potential legal phrases (words >= 4 chars)
    legal_keywords = [w for w in re.findall(r'[a-z]{4,}', q_lower) if w not in {"what", "which", "where", "when", "under", "this", "that", "with", "from", "have", "been", "their", "about", "could", "would", "should"}]

    for row in rows:
        emb = row.embedding
        if not emb:
            continue
        v = np.array(emb, dtype=np.float32)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            continue
        
        # Dense cosine similarity
        dense_sim = float(np.dot(q_vec, v / v_norm))

        meta = row.chunk_metadata or {}
        is_toc = meta.get("is_toc") or "arrangement of sections" in (row.section_title or "").lower() or "table of contents" in (row.section_title or "").lower()

        # If it's a TOC chunk and user isn't asking for TOC, apply a strong discount so substantive text ranks higher
        if is_toc and not is_asking_for_toc:
            dense_sim = dense_sim * 0.50

        # Hybrid Boost 1: Explicit Section Number match
        sec_num_in_chunk = str(meta.get("section_number") or "").lower()
        sec_title_lower = (row.section_title or "").lower()
        if target_sec_num:
            if sec_num_in_chunk == target_sec_num or f"section {target_sec_num}" in sec_title_lower or f"sec {target_sec_num}" in sec_title_lower:
                dense_sim += 0.15

        # Hybrid Boost 2: Title & Content keyword match
        content_lower = row.content[:1500].lower()
        matches_count = sum(1 for kw in legal_keywords if kw in sec_title_lower or kw in content_lower)
        if legal_keywords:
            dense_sim += min(0.10, (matches_count / len(legal_keywords)) * 0.10)

        sim = min(1.0, max(0.0, dense_sim))

        if sim >= min_score:
            scored_chunks.append(RetrievedChunk(
                chunk_id=str(row.id),
                document_id=str(row.document_id),
                document_title=row.document_title,
                act_number=row.act_number,
                chunk_index=row.chunk_index,
                section_title=row.section_title,
                content=row.content,
                score=round(sim, 4),
                metadata=meta,
            ))

    # 4. Sort descending by similarity score and take top_k
    scored_chunks.sort(key=lambda x: x.score, reverse=True)
    results = scored_chunks[:top_k]

    logger.info(
        "Retrieved %d / %d chunks for query '%s' (top score: %.4f)",
        len(results),
        len(rows),
        query,
        results[0].score if results else 0.0,
    )

    return results
