"""
RAG API Routes
==============
Exposes endpoints for:
- Document vector indexing (chunk + embed)
- Semantic vector search over legal knowledge base
- Grounded RAG question answering with pinpoint statutory citations
- Chunk inspection and statistics
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func, select

from app.db import get_db
from app.models import Document, RAGChunk
from app.services.rag.indexer import index_document
from app.services.rag.retriever import retrieve_similar_chunks
from app.services.rag.generator import generate_rag_answer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class RAGSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Legal question or topic to search")
    top_k: int = Field(default=5, ge=1, le=20, description="Max number of chunks to retrieve")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum cosine similarity score")
    document_ids: list[uuid.UUID] | None = Field(default=None, description="Optional document filter")


class RAGAskRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Legal question to answer with citations")
    top_k: int = Field(default=5, ge=1, le=15, description="Number of context passages to feed into LLM")
    document_ids: list[uuid.UUID] | None = Field(default=None, description="Optional document filter")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/index/{document_id}", status_code=status.HTTP_200_OK)
def index_document_endpoint(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    """
    Extract text, generate semantic legal chunks, compute 384-d embeddings,
    and persist into the pgvector/PostgreSQL rag_chunks table.
    """
    try:
        result = index_document(db=db, document_id=document_id)
        return result
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err))
    except Exception as err:
        logger.exception("Failed to index document %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {err}",
        )


@router.post("/index-all", status_code=status.HTTP_200_OK)
def index_all_documents(db: Session = Depends(get_db)):
    """Index all documents in the database that have not yet been indexed."""
    docs = db.query(Document).all()
    results = []
    for doc in docs:
        try:
            res = index_document(db=db, document_id=doc.id)
            results.append({"document_id": str(doc.id), "status": "success", "chunks": res["chunks_indexed"]})
        except Exception as exc:
            results.append({"document_id": str(doc.id), "status": "error", "error": str(exc)})

    return {"total_documents": len(docs), "results": results}


@router.post("/search", status_code=status.HTTP_200_OK)
def vector_search(
    req: RAGSearchRequest,
    db: Session = Depends(get_db),
):
    """
    Dense semantic vector search over all indexed legal chunks.
    Returns ranked passages with cosine similarity scores and section titles.
    """
    chunks = retrieve_similar_chunks(
        db=db,
        query=req.query,
        top_k=req.top_k,
        document_ids=req.document_ids,
        min_score=req.min_score,
    )
    return {
        "query": req.query,
        "count": len(chunks),
        "results": [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "document_title": c.document_title,
                "act_number": c.act_number,
                "section_title": c.section_title,
                "chunk_index": c.chunk_index,
                "similarity_score": c.score,
                "content": c.content,
                "metadata": c.metadata,
            }
            for c in chunks
        ],
    }


@router.post("/ask", status_code=status.HTTP_200_OK)
async def rag_ask(
    req: RAGAskRequest,
    db: Session = Depends(get_db),
):
    """
    Full Retrieval-Augmented Generation (RAG):
    1. Retrieves top-k most relevant legal chunks using vector similarity.
    2. Sends context to meta/muse-glimmer-30b.
    3. Synthesizes an authoritative legal answer citing exact sections.
    """
    chunks = retrieve_similar_chunks(
        db=db,
        query=req.query,
        top_k=req.top_k,
        document_ids=req.document_ids,
    )
    result = await generate_rag_answer(query=req.query, chunks=chunks)
    return result


@router.get("/documents/{document_id}/chunks")
def get_document_chunks(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    """Inspect all chunks and section metadata for a specific document."""
    chunks = (
        db.query(RAGChunk)
        .filter(RAGChunk.document_id == document_id)
        .order_by(RAGChunk.chunk_index)
        .all()
    )
    return {
        "document_id": str(document_id),
        "total_chunks": len(chunks),
        "chunks": [
            {
                "id": str(c.id),
                "chunk_index": c.chunk_index,
                "section_title": c.section_title,
                "content": c.content,
                "metadata": c.chunk_metadata,
                "has_embedding": c.embedding is not None,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in chunks
        ],
    }


@router.get("/stats")
def get_rag_stats(db: Session = Depends(get_db)):
    """Summary statistics for the vector database and knowledge base chunks."""
    total_chunks = db.query(func.count(RAGChunk.id)).scalar() or 0
    indexed_docs = db.query(func.count(func.distinct(RAGChunk.document_id))).scalar() or 0
    total_docs = db.query(func.count(Document.id)).scalar() or 0

    return {
        "total_chunks": total_chunks,
        "indexed_documents": indexed_docs,
        "total_documents": total_docs,
        "vector_dimensions": 384,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "llm_model": "meta/muse-glimmer-30b",
    }
