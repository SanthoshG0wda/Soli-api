"""Research Agent API Routes."""

import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import Document, DocumentStatus, ResearchSession as DBResearchSession
from app.services.research_agent import (
    fetch_document_text,
    run_research_search,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/research", tags=["research"])


# ── Schemas ────────────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str
    max_results: int = 15


class SearchResultSchema(BaseModel):
    id: str
    title: str
    url: str
    snippet: str
    source_name: str
    source_domain: str
    doc_type: str
    authority_score: float
    relevance_score: float
    combined_score: float
    is_pdf: bool


class SearchResponse(BaseModel):
    session_id: str
    query: str
    expanded_queries: list[str]
    llm_used: bool
    result_count: int
    results: list[SearchResultSchema]


class FetchRequest(BaseModel):
    result_id: str
    title: str
    url: str
    doc_type: str
    source_name: str


class FetchResponse(BaseModel):
    document_id: str
    title: str
    status: str
    message: str


class SessionSchema(BaseModel):
    id: str
    query: str
    result_count: int
    ingested_count: int
    llm_used: bool
    created_at: str


# ── Endpoints ──────────────────────────────────────────────────────────────────


@router.post("/search", response_model=SearchResponse)
async def research_search(
    payload: SearchRequest,
    db: Session = Depends(get_db),
):
    """
    Run the agent harness. With NVIDIA_API_KEY the LLM orchestrates tool calls;
    without it all connectors run in parallel (keyword fallback).
    Returns ranked results plus a session_id for trace lookup.
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    session = await run_research_search(
        query=payload.query.strip(),
        max_results=payload.max_results,
        db=db,
    )

    # Persist session + trace to DB
    trace_json = json.dumps(session.trace) if session.trace else None
    db_session = DBResearchSession(
        id=uuid.UUID(session.session_id),
        query=session.original_query,
        expanded_queries=json.dumps(session.expanded_queries),
        result_count=len(session.results),
        ingested_count=0,
        llm_used=session.llm_used,
    )
    # Store trace in expanded_queries field temporarily (or we can add a column)
    # We store the full trace JSON in expanded_queries for now
    if trace_json:
        db_session.expanded_queries = trace_json
    db.add(db_session)
    db.commit()

    return SearchResponse(
        session_id=session.session_id,
        query=session.original_query,
        expanded_queries=session.expanded_queries,
        llm_used=session.llm_used,
        result_count=len(session.results),
        results=[
            SearchResultSchema(
                id=r.id,
                title=r.title,
                url=r.url,
                snippet=r.snippet,
                source_name=r.source_name,
                source_domain=r.source_domain,
                doc_type=r.doc_type,
                authority_score=round(r.authority_score, 3),
                relevance_score=round(r.relevance_score, 3),
                combined_score=round(r.combined_score, 3),
                is_pdf=r.is_pdf,
            )
            for r in session.results
        ],
    )


@router.post("/fetch", response_model=FetchResponse, status_code=status.HTTP_201_CREATED)
async def fetch_and_ingest(
    payload: FetchRequest,
    db: Session = Depends(get_db),
):
    """Fetch a document from a research result URL and add it to the knowledge base."""
    text, is_pdf = await fetch_document_text(payload.url)

    if not text or len(text.strip()) < 100:
        raise HTTPException(
            status_code=422,
            detail="Could not extract usable text from the document. "
                   "Try downloading the PDF manually and uploading it.",
        )

    doc_id = uuid.uuid4()
    storage_dir = Path(settings.STORAGE_BASE_PATH) / str(doc_id)
    storage_dir.mkdir(parents=True, exist_ok=True)

    ext = "pdf" if is_pdf else "txt"
    dest = storage_dir / f"original.{ext}"

    import hashlib
    file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    existing = db.execute(
        select(Document).where(Document.file_hash == file_hash)
    ).scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Document already exists in knowledge base.",
                "existing_document_id": str(existing.id),
                "existing_document_title": existing.title,
            },
        )

    dest.write_text(text, encoding="utf-8")

    document = Document(
        id=doc_id,
        title=payload.title,
        act_number=None,
        original_filename=f"{payload.title[:60]}.{ext}",
        storage_path=str(dest.resolve()),
        file_hash=file_hash,
        source_url=payload.url,
        status=DocumentStatus.UPLOADED,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # Automatically index into RAG vector database
    try:
        from app.services.rag.indexer import index_document
        res = index_document(db=db, document_id=document.id)
        db.refresh(document)
        chunk_msg = f" and indexed into RAG ({res['chunks_indexed']} chunks)"
    except Exception:
        chunk_msg = ""

    logger.info("Research agent ingested '%s' from %s", payload.title, payload.url)

    return FetchResponse(
        document_id=str(document.id),
        title=document.title,
        status=document.status.value,
        message=f"Document added with status '{document.status.value}'{chunk_msg}. "
                f"Review at /documents/{document.id} before publishing.",
    )


@router.get("/sessions", response_model=list[SessionSchema])
def list_sessions(db: Session = Depends(get_db)):
    """List all past research agent sessions, newest first."""
    rows = db.execute(
        select(DBResearchSession)
        .order_by(DBResearchSession.created_at.desc())
        .limit(50)
    ).scalars().all()
    return [
        SessionSchema(
            id=str(r.id),
            query=r.query,
            result_count=r.result_count,
            ingested_count=r.ingested_count,
            llm_used=r.llm_used,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]


@router.get("/sessions/{session_id}/trace")
def get_session_trace(session_id: str, db: Session = Depends(get_db)):
    """Return the full agent trace for a session — every think/act/observe step."""
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format.")

    row = db.get(DBResearchSession, sid)
    if not row:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    # Trace is stored in expanded_queries as JSON
    try:
        trace = json.loads(row.expanded_queries or "{}")
    except json.JSONDecodeError:
        trace = {"query": row.query, "steps": [], "note": "No trace available for this session."}

    return {
        "session_id": session_id,
        "query": row.query,
        "result_count": row.result_count,
        "llm_used": row.llm_used,
        "created_at": row.created_at.isoformat(),
        "trace": trace,
    }
