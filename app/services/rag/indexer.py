"""
RAG Document Indexer
====================
Extracts text from uploaded legal documents (PDF/HTML/Text), segments into
semantic legal chunks, generates dense embeddings with FastEmbed, and commits
the chunks into the PostgreSQL `rag_chunks` table.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
import uuid
from typing import Any

from sqlalchemy.orm import Session
from sqlalchemy import delete

from app.models import Document, DocumentStatus, RAGChunk
from app.services.rag.chunker import chunk_legal_text
from app.services.rag.embedder import embed_texts

logger = logging.getLogger(__name__)


def extract_document_text(file_path: Path) -> str:
    """
    Extract plain text from a stored file (PDF, HTML, or TXT).
    Inspects magic bytes to detect true binary PDFs and falls back gracefully
    to text/HTML decoding if a text file was saved with a .pdf extension.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Stored document not found at: {file_path}")

    raw_bytes = file_path.read_bytes()
    if not raw_bytes:
        return ""

    # Check if the file is genuinely a binary PDF (starts with %PDF-)
    if raw_bytes.startswith(b"%PDF-"):
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(raw_bytes), strict=False)
            pages_text = []
            for i, page in enumerate(reader.pages):
                try:
                    t = page.extract_text() or ""
                    if t.strip():
                        pages_text.append(f"--- Page {i+1} ---\n{t}")
                except Exception:
                    continue
            if pages_text:
                return "\n\n".join(pages_text)
        except Exception as exc:
            logger.warning("pypdf parsing failed on %s (%s) — falling back to text decoding", file_path, exc)

    # If not a binary PDF or pypdf failed, decode as text / HTML
    decoded = raw_bytes.decode("utf-8", errors="replace")

    # If it contains HTML markup, strip tags
    if "<html" in decoded.lower() or "<body" in decoded.lower() or "<div" in decoded.lower():
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(decoded, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find("body")
        return main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

    return decoded


def index_document(db: Session, document_id: uuid.UUID) -> dict[str, Any]:
    """
    Index a single Document by ID: extract, chunk, embed, and store in rag_chunks.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise ValueError(f"Document with ID {document_id} not found")

    file_path = Path(doc.storage_path)
    logger.info("Indexing document '%s' from %s", doc.title, file_path)

    # 1. Extract raw text
    text = extract_document_text(file_path)
    if not text.strip():
        raise ValueError(f"Extracted empty text from document: {doc.original_filename}")

    # 2. Chunk text semantically
    chunks = chunk_legal_text(text, act_title=doc.title)
    logger.info("Segmented document into %d chunks", len(chunks))

    # 3. Generate dense vector embeddings (batch)
    chunk_texts = [c.content for c in chunks]
    embeddings = embed_texts(chunk_texts)

    # 4. Remove any existing chunks for this document
    db.execute(delete(RAGChunk).where(RAGChunk.document_id == document_id))

    # 5. Insert new chunks with embeddings
    for i, chunk in enumerate(chunks):
        rag_chunk = RAGChunk(
            document_id=doc.id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            section_title=chunk.section_title,
            chunk_metadata=chunk.metadata,
            embedding=embeddings[i] if i < len(embeddings) else None,
        )
        db.add(rag_chunk)

    # 6. Update document status
    doc.status = DocumentStatus.PARSED
    db.commit()

    logger.info(
        "Document '%s' successfully indexed with %d chunks",
        doc.title,
        len(chunks),
    )

    return {
        "document_id": str(doc.id),
        "title": doc.title,
        "chunks_indexed": len(chunks),
        "status": doc.status.value,
    }
