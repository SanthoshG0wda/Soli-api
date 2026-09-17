"""
KB Store Connector
==================
Queries the local PostgreSQL knowledge base to:
- Check if a document already exists (duplicate detection)
- List recently ingested documents
"""
from __future__ import annotations

import hashlib
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.connectors.base import BaseConnector
from app.models import Document

logger = logging.getLogger(__name__)


class KBStoreConnector(BaseConnector):
    source_name = "Local Knowledge Base"
    source_domain = "localhost"
    doc_type = "internal"
    authority_score = 1.0

    def __init__(self, db: Session) -> None:
        self.db = db

    def check_duplicate_by_url(self, url: str) -> dict:
        """Check if a document with this source_url already exists in the KB."""
        existing = self.db.execute(
            select(Document).where(Document.source_url == url)
        ).scalar_one_or_none()

        if existing:
            return {
                "exists": True,
                "document_id": str(existing.id),
                "title": existing.title,
                "status": existing.status.value,
            }
        return {"exists": False}

    def check_duplicate_by_hash(self, content: str) -> dict:
        """Check if content (by SHA-256 hash) already exists in the KB."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        existing = self.db.execute(
            select(Document).where(Document.file_hash == content_hash)
        ).scalar_one_or_none()

        if existing:
            return {
                "exists": True,
                "document_id": str(existing.id),
                "title": existing.title,
                "status": existing.status.value,
            }
        return {"exists": False, "hash": content_hash}

    def list_recent(self, limit: int = 5) -> list[dict]:
        """List the most recently ingested documents."""
        rows = self.db.execute(
            select(Document).order_by(Document.uploaded_at.desc()).limit(limit)
        ).scalars().all()
        return [
            {
                "id": str(d.id),
                "title": d.title,
                "status": d.status.value,
                "source_url": d.source_url,
                "uploaded_at": d.uploaded_at.isoformat(),
            }
            for d in rows
        ]
