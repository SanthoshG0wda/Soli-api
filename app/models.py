from datetime import datetime, timezone
import enum
import uuid
from sqlalchemy import ARRAY, Boolean, DateTime, Enum as SQLEnum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base


class DocumentStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    PARSING = "parsing"
    PARSED = "parsed"
    VALIDATING = "validating"
    VALIDATED = "validated"
    IN_REVIEW = "in_review"
    PUBLISHED = "published"
    REJECTED = "rejected"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String, nullable=False)
    act_number: Mapped[str | None] = mapped_column(String, nullable=True)
    original_filename: Mapped[str] = mapped_column(String, nullable=False)
    storage_path: Mapped[str] = mapped_column(String, nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    source_url: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[DocumentStatus] = mapped_column(
        SQLEnum(
            DocumentStatus,
            name="document_status",
            native_enum=True,
            values_callable=lambda obj: [e.value for e in obj],
        ),
        nullable=False,
        default=DocumentStatus.UPLOADED,
    )
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class ResearchSession(Base):
    """Tracks each research agent query and its outcomes."""

    __tablename__ = "research_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    expanded_queries: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ingested_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class RAGChunk(Base):
    """Stores text chunks and vector embeddings for the RAG pipeline."""

    __tablename__ = "rag_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    section_title: Mapped[str | None] = mapped_column(String, nullable=True)
    chunk_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    embedding: Mapped[list[float] | None] = mapped_column(ARRAY(Float), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
