import hashlib
import logging
from pathlib import Path
import shutil
import uuid
import re
import httpx
from bs4 import BeautifulSoup
from fastapi import HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.config import settings
from app.models import Document, DocumentStatus

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}


def sanitize_legal_filename(title: str, default: str = "document.pdf") -> str:
    """
    Standardizes legal document names into clean, readable filenames:
    e.g. 'Bharatiya Nyaya Sanhita, 2023' -> 'Bharatiya_Nyaya_Sanhita_2023.pdf'
    """
    cleaned = title.strip()
    cleaned = re.sub(r"\.pdf$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    cleaned = re.sub(r"[\s_]+", "_", cleaned).strip("_")
    if not cleaned:
        return default
    return f"{cleaned[:100]}.pdf"


def validate_pdf(file: UploadFile) -> None:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file extension: file must be a .pdf",
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type '{file.content_type}': only application/pdf is allowed",
        )


def create_document_from_upload(
    db: Session,
    file: UploadFile,
    title: str,
    act_number: str | None = None,
    source_url: str | None = None,
) -> Document:
    validate_pdf(file)

    document_id = uuid.uuid4()
    storage_dir = Path(settings.STORAGE_BASE_PATH) / str(document_id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    destination_file = storage_dir / "original.pdf"

    hasher = hashlib.sha256()
    bytes_written = 0

    try:
        with open(destination_file, "wb") as f:
            while chunk := file.file.read(64 * 1024):
                hasher.update(chunk)
                f.write(chunk)
                bytes_written += len(chunk)

        if bytes_written == 0:
            if destination_file.exists():
                destination_file.unlink(missing_ok=True)
            if storage_dir.exists():
                shutil.rmtree(storage_dir, ignore_errors=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)",
            )

        file_hash = hasher.hexdigest()

        # Check for duplicate document by SHA-256 hash
        existing = db.execute(
            select(Document).where(Document.file_hash == file_hash)
        ).scalar_one_or_none()

        if existing:
            # Clean up the newly streamed file on disk
            if destination_file.exists():
                destination_file.unlink(missing_ok=True)
            if storage_dir.exists():
                shutil.rmtree(storage_dir, ignore_errors=True)

            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": f"Document already exists with identical file hash: {file_hash}",
                    "existing_document_id": str(existing.id),
                    "existing_document_title": existing.title,
                },
            )

        standardized_filename = file.filename or sanitize_legal_filename(title)

        document = Document(
            id=document_id,
            title=title.strip(),
            act_number=act_number.strip() if act_number else None,
            original_filename=standardized_filename,
            storage_path=str(destination_file.resolve()),
            file_hash=file_hash,
            source_url=source_url.strip() if source_url else None,
            status=DocumentStatus.UPLOADED,
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        # Automatically index into RAG vector database
        try:
            from app.services.rag.indexer import index_document
            index_document(db=db, document_id=document.id)
            db.refresh(document)
        except Exception as exc:
            logger.warning("RAG auto-indexing failed for '%s' (%s): %s", document.title, document.id, exc)

        return document

    except HTTPException:
        raise
    except Exception as e:
        if destination_file.exists():
            destination_file.unlink(missing_ok=True)
        if storage_dir.exists():
            shutil.rmtree(storage_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file upload: {str(e)}",
        )


def create_document_from_url(
    db: Session,
    url: str,
    title: str,
    act_number: str | None = None,
    source_url: str | None = None,
) -> Document:
    clean_filename = sanitize_legal_filename(title)
    document_id = uuid.uuid4()
    storage_dir = Path(settings.STORAGE_BASE_PATH) / str(document_id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    destination_file = storage_dir / "original.pdf"

    hasher = hashlib.sha256()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        with httpx.Client(follow_redirects=True, timeout=45.0, headers=headers) as client:
            resp = client.get(url)
            if resp.status_code >= 400:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Source server returned HTTP {resp.status_code} while accessing {url}",
                )

            content = resp.content
            if not content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Fetched document content was empty",
                )

            content_type = resp.headers.get("content-type", "").lower()
            is_pdf = content.startswith(b"%PDF-") or "application/pdf" in content_type

            if not is_pdf and ("text/html" in content_type or b"<html" in content.lower()):
                soup = BeautifulSoup(content, "html.parser")
                for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    tag.decompose()

                main_elem = (
                    soup.find("div", class_="judgments")
                    or soup.find("div", class_="doc")
                    or soup.find("main")
                    or soup.find("article")
                    or soup.find("body")
                    or soup
                )
                extracted_text = main_elem.get_text(separator="\n\n", strip=True)

                content_to_save = (
                    f"Title: {title}\n"
                    f"Act / Citation: {act_number or 'N/A'}\n"
                    f"Official Source URL: {url}\n\n"
                    f"{extracted_text}"
                ).encode("utf-8")
            else:
                content_to_save = content

            hasher.update(content_to_save)
            destination_file.write_bytes(content_to_save)

        file_hash = hasher.hexdigest()

        # Check duplicate
        existing = db.execute(
            select(Document).where(Document.file_hash == file_hash)
        ).scalar_one_or_none()

        if existing:
            if destination_file.exists():
                destination_file.unlink(missing_ok=True)
            if storage_dir.exists():
                shutil.rmtree(storage_dir, ignore_errors=True)

            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": f"Document already exists with identical file hash: {file_hash}",
                    "existing_document_id": str(existing.id),
                    "existing_document_title": existing.title,
                },
            )

        document = Document(
            id=document_id,
            title=title.strip(),
            act_number=act_number.strip() if act_number else None,
            original_filename=clean_filename,
            storage_path=str(destination_file.resolve()),
            file_hash=file_hash,
            source_url=(source_url or url).strip(),
            status=DocumentStatus.UPLOADED,
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        # Index document into RAG vector database
        try:
            from app.services.rag.indexer import index_document
            index_document(db=db, document_id=document.id)
            db.refresh(document)
        except Exception as exc:
            logger.warning("RAG auto-indexing failed for URL document '%s' (%s): %s", document.title, document.id, exc)

        return document

    except HTTPException:
        raise
    except Exception as e:
        if destination_file.exists():
            destination_file.unlink(missing_ok=True)
        if storage_dir.exists():
            shutil.rmtree(storage_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch and process document: {str(e)}",
        )



def get_document_by_id(db: Session, document_id: uuid.UUID) -> Document:
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id {document_id} not found",
        )
    return document


def list_documents(db: Session) -> list[Document]:
    stmt = select(Document).order_by(Document.uploaded_at.desc())
    return list(db.execute(stmt).scalars().all())


def delete_document(db: Session, document_id: uuid.UUID) -> None:
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id {document_id} not found",
        )

    # Remove file storage directory (contains original.pdf and any derived artifacts)
    # storage_path is like /.../storage/<uuid>/original.pdf — remove parent dir
    try:
        doc_path = Path(document.storage_path) if document.storage_path else None
        storage_dir = doc_path.parent if doc_path and doc_path.parent.exists() else None
        if storage_dir and storage_dir.exists():
            # Safety: ensure we only delete inside STORAGE_BASE_PATH
            try:
                base = Path(settings.STORAGE_BASE_PATH).resolve()
                target = storage_dir.resolve()
                if str(target).startswith(str(base)):
                    shutil.rmtree(target, ignore_errors=True)
                else:
                    # Fallback: just delete the single file
                    if doc_path and doc_path.exists():
                        doc_path.unlink(missing_ok=True)
            except Exception:
                if doc_path and doc_path.exists():
                    doc_path.unlink(missing_ok=True)
                if storage_dir and storage_dir.exists():
                    shutil.rmtree(storage_dir, ignore_errors=True)
    except Exception:
        pass  # Storage cleanup failure should not block DB deletion

    # Delete DB row — RAG chunks cascade via FK ondelete="CASCADE"
    db.delete(document)
    db.commit()
