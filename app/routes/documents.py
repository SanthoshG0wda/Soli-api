import uuid
from uuid import UUID
from pathlib import Path
import httpx
from bs4 import BeautifulSoup
from fastapi import APIRouter, Depends, File, Form, UploadFile, status, HTTPException
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session
from app.db import get_db
from app.schemas import DocumentResponse, DuplicateDocumentErrorResponse, FetchUrlRequest
from app.services import document_service


router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/proxy-download")
def proxy_download_file(
    url: str,
    title: str,
):
    clean_filename = document_service.sanitize_legal_filename(title)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        with httpx.Client(follow_redirects=True, timeout=60.0, headers=headers) as client:
            resp = client.get(url)
            if resp.status_code >= 400:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Source server returned HTTP {resp.status_code} for {url}",
                )

            content = resp.content
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
                text_content = (
                    f"Title: {title}\n"
                    f"Source URL: {url}\n\n"
                    f"{main_elem.get_text(separator=chr(10) + chr(10), strip=True)}"
                )
                txt_filename = clean_filename[:-4] + ".txt" if clean_filename.endswith(".pdf") else clean_filename + ".txt"
                return Response(
                    content=text_content.encode("utf-8"),
                    media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{txt_filename}"'},
                )

            return Response(
                content=content,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{clean_filename}"'},
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stream document from source: {str(e)}",
        )



@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        409: {
            "model": DuplicateDocumentErrorResponse,
            "description": "Duplicate document hash conflict",
        },
        400: {
            "description": "Invalid file format or empty payload",
        },
    },
)
def upload_document(
    file: UploadFile = File(..., description="PDF document binary stream"),
    title: str = Form(..., description="Title of the legislation or legal document"),
    act_number: str | None = Form(None, description="Official Act number/year (e.g. Act No. 9 of 1872)"),
    source_url: str | None = Form(None, description="Original source or gazette URL"),
    db: Session = Depends(get_db),
):
    return document_service.create_document_from_upload(
        db=db,
        file=file,
        title=title,
        act_number=act_number,
        source_url=source_url,
    )


@router.post(
    "/fetch-url",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        409: {
            "model": DuplicateDocumentErrorResponse,
            "description": "Duplicate document hash conflict",
        },
        400: {
            "description": "Failed to fetch document or invalid content",
        },
    },
)
def fetch_document_from_url(
    payload: FetchUrlRequest,
    db: Session = Depends(get_db),
):
    return document_service.create_document_from_url(
        db=db,
        url=payload.url,
        title=payload.title,
        act_number=payload.act_number,
        source_url=payload.source_url or payload.url,
    )


@router.get("/{id}/download")
def download_document(
    id: uuid.UUID,
    db: Session = Depends(get_db),
):
    doc = document_service.get_document_by_id(db=db, document_id=id)
    if not doc.storage_path or not Path(doc.storage_path).exists():
        raise HTTPException(status_code=404, detail="Document file not found on disk")
    return FileResponse(
        path=doc.storage_path,
        filename=doc.original_filename,
        media_type="application/pdf",
    )



@router.get(
    "/{id}",
    response_model=DocumentResponse,
    responses={
        404: {
            "description": "Document not found",
        },
    },
)
def get_document(
    id: uuid.UUID,
    db: Session = Depends(get_db),
):
    return document_service.get_document_by_id(db=db, document_id=id)


@router.delete(
    "/{id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"description": "Document not found"},
    },
)
def delete_document(
    id: uuid.UUID,
    db: Session = Depends(get_db),
):
    document_service.delete_document(db=db, document_id=id)
    return None


@router.get(
    "",
    response_model=list[DocumentResponse],
)
def list_documents(
    db: Session = Depends(get_db),
):
    return document_service.list_documents(db=db)
