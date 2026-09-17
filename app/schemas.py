from datetime import datetime
import uuid
from pydantic import BaseModel, ConfigDict
from app.models import DocumentStatus


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    act_number: str | None = None
    source_url: str | None = None
    original_filename: str
    status: DocumentStatus
    uploaded_at: datetime


class DuplicateDocumentDetail(BaseModel):
    message: str
    existing_document_id: str
    existing_document_title: str


class DuplicateDocumentErrorResponse(BaseModel):
    detail: DuplicateDocumentDetail


class FetchUrlRequest(BaseModel):
    url: str
    title: str
    act_number: str | None = None
    source_url: str | None = None

