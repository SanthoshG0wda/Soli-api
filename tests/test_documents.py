import io
import shutil
import uuid
import pytest
from fastapi.testclient import TestClient
from app.config import settings
from app.db import Base, engine
from sqlalchemy.orm import Session
from sqlalchemy import delete
from app.models import Document
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def setup_database_and_storage():
    Base.metadata.create_all(bind=engine)
    with Session(bind=engine) as session:
        # Only clean up test-created files so real corpus documents are never wiped
        session.execute(delete(Document).where(Document.title.in_([
            "The Indian Contract Act, 1872",
            "Contract Act Duplicate Test",
            "Test Legal Statute 2024",
        ])))
        session.commit()
    # Use temporary test storage directory
    original_storage = settings.STORAGE_BASE_PATH
    test_storage = "./test_storage"
    settings.STORAGE_BASE_PATH = test_storage
    yield
    # Teardown test storage
    settings.STORAGE_BASE_PATH = original_storage
    shutil.rmtree(test_storage, ignore_errors=True)


@pytest.fixture
def client():

    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_document_success(client: TestClient):
    # Minimal valid PDF header and content
    pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"
    file_payload = ("contract_act.pdf", io.BytesIO(pdf_content), "application/pdf")

    response = client.post(
        "/documents/upload",
        data={
            "title": "The Indian Contract Act, 1872",
            "act_number": "Act No. 9 of 1872",
            "source_url": "https://indiacode.nic.in/handle/123456789/2187",
        },
        files={"file": file_payload},
    )

    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["title"] == "The Indian Contract Act, 1872"
    assert data["act_number"] == "Act No. 9 of 1872"
    assert data["original_filename"] == "contract_act.pdf"
    assert data["status"] in ("uploaded", "parsed")
    assert data["source_url"] == "https://indiacode.nic.in/handle/123456789/2187"
    assert "uploaded_at" in data

    # Test GET by ID
    doc_id = data["id"]
    get_res = client.get(f"/documents/{doc_id}")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["id"] == doc_id
    assert get_data["title"] == data["title"]
    assert get_data["status"] in ("uploaded", "parsed")


def test_upload_duplicate_hash_conflict_409(client: TestClient):
    # Same PDF content should trigger duplicate hash conflict
    pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"
    file_payload = ("duplicate_contract.pdf", io.BytesIO(pdf_content), "application/pdf")

    response = client.post(
        "/documents/upload",
        data={
            "title": "Duplicate Contract Act",
            "act_number": "Act 9",
        },
        files={"file": file_payload},
    )

    assert response.status_code == 409
    error_payload = response.json()
    assert "detail" in error_payload
    detail = error_payload["detail"]
    assert "message" in detail
    assert "existing_document_id" in detail
    assert "existing_document_title" in detail
    assert detail["existing_document_title"] == "The Indian Contract Act, 1872"


def test_upload_invalid_extension_rejected(client: TestClient):
    file_payload = ("test.txt", io.BytesIO(b"Hello world"), "application/pdf")
    response = client.post(
        "/documents/upload",
        data={"title": "Test Document"},
        files={"file": file_payload},
    )
    assert response.status_code == 400
    assert "Invalid file extension" in response.json()["detail"]


def test_upload_invalid_content_type_rejected(client: TestClient):
    file_payload = ("test.pdf", io.BytesIO(b"Not a PDF"), "text/plain")
    response = client.post(
        "/documents/upload",
        data={"title": "Test Document"},
        files={"file": file_payload},
    )
    assert response.status_code == 400
    assert "Invalid content type" in response.json()["detail"]


def test_upload_empty_file_rejected(client: TestClient):
    file_payload = ("empty.pdf", io.BytesIO(b""), "application/pdf")
    response = client.post(
        "/documents/upload",
        data={"title": "Empty Document"},
        files={"file": file_payload},
    )
    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]


def test_get_document_not_found(client: TestClient):
    fake_id = str(uuid.uuid4())
    response = client.get(f"/documents/{fake_id}")
    assert response.status_code == 404
    assert response.json()["detail"] == f"Document with id {fake_id} not found"


def test_list_documents(client: TestClient):
    response = client.get("/documents")
    assert response.status_code == 200
    docs = response.json()
    assert isinstance(docs, list)
    assert len(docs) >= 1
    first = docs[0]
    for field in ["id", "title", "act_number", "source_url", "original_filename", "status", "uploaded_at"]:
        assert field in first
