import hashlib
import uuid
from pathlib import Path
from app.db import SessionLocal
from app.models import Document, DocumentStatus, RAGChunk
from app.services.rag.indexer import index_document

DOCS = [
    ("fec346df-24d0-473b-8054-439273ff0b5a", "The Information Technology Act, 2000", "Act No. 21 of 2000"),
    ("e17b9ef1-b37a-443e-b9c6-58ee500a9bb1", "Article 14, Constitution of India", "Article 14"),
    ("6696c45c-f651-40cc-b014-5091a9854445", "Article 15, Constitution of India", "Article 15"),
    ("fd7a6b57-754b-49a7-b6f2-40f31772a9e4", "Article 19(1)(a), Constitution of India", "Article 19(1)(a)"),
    ("63a0f2e6-8c84-4244-a677-799571b4614b", "Article 20, Constitution of India", "Article 20"),
    ("5b410fd5-115c-41a9-860c-850c959b3afa", "Article 21, Constitution of India", "Article 21"),
    ("01d73073-26a9-402f-bb3f-a705ec655c90", "Article 22, Constitution of India", "Article 22"),
    ("2c407847-5051-4fe1-9e5c-da5b700f767a", "The Bharatiya Nyaya Sanhita, 2023", "Act No. 45 of 2023"),
    ("480a4218-0e70-4f8c-a3b7-795b98c45bba", "The Bharatiya Nagarik Suraksha Sanhita, 2023", "Act No. 46 of 2023"),
    ("c544a506-752e-46d1-b8b0-2595e618fc31", "The Bharatiya Sakshya Adhiniyam, 2023", "Act No. 47 of 2023"),
]

def seed():
    db = SessionLocal()
    try:
        for folder_id, title, act_no in DOCS:
            pdf_path = Path("storage") / folder_id / "original.pdf"
            if not pdf_path.exists():
                print(f"Skipping {title}: file not found at {pdf_path}")
                continue

            with open(pdf_path, "rb") as f:
                content = f.read()
            file_hash = hashlib.sha256(content).hexdigest()

            # Check if exists
            doc = db.query(Document).filter(Document.file_hash == file_hash).first()
            doc_uuid = uuid.UUID(folder_id)

            if not doc:
                doc = Document(
                    id=doc_uuid,
                    title=title,
                    act_number=act_no,
                    original_filename=f"{title.replace(' ', '_').lower()}.pdf",
                    storage_path=str(pdf_path.resolve()),
                    file_hash=file_hash,
                    source_url=None,
                    status=DocumentStatus.UPLOADED,
                )
                db.add(doc)
                db.commit()
                db.refresh(doc)
                print(f"Registered document: {title} ({doc.id})")
            else:
                print(f"Document already in DB: {title} ({doc.id})")

            # Check if indexed
            chunk_count = db.query(RAGChunk).filter(RAGChunk.document_id == doc.id).count()
            if chunk_count == 0:
                print(f"Indexing {title} into NeonDB pgvector...")
                result = index_document(db, doc.id)
                print(f"Indexed {result['chunks_indexed']} chunks for {title}")
            else:
                print(f"{title} already has {chunk_count} chunks indexed.")

        total_docs = db.query(Document).count()
        total_chunks = db.query(RAGChunk).count()
        print(f"\n[DONE] NeonDB status: {total_docs} documents, {total_chunks} rag_chunks indexed!")
    finally:
        db.close()

if __name__ == "__main__":
    seed()
