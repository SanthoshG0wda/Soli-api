# Soli Backend — Legal Knowledge Base & RAG Platform

> **Soli** backend is a FastAPI + pgvector ingestion & retrieval platform for Indian legal research. It ingests PDFs (Acts, Judgments, Books), chunks by statutory structure, embeds with `BAAI/bge-small-en-v1.5` (384d), stores in PostgreSQL+pgvector, and serves grounded RAG with pinpoint citations.

![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)
![pgvector](https://img.shields.io/badge/pgvector-0.5-336791?logo=postgresql)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-D71F00?logo=sqlalchemy)

---

## 1. Overview

`backend/` exposes three domains under one FastAPI app (`app/main.py:8`):

- **Documents** — upload/list/get/delete PDFs with SHA256 dedup + file storage
- **RAG** — `index` (chunk+embed), `search` (vector similarity), `ask` (LLM synthesis), `stats`, corpus checklist
- **Research Agent** — `search`/`fetch`/`sessions` over verified Indian public sources (DDGS + LLM query expansion)

All routes are consumed by `frontend/` via `NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:8000`).

---

## 2. Features

| Domain | Endpoints & Behavior |
|--------|----------------------|
| **Documents** | `POST /documents/upload` (`multipart`: `file` PDF only `application/pdf`, `title`, `act_number`, `source_url`) → streams to `storage/<uuid>/original.pdf`, `SHA256`, `409` on duplicate (`app/services/document_service.py:67`), `201` + auto `index_document` (`services/rag/indexer.py`). `GET /documents` / `GET /documents/{id}` / `DELETE /documents/{id}` (`204`, cascades `rag_chunks` via `FK ondelete="CASCADE"` `models.py:71`, removes dir guarded by `STORAGE_BASE_PATH`). |
| **RAG Index** | `POST /rag/index/{id}` and `POST /rag/index-all` (`routes/rag.py:48`) → `services/rag/indexer.py` parses PDF (`pypdf`), splits by statutory sections/clauses, embeds via `fastembed` (`BAAI/bge-small-en-v1.5` → 384d `ARRAY(Float)` `models.py:77`), persists `RAGChunk`. |
| **RAG Search** | `POST /rag/search` (`routes/rag.py:85`) body `{query, top_k 1-20, min_score 0-1, document_ids[]}` → `services/rag/retriever.py` `retrieve_similar_chunks` (cosine via `pgvector`), returns `chunk_id/document_id/title/act_number/section_title/chunk_index/similarity_score/content/metadata`. |
| **RAG Ask** | `POST /rag/ask` (`routes/rag.py:121`) → `retrieve_similar_chunks` → `services/rag/generator.py` `generate_rag_answer` (NVIDIA `meta/muse-glimmer-30b` via `createOpenAI` `baseURL https://integrate.api.nvidia.com/v1`) with strict grounding prompt, returns `{answer, query, sources[], model_used}`. The `frontend` also uses `POST /rag/search` directly in `app/api/chat/route.ts:46` to implement Vercel AI SDK streaming with `MIN_SIMILARITY=0.62` + judgment filter. |
| **Stats** | `GET /rag/stats` (`rag.py:172`) → `{total_chunks, indexed_documents, total_documents, vector_dimensions:384, embedding_model, llm_model}`. `GET /rag/documents/{id}/chunks` for inspection. |
| **Research** | `POST /research/search` / `POST /research/fetch` / `GET /research/sessions` (`routes/research.py:1`) — DDGS search + LLM expansion, fetch & auto-ingest. |
| **Evaluation** | `/evaluation` (`routes/evaluation.py:1`) + 8-Tier corpus checklist (`app/eval/*`) for `Benchmark & Trust` UI. |
| **Health** | `GET /health` → `{"status":"ok"}` |

---

## 3. Tech Stack

- **Runtime:** Python 3.12, FastAPI 0.141, Uvicorn 0.52, Pydantic 2.13 + Settings 2.15
- **DB:** PostgreSQL 15+ with `pgvector` extension, SQLAlchemy 2.0 (`psycopg2-binary`), Alembic 1.20 (`alembic.ini`, `alembic/`)
- **ML:** `fastembed 0.8` (`BAAI/bge-small-en-v1.5`), `openai` SDK via `createOpenAI` for NVIDIA NIM, `pypdf 6.18` for PDF parsing, `beautifulsoup4` + `lxml` + `httpx` + `ddgs` for research agent
- **Config:** `app/config.py:1` (`pydantic-settings`, `STORAGE_BASE_PATH`, `DATABASE_URL`, `NVIDIA_API_KEY/BASE_URL/LLM_MODEL`)
- **Tests:** `pytest 9.1` (`tests/`, `tool.pytest.ini_options` `pythonpath=["."]`)

---

## 4. Architecture

```
backend/
├── app/
│   ├── main.py              # FastAPI + CORS (localhost:3000) + 3 routers + /health
│   ├── config.py            # Settings (env)
│   ├── db.py                # Base, engine, SessionLocal, get_db
│   ├── models.py            # Document (UUID PK, file_hash unique, status enum), ResearchSession, RAGChunk (FK CASCADE, ARRAY(Float) embedding, JSONB chunk_metadata, section_title)
│   ├── schemas.py           # Pydantic I/O (DocumentResponse, DuplicateDocumentErrorResponse)
│   ├── routes/
│   │   ├── documents.py     # POST /upload (409), GET /{id}, GET /, DELETE /{id} (new)
│   │   ├── rag.py           # /index, /index-all, /search, /ask, /stats, /documents/{id}/chunks
│   │   ├── research.py      # /search, /fetch, /sessions
│   │   └── evaluation.py    # 8-tier evaluation
│   └── services/
│       ├── document_service.py  # validate_pdf (extension+content-type), SHA256 streaming, dedup, storage/<uuid>/original.pdf, auto index
│       ├── rag/
│       │   ├── indexer.py   # extract → section splitter → clause chunker → embed → pgvector upsert
│       │   ├── retriever.py # cosine similarity, top_k, min_score, document_ids filter
│       │   └── generator.py # system prompt with [Source N] context → NVIDIA LLM
│       └── agent/           # DDGS + query expansion
├── alembic/ + alembic.ini
├── storage/                 # per-document dirs (gitignored, created at runtime)
├── tests/
├── pyproject.toml
└── requirements.txt
```

**Data flow:** `PDF upload → storage/original.pdf → SHA256 → Document(UPLOADED) → indexer (parsed/validated/published) → RAGChunk(embedding[]) → pgvector → retriever → generator → citations [1]`.

---

## 5. Getting Started

**Prereqs:** Python 3.12, PostgreSQL with `pgvector`, `uv` or `pip`.

```bash
cd backend
# env
cp .env.example .env  # fill DATABASE_URL, STORAGE_BASE_PATH, NVIDIA_API_KEY
# install
uv sync  # or pip install -r requirements.txt / pip install -e .
# DB
createdb soli
psql soli -c "CREATE EXTENSION IF NOT EXISTS vector;"
alembic upgrade head
# run
uv run uvicorn app.main:app --reload --port 8000
# or
uvicorn app.main:app --reload
```

**`.env` / `.env.example`:**
```env
DATABASE_URL=postgresql://user:pass@localhost:5432/soli
STORAGE_BASE_PATH=./storage
NVIDIA_API_KEY=nvapi-...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_LLM_MODEL=meta/llama-3.2-11b-vision-instruct
```

**Frontend must point:** `frontend/.env.local` → `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`

---

## 6. API Reference (collapsed)

**Documents**
- `POST /documents/upload` `multipart` → `201 DocumentResponse` / `409 {message, existing_document_id, existing_document_title}` / `400`
- `GET /documents` → `DocumentResponse[]` `ORDER BY uploaded_at DESC`
- `GET /documents/{id}` → `DocumentResponse` / `404`
- `DELETE /documents/{id}` → `204` (removes file dir + `rag_chunks` cascade)

**RAG**
- `POST /rag/index/{id}` → `{chunks_indexed}`
- `POST /rag/index-all` → `{total_documents, results[]}`
- `POST /rag/search` `{query, top_k=5, min_score=0, document_ids?}` → `{query, count, results[]}`
- `POST /rag/ask` `{query, top_k=5, document_ids?}` → `{answer, query, sources[] (source_number, document_id, title, section_title, act_number, similarity_score, chunk_id, content_preview), model_used}`
- `GET /rag/stats` / `GET /rag/documents/{id}/chunks`

**Research**
- `POST /research/search` `{query, max_results=15}` → `{session_id, query, expanded_queries, llm_used, result_count, results[]}`
- `POST /research/fetch` `{result_id, title, url, doc_type, source_name}` → ingests to `documents`
- `GET /research/sessions`

**CORS:** `http://localhost:3000`, `http://127.0.0.1:3000` (`main.py:14`).

---

## 7. RAG Pipeline Details

1. **Extract:** `pypdf` → raw text, `STORAGE_BASE_PATH/<uuid>/original.pdf` retained.
2. **Parse:** section boundary splitter (e.g., `Section 73`) → `DocumentStatus` transitions `uploaded → extracted → parsed`.
3. **Chunk:** clause-level (`(a)`, `(i)`) + overlap, `chunk_index`, `section_title`, `chunk_metadata` JSONB.
4. **Embed:** `fastembed` `BAAI/bge-small-en-v1.5` → 384-d `ARRAY(Float)` (`models.py:77`).
5. **Store:** `rag_chunks` with `document_id FK CASCADE`, `embedding` indexed via `pgvector` IVFFlat/HNSW (see `alembic`).
6. **Retrieve:** cosine similarity, `top_k`, `min_score`, `document_ids` filter. `frontend` adds `MIN_SIMILARITY=0.62` + judgment-aware filter (`route.ts:60`) to avoid hallucinated judgments.
7. **Generate:** `systemPrompt` injects `[Source N] title (§73) — Act: content` (`route.ts:70`), strict `STRICT GROUNDING RULES` + fallback canned `No judgments found…` when `filtered=0`.

---

## 8. Testing

```bash
pytest -q
pytest tests/test_documents.py -v
pytest tests/test_rag.py -v
```

`tool.pytest.ini_options` already sets `pythonpath=["."]`, `filterwarnings` ignores `StarletteDeprecationWarning`.

---

## 9. Deployment Notes

- Set `DATABASE_URL` to managed Postgres with `vector` extension, run `alembic upgrade head` on deploy.
- Mount `STORAGE_BASE_PATH` to persistent volume (or S3 via `hf-cli` bucket skill for multi-replica).
- Set `NVIDIA_API_KEY` as secret; `NVIDIA_BASE_URL` defaults to `integrate.api.nvidia.com`.
- Update CORS to production frontend domain.
- Health: `GET /health` for load balancer.

---

## 10. Troubleshooting

- `409 Duplicate` → same SHA256 already stored, response includes `existing_document_id` to link.
- `400 Invalid file extension/content-type` → only `application/pdf` + `.pdf` allowed (`document_service.py:12`).
- `500 Indexing failed` → check `pypdf` extraction + `fastembed` model download (first run pulls `BAAI/bge-small-en-v1.5`).
- `0 chunks` after upload → check `vector_dimensions` 384 and `pgvector` extension.
- Frontend `Network error: Unable to connect to backend` (`lib/api.ts:116`) → verify `NEXT_PUBLIC_API_BASE_URL` and CORS.
