from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.documents import router as documents_router
from app.routes.rag import router as rag_router
from app.routes.evaluation import router as evaluation_router


app = FastAPI(
    title="Soli Legal Knowledge Base - Ingestion Platform",
    version="0.1.0",
    description="Internal administration platform for legislation ingestion, extraction, and verification.",
)

# Enable CORS for Next.js frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(rag_router)
app.include_router(evaluation_router)


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok"}
