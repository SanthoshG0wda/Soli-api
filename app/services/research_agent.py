"""
Research Agent Service — Adapter Layer
=======================================
Thin adapter between the API routes and the AgentHarness.
Maps AgentRunResult → ResearchSession / ResearchResult so the
existing /research/* routes need no changes.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from bs4 import BeautifulSoup

from app.agent.harness import AgentHarness, AgentRunResult
from app.agent import registry   # triggers tool registration

logger = logging.getLogger(__name__)


# ── Public result types (kept for API route compatibility) ────────────────────

@dataclass
class ResearchResult:
    id: str = ""
    title: str = ""
    url: str = ""
    snippet: str = ""
    source_name: str = ""
    source_domain: str = ""
    doc_type: str = "unknown"
    authority_score: float = 0.5
    relevance_score: float = 0.0
    combined_score: float = 0.0
    is_pdf: bool = False
    fetched_text: str | None = None


@dataclass
class ResearchSession:
    session_id: str = ""
    original_query: str = ""
    expanded_queries: list[str] = field(default_factory=list)
    results: list[ResearchResult] = field(default_factory=list)
    llm_used: bool = False
    trace: dict | None = None


# ── Main entry point ──────────────────────────────────────────────────────────

async def run_research_search(
    query: str,
    max_results: int = 15,
    db: Any = None,
) -> ResearchSession:
    """
    Run the full agent harness and return results as a ResearchSession.
    Delegates all logic to AgentHarness.
    """
    harness = AgentHarness(registry=registry, db=db)
    run: AgentRunResult = await harness.run(query)

    results = [
        ResearchResult(
            id=r.result_id,
            title=r.title,
            url=r.url,
            snippet=r.snippet,
            source_name=r.source_name,
            source_domain=r.source_domain,
            doc_type=r.doc_type,
            authority_score=r.authority_score,
            relevance_score=r.relevance_score,
            combined_score=r.combined_score,
            is_pdf=r.is_pdf,
        )
        for r in run.results[:max_results]
    ]

    return ResearchSession(
        session_id=run.session_id,
        original_query=query,
        expanded_queries=[query],
        results=results,
        llm_used=run.llm_used,
        trace=run.trace.to_dict() if run.trace else None,
    )


# ── Document fetching (used by /research/fetch) ────────────────────────────────

async def fetch_document_text(url: str) -> tuple[str | None, bool]:
    """Fetch PDF or HTML content from a URL. Returns (text, is_pdf)."""
    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "Soli-Research-Agent/2.0"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")

            if "pdf" in ct or url.lower().endswith(".pdf"):
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(resp.content))
                text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
                return text[:50000], True

            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            main = soup.find("main") or soup.find("article") or soup.find("body")
            return (main.get_text(separator="\n", strip=True) if main else "")[:50000], False

    except Exception as exc:
        logger.warning("fetch_document_text failed for %s: %s", url, exc)
        return None, False
