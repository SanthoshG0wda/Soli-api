"""
Soli Research Agent — Tool Registration
=========================================
Registers all connector-backed tools into the global registry at import time.
Import this module once at app startup to activate the full agent harness.
"""
from __future__ import annotations

import logging
from typing import Any

from app.agent.tool import registry, tool, ToolParameter
from app.agent.connectors.india_kanoon import IndiaKanoonConnector
from app.agent.connectors.india_code import IndiaCodeConnector
from app.agent.connectors.law_commission import LawCommissionConnector
from app.agent.connectors.ddg import DDGConnector
from app.agent.connectors.perplexica import PerplexicaConnector
from app.agent.connectors.searxng import SearxngConnector

logger = logging.getLogger(__name__)

# ── Instantiate connectors ────────────────────────────────────────────────────

_kanoon = IndiaKanoonConnector()
_india_code = IndiaCodeConnector()
_law_comm = LawCommissionConnector()
_ddg = DDGConnector()
_perplexica = PerplexicaConnector()
_searxng = SearxngConnector()


# ── Register tools ─────────────────────────────────────────────────────────────

@tool(
    name="search_indian_kanoon",
    description=(
        "Search Indian Supreme Court and High Court judgments on IndiaKanoon. "
        "Use this for case law, judicial decisions, and precedents."
    ),
    parameters=[
        ToolParameter("query", "string", "Legal search query (case name, legal principle, or topic)", required=True),
        ToolParameter("max_results", "integer", "Maximum number of results to return (default 8)", required=False),
    ],
)
async def search_indian_kanoon(query: str, max_results: int = 8) -> list[dict]:
    return await _kanoon.search(query, max_results)


@tool(
    name="fetch_judgment_text",
    description=(
        "Fetch the full text of a specific judgment from its IndiaKanoon URL. "
        "Use this to read the content of a promising judgment before adding it to the KB."
    ),
    parameters=[
        ToolParameter("url", "string", "Full IndiaKanoon judgment URL", required=True),
    ],
)
async def fetch_judgment_text(url: str) -> dict:
    text = await _kanoon.fetch_judgment(url)
    return {"text": text[:3000] if text else None, "url": url, "fetched": text is not None}


@tool(
    name="search_india_code",
    description=(
        "Search India Code (indiacode.nic.in) for Central Acts and legislation. "
        "Use this for finding specific Acts, their sections, and amendments."
    ),
    parameters=[
        ToolParameter("query", "string", "Act name or subject area to search", required=True),
        ToolParameter("max_results", "integer", "Maximum number of results to return (default 6)", required=False),
    ],
)
async def search_india_code(query: str, max_results: int = 6) -> list[dict]:
    return await _india_code.search(query, max_results)


@tool(
    name="search_law_commission",
    description=(
        "Search Law Commission of India reports. "
        "Use this for finding Law Commission recommendations on specific legal topics."
    ),
    parameters=[
        ToolParameter("query", "string", "Legal topic to search in Law Commission reports", required=True),
        ToolParameter("max_results", "integer", "Maximum number of results to return (default 5)", required=False),
    ],
)
async def search_law_commission(query: str, max_results: int = 5) -> list[dict]:
    return await _law_comm.search(query, max_results)


@tool(
    name="search_ddg",
    description=(
        "Broad web search scoped to Indian legal domains (IndiaKanoon, India Code, "
        "Legislative.gov.in, LiveLaw, Bar & Bench). "
        "Use this as a fallback when source-specific searches return too few results."
    ),
    parameters=[
        ToolParameter("query", "string", "Search query", required=True),
        ToolParameter("max_results", "integer", "Maximum number of results to return (default 10)", required=False),
    ],
)
async def search_ddg(query: str, max_results: int = 10) -> list[dict]:
    return await _ddg.search(query, max_results)


@tool(
    name="check_kb_duplicate",
    description=(
        "Check if a document URL already exists in the local knowledge base. "
        "ALWAYS call this before recommending a document for ingestion."
    ),
    parameters=[
        ToolParameter("url", "string", "Document URL to check for duplicates", required=True),
    ],
)
async def check_kb_duplicate(url: str) -> dict:
    """
    Note: This tool requires DB access. It is registered here but the harness
    injects a DB-aware version when a session is available.
    Without DB, always returns exists=false (safe default).
    """
    return {"exists": False, "note": "No DB session available for duplicate check"}


@tool(
    name="fetch_url_content",
    description=(
        "Fetch and extract the text content from any URL (PDF or HTML). "
        "Use this to read the full content of a document before deciding to ingest it."
    ),
    parameters=[
        ToolParameter("url", "string", "URL to fetch content from", required=True),
    ],
)
async def fetch_url_content(url: str) -> dict:
    import io
    import httpx
    from bs4 import BeautifulSoup
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True,
                                     headers={"User-Agent": "Soli-Research-Agent/2.0"}) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "pdf" in ct or url.lower().endswith(".pdf"):
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(resp.content))
                text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
                return {"text": text[:5000], "is_pdf": True, "url": url}
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            main = soup.find("main") or soup.find("article") or soup.find("body")
            text = (main.get_text(separator="\n", strip=True) if main else "")[:5000]
            return {"text": text, "is_pdf": False, "url": url}
    except Exception as exc:
        return {"error": str(exc), "url": url}


@tool(
    name="search_perplexica",
    description=(
        "Search the web using Perplexica AI search engine. "
        "Returns AI-synthesized responses and cited sources from the web and academic databases. "
        "Use this for broad topical research, general legal/business context, and finding external references."
    ),
    parameters=[
        ToolParameter("query", "string", "Search query or question", required=True),
        ToolParameter("focus_mode", "string", "Search mode: 'webSearch' (default) or 'academic'", required=False),
        ToolParameter("max_results", "integer", "Maximum number of source results to return (default 8)", required=False),
    ],
)
async def search_perplexica(query: str, focus_mode: str = "webSearch", max_results: int = 8) -> list[dict]:
    return await _perplexica.search(query, focus_mode=focus_mode, max_results=max_results)


@tool(
    name="search_searxng",
    description=(
        "Search the web using SearXNG meta-search engine (aggregates Google, Bing, DuckDuckGo, ArXiv, Wikipedia). "
        "Use this for comprehensive meta-search across multiple search providers."
    ),
    parameters=[
        ToolParameter("query", "string", "Search query", required=True),
        ToolParameter("categories", "string", "Search category: 'general' (default), 'science', 'it', 'news'", required=False),
        ToolParameter("max_results", "integer", "Maximum number of results to return (default 10)", required=False),
    ],
)
async def search_searxng(query: str, categories: str = "general", max_results: int = 10) -> list[dict]:
    return await _searxng.search(query, categories=categories, max_results=max_results)


logger.info(
    "Agent harness initialised with %d tools: %s",
    len(registry.all()),
    [t.name for t in registry.all()],
)
