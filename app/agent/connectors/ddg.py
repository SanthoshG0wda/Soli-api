"""
DuckDuckGo Connector
====================
Broad fallback web search scoped to authoritative Indian legal domains.
Uses the ddgs library (renamed from duckduckgo-search).
"""
from __future__ import annotations

import logging
from urllib.parse import urlparse

from ddgs import DDGS

from app.agent.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

LEGAL_DOMAINS = [
    "indiankanoon.org",
    "indiacode.nic.in",
    "legislative.gov.in",
    "lawcommissionofindia.nic.in",
    "sci.gov.in",
    "barandbench.com",
    "livelaw.in",
    "manupatra.com",
    "scconline.com",
]

AUTHORITY_MAP: dict[str, tuple[str, float]] = {
    "indiacode.nic.in":         ("act",      1.0),
    "legislative.gov.in":       ("act",      1.0),
    "sci.gov.in":               ("judgment", 1.0),
    "indiankanoon.org":         ("judgment", 0.9),
    "lawcommissionofindia.nic.in": ("report", 0.85),
    "livelaw.in":               ("article",  0.65),
    "barandbench.com":          ("article",  0.60),
    "manupatra.com":            ("judgment", 0.80),
}


def _classify(url: str) -> tuple[str, str, float]:
    """Return (source_name, doc_type, authority_score) for a URL."""
    host = urlparse(url).netloc.lower().lstrip("www.")
    for domain, (dtype, score) in AUTHORITY_MAP.items():
        if domain in host:
            return domain, dtype, score
    return host, "unknown", 0.4


class DDGConnector(BaseConnector):
    source_name = "DuckDuckGo (Legal)"
    source_domain = "duckduckgo.com"
    doc_type = "unknown"
    authority_score = 0.5

    async def search(self, query: str, max_results: int = 10) -> list[dict]:
        """Search DuckDuckGo scoped to Indian legal domains."""
        max_results = int(max_results)
        site_filter = " OR ".join(f"site:{d}" for d in LEGAL_DOMAINS[:6])
        full_query = f"{query} ({site_filter})"
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(full_query, max_results=max_results))
        except Exception as exc:
            logger.warning("DuckDuckGo search failed: %s", exc)
            return []

        results = []
        for h in hits:
            url = h.get("href", "")
            if not url:
                continue
            source_domain, doc_type, authority = _classify(url)
            results.append({
                "title": h.get("title", "Untitled"),
                "url": url,
                "snippet": h.get("body", "")[:400],
                "source_name": source_domain,
                "source_domain": source_domain,
                "doc_type": doc_type,
                "authority_score": authority,
                "is_pdf": url.lower().endswith(".pdf"),
            })
        logger.info("DDG: %d results for '%s'", len(results), query)
        return results
