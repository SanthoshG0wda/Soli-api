"""
Law Commission of India Connector
===================================
Searches Law Commission reports via DuckDuckGo site-scoped search.
"""
from __future__ import annotations

import logging
from ddgs import DDGS

from app.agent.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class LawCommissionConnector(BaseConnector):
    source_name = "Law Commission of India"
    source_domain = "lawcommissionofindia.nic.in"
    doc_type = "report"
    authority_score = 0.85

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search Law Commission reports via DuckDuckGo site-scoped search."""
        max_results = int(max_results)
        full_query = f"site:lawcommissionofindia.nic.in {query} report"
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(full_query, max_results=max_results))
            results = []
            for h in hits:
                url = h.get("href", "")
                if not url:
                    continue
                results.append(self._base_result(
                    title=h.get("title", "Untitled"),
                    url=url,
                    snippet=h.get("body", "")[:300],
                ))
            logger.info("LawCommission (DDG): %d results for '%s'", len(results), query)
            return results
        except Exception as exc:
            logger.warning("LawCommission search failed: %s", exc)
            return []
