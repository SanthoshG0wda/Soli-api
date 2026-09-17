"""
India Code Connector
====================
Searches indiacode.nic.in for Central Acts.
Uses DuckDuckGo site-scoped search (most reliable for government sites
that frequently change their internal URL structure).
"""
from __future__ import annotations

import logging
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

from app.agent.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Soli-Research-Agent/2.0)",
    "Accept": "text/html,application/xhtml+xml",
}


class IndiaCodeConnector(BaseConnector):
    source_name = "India Code"
    source_domain = "indiacode.nic.in"
    doc_type = "act"
    authority_score = 1.0

    async def search(self, query: str, max_results: int = 6) -> list[dict]:
        """Search India Code via DuckDuckGo site-scoped search."""
        max_results = int(max_results)
        full_query = f"site:indiacode.nic.in {query} Act"
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(full_query, max_results=max_results))
            results = []
            for h in hits:
                url = h.get("href", "")
                if not url or "indiacode.nic.in" not in url:
                    continue
                results.append(self._base_result(
                    title=h.get("title", "Untitled"),
                    url=url,
                    snippet=h.get("body", "")[:300],
                ))
            logger.info("IndiaCode (DDG): %d results for '%s'", len(results), query)
            return results
        except Exception as exc:
            logger.warning("IndiaCode search failed: %s", exc)
            return []

    async def fetch_act_text(self, url: str) -> str | None:
        """Fetch the full text of a Central Act."""
        try:
            async with httpx.AsyncClient(headers=HEADERS, timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            main = soup.find("main") or soup.find("article") or soup.find("body")
            if main:
                return main.get_text(separator="\n", strip=True)[:40000]
        except Exception as exc:
            logger.warning("IndiaCode fetch failed for %s: %s", url, exc)
        return None
