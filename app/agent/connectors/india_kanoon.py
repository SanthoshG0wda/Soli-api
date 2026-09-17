"""
IndiaKanoon Connector
=====================
Searches Indian Supreme Court and High Court judgments.
Uses the IndiaKanoon search page (scrape) with HTML-fallback extraction.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from app.agent.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Soli-Research-Agent/2.0)",
    "Accept": "text/html,application/xhtml+xml",
}


class IndiaKanoonConnector(BaseConnector):
    source_name = "Indian Kanoon"
    source_domain = "indiankanoon.org"
    doc_type = "judgment"
    authority_score = 0.9

    async def search(self, query: str, max_results: int = 8) -> list[dict]:
        """Search Indian Kanoon for SC/HC judgments matching query."""
        max_results = int(max_results)
        url = f"https://indiankanoon.org/search/?formInput={quote_plus(query)}&pagenum=0"
        try:
            async with httpx.AsyncClient(headers=HEADERS, timeout=20.0, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            results = []
            for item in soup.select(".result")[:max_results]:
                title_tag = item.select_one("a.result-title") or item.select_one("a")
                if not title_tag:
                    continue
                href = title_tag.get("href", "")
                full_url = f"https://indiankanoon.org{href}" if href.startswith("/") else href
                snippet_tag = item.select_one(".snippet") or item.select_one("p")
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                results.append(self._base_result(
                    title=title_tag.get_text(strip=True),
                    url=full_url,
                    snippet=snippet,
                ))
            logger.info("IndiaKanoon: %d results for '%s'", len(results), query)
            return results
        except Exception as exc:
            logger.warning("IndiaKanoon search failed: %s", exc)
            return []

    async def fetch_judgment(self, url: str) -> str | None:
        """Fetch full judgment text from an IndiaKanoon document URL."""
        try:
            async with httpx.AsyncClient(headers=HEADERS, timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            # Main judgment body is in #judgments div
            body = soup.select_one("#judgments") or soup.select_one(".judgment")
            if body:
                for tag in body(["script", "style", "nav"]):
                    tag.decompose()
                return body.get_text(separator="\n", strip=True)[:30000]
        except Exception as exc:
            logger.warning("IndiaKanoon fetch failed for %s: %s", url, exc)
        return None
