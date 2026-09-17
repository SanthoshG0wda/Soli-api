"""
SearXNG Connector
=================
Direct integration with SearXNG (the meta-search engine underpinning Perplexica).
Queries SearXNG's JSON API to retrieve aggregated results across Google, Bing,
DuckDuckGo, Wikipedia, ArXiv, and other search backends.
"""
from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from app.agent.connectors.base import BaseConnector
from app.config import settings

logger = logging.getLogger(__name__)


class SearxngConnector(BaseConnector):
    source_name = "SearXNG Meta Search"
    source_domain = "searxng.local"
    doc_type = "unknown"
    authority_score = 0.75

    async def search(
        self,
        query: str,
        categories: str = "general",
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search the web using a SearXNG instance.

        Parameters
        ----------
        query:
            Search terms.
        categories:
            SearXNG categories, e.g. 'general', 'science', 'it', 'news'.
        max_results:
            Maximum number of results to return.
        """
        max_results = int(max_results)
        base_url = settings.SEARXNG_API_URL.rstrip("/")
        endpoint = f"{base_url}/search"

        params = {
            "q": query,
            "format": "json",
            "categories": categories,
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(endpoint, params=params)
                resp.raise_for_status()
                data = resp.json()

            results: list[dict[str, Any]] = []
            raw_hits = data.get("results", [])

            for hit in raw_hits[:max_results]:
                url = hit.get("url", "")
                if not url:
                    continue

                title = hit.get("title", "Untitled")
                snippet = hit.get("content", "")[:400]
                parsed_domain = urlparse(url).netloc or self.source_domain
                engine = hit.get("engine", "searxng")

                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source_name": f"{self.source_name} ({engine})",
                    "source_domain": parsed_domain,
                    "doc_type": "report" if categories == "science" else self.doc_type,
                    "authority_score": self.authority_score,
                    "is_pdf": url.lower().endswith(".pdf"),
                })

            logger.info(
                "SearXNG returned %d results for '%s' (category=%s)",
                len(results),
                query,
                categories,
            )
            return results

        except httpx.ConnectError:
            logger.warning(
                "SearXNG not reachable at %s. Ensure SearXNG is running or update SEARXNG_API_URL in .env.",
                endpoint,
            )
            return []
        except httpx.TimeoutException:
            logger.warning("SearXNG request timed out for query '%s'", query)
            return []
        except Exception as exc:
            logger.warning("SearXNG search failed: %s", exc)
            return []
