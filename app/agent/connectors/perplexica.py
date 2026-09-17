"""
Perplexica Connector
====================
Integrates with Perplexica (AI-powered search engine / Perplexity alternative).
Hits the Perplexica API (POST /api/search) to retrieve AI-synthesized answers
along with cited web and academic sources.
"""
from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from app.agent.connectors.base import BaseConnector
from app.config import settings

logger = logging.getLogger(__name__)


class PerplexicaConnector(BaseConnector):
    source_name = "Perplexica AI Search"
    source_domain = "perplexica.local"
    doc_type = "article"
    authority_score = 0.8

    async def search(
        self,
        query: str,
        focus_mode: str = "webSearch",
        max_results: int = 8,
    ) -> list[dict[str, Any]]:
        """
        Search the web via Perplexica AI answering engine.

        Parameters
        ----------
        query:
            Search query string.
        focus_mode:
            Search focus: 'webSearch' (general web), 'academic' (scholarly papers),
            or 'writing' (unsearched generation).
        max_results:
            Maximum number of source documents to return.
        """
        max_results = int(max_results)
        base_url = settings.PERPLEXICA_API_URL.rstrip("/")
        endpoint = f"{base_url}/api/search"

        payload = {
            "query": query,
            "focusMode": focus_mode,
            "optimizationMode": "balanced",
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                resp = await client.post(endpoint, json=payload)
                resp.raise_for_status()
                data = resp.json()

            results: list[dict[str, Any]] = []

            # 1. Parse cited sources from Perplexica response
            sources = data.get("sources", [])
            for src in sources[:max_results]:
                url = src.get("url") or src.get("metadata", {}).get("url", "")
                if not url:
                    continue

                title = (
                    src.get("title")
                    or src.get("metadata", {}).get("title")
                    or url
                )
                snippet = (
                    src.get("snippet")
                    or src.get("pageContent")
                    or src.get("content")
                    or ""
                )[:400]

                parsed_domain = urlparse(url).netloc or self.source_domain

                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source_name": self.source_name,
                    "source_domain": parsed_domain,
                    "doc_type": "report" if focus_mode == "academic" else self.doc_type,
                    "authority_score": 0.85 if focus_mode == "academic" else self.authority_score,
                    "is_pdf": url.lower().endswith(".pdf"),
                })

            # If Perplexica generated an answer/summary, attach snippet if no sources found
            message = data.get("message")
            if not results and message:
                results.append(self._base_result(
                    title=f"Perplexica Summary: {query[:60]}",
                    url=f"{base_url}#search",
                    snippet=message[:400],
                ))

            logger.info(
                "Perplexica returned %d sources for query '%s' (mode=%s)",
                len(results),
                query,
                focus_mode,
            )
            return results

        except httpx.ConnectError:
            logger.warning(
                "Perplexica not reachable at %s. Ensure the Perplexica service is running.",
                endpoint,
            )
            return []
        except httpx.TimeoutException:
            logger.warning("Perplexica request timed out for query '%s'", query)
            return []
        except Exception as exc:
            logger.warning("Perplexica search failed: %s", exc)
            return []
