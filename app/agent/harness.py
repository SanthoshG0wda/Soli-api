"""
Agent Harness
=============
The main agent loop. Orchestrates tool-calling via NVIDIA NIM function calling.
Falls back to parallel connector execution when no API key is configured.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.agent.tool import ToolRegistry
from app.agent.trace import AgentTrace, StepKind
from app.config import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_ITERATIONS = 5           # optimized for fast, responsive tool-calling loop
SYSTEM_PROMPT = """\
You are Soli Research Agent, an expert Indian legal research assistant.
Your job is to find authoritative Indian legal documents (Acts, Judgments, \
Law Commission Reports) relevant to the user's research query.

Use the available tools strategically:
1. Start with source-specific searches (search_indian_kanoon, search_india_code) \
   for targeted results.
2. Use search_ddg for broader discovery.
3. Use check_kb_duplicate before recommending any document to avoid duplicates.
4. Use fetch_url_content to get the actual text of a promising document when needed.

When you have gathered enough results (at least 5-10 unique documents), \
produce your final answer as a JSON object:
{
  "results": [
    {
      "title": "...",
      "url": "...",
      "snippet": "...",
      "source_name": "...",
      "source_domain": "...",
      "doc_type": "act|judgment|report|article|unknown",
      "authority_score": 0.0-1.0,
      "is_pdf": true|false
    }
  ]
}

Include ONLY documents that are directly relevant to the query. \
Do NOT include duplicates already in the knowledge base.\
"""


# ── Result Types ──────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    url: str = ""
    snippet: str = ""
    source_name: str = ""
    source_domain: str = ""
    doc_type: str = "unknown"
    authority_score: float = 0.5
    relevance_score: float = 0.7
    combined_score: float = 0.0
    is_pdf: bool = False


@dataclass
class AgentRunResult:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    results: list[AgentResult] = field(default_factory=list)
    trace: AgentTrace | None = None
    llm_used: bool = False
    iterations_used: int = 0


# ── Agent Harness ─────────────────────────────────────────────────────────────

class AgentHarness:
    """
    Orchestrates the research agent loop.

    With NVIDIA_API_KEY: uses NIM tool-calling to let the LLM decide which
    connectors to call, in what order, how many times.

    Without NVIDIA_API_KEY: runs all connectors in parallel (keyword fallback).
    """

    def __init__(self, registry: ToolRegistry, db: Any = None) -> None:
        self.registry = registry
        self.db = db  # SQLAlchemy session, passed through to KB connector tools

    # ── Public Entry Point ────────────────────────────────────────────────────

    async def run(self, query: str) -> AgentRunResult:
        trace = AgentTrace(query=query)
        run = AgentRunResult(query=query)

        if settings.NVIDIA_API_KEY:
            await self._llm_loop(query, trace, run)
        else:
            await self._fallback_parallel(query, trace, run)

        run.trace = trace
        run.iterations_used = trace.iterations_used
        run.llm_used = trace.llm_used
        trace.done(f"{len(run.results)} results")
        return run

    # ── LLM Tool-Calling Loop ─────────────────────────────────────────────────

    async def _llm_loop(self, query: str, trace: AgentTrace, run: AgentRunResult) -> None:
        trace.llm_used = True
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Research query: {query}"},
        ]
        tools = self.registry.nim_schemas()

        for iteration in range(1, MAX_ITERATIONS + 1):
            trace.iterations_used = iteration
            t0 = time.monotonic()

            try:
                response = await self._call_nim(messages, tools)
            except Exception as exc:
                trace.error(f"NIM call failed at iteration {iteration}: {exc}")
                logger.warning("NIM call failed: %s — falling back to parallel", exc)
                await self._fallback_parallel(query, trace, run)
                return

            choice = response["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason", "")

            # ── Think step: capture any reasoning or text the model produced
            thought = message.get("reasoning_content") or message.get("content")
            if thought:
                trace.think(str(thought), iteration=iteration)

            # ── Done: model produced its final JSON answer
            if finish_reason == "stop" or not message.get("tool_calls"):
                content = message.get("content") or ""
                parsed = self._parse_final_answer(content)
                if parsed:
                    run.results = parsed
                    logger.info("Agent done after %d iterations: %d results", iteration, len(parsed))
                    return
                # If we got no parseable results, keep looping (may have more tool calls)
                if iteration >= MAX_ITERATIONS:
                    break

            # ── Tool calls: dispatch each one
            tool_calls = message.get("tool_calls", [])
            if not tool_calls:
                break

            # Add assistant message with tool calls to history
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": tool_calls,
            }
            messages.append(assistant_msg)

            # Execute all tool calls (possibly parallel)
            tool_results = await asyncio.gather(*[
                self._dispatch_tool(tc, trace, iteration) for tc in tool_calls
            ])

            # Add each tool result to message history
            for tc, result in zip(tool_calls, tool_results):
                tool_call_id = tc.get("id", str(uuid.uuid4()))
                trace.observe(tc["function"]["name"], result, elapsed_ms=(time.monotonic() - t0) * 1000, iteration=iteration)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tc["function"]["name"],
                    "content": json.dumps(result, default=str)[:4000],
                })

        # If loop ended without final answer, collect anything from tool observations
        run.results = self._collect_from_trace(trace)
        logger.info("Agent loop ended: %d results collected from trace", len(run.results))

    async def _dispatch_tool(self, tool_call: dict, trace: AgentTrace, iteration: int) -> Any:
        fn = tool_call.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", "{}")
        trace.act(name, json.loads(args) if isinstance(args, str) else args, iteration=iteration)
        return await self.registry.dispatch(name, args)

    # ── NIM API Call ──────────────────────────────────────────────────────────

    async def _call_nim(self, messages: list[dict], tools: list[dict]) -> dict:
        payload = {
            "model": settings.NVIDIA_LLM_MODEL,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.NVIDIA_BASE_URL}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            return resp.json()

    # ── Fallback: parallel connector execution ────────────────────────────────

    async def _fallback_parallel(self, query: str, trace: AgentTrace, run: AgentRunResult) -> None:
        """Run all search tools in parallel without LLM orchestration."""
        trace.think("LLM not available — running all connectors in parallel", iteration=0)
        search_tools = [t for t in self.registry.all() if "search" in t.name]
        tasks = [
            self.registry.dispatch(t.name, {"query": query, "max_results": 8})
            for t in search_tools
        ]
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        seen_urls: set[str] = set()
        all_results: list[AgentResult] = []

        for raw_list in results_lists:
            if isinstance(raw_list, Exception) or not isinstance(raw_list, list):
                continue
            for item in raw_list:
                if not isinstance(item, dict):
                    continue
                url = item.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                all_results.append(self._dict_to_result(item, query))

        # Rank
        all_results.sort(key=lambda r: r.combined_score, reverse=True)
        run.results = all_results[:15]
        trace.iterations_used = 1

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_final_answer(self, content: str) -> list[AgentResult] | None:
        """Extract the JSON results array from the model's final answer."""
        try:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group())
            raw_results = data.get("results", [])
            if not raw_results:
                return None
            return [self._dict_to_result(r, "") for r in raw_results if isinstance(r, dict)]
        except Exception:
            return None

    def _collect_from_trace(self, trace: AgentTrace) -> list[AgentResult]:
        """If LLM loop ended without a final answer, collect items from OBSERVE steps."""
        seen: set[str] = set()
        results: list[AgentResult] = []
        for step in trace.steps:
            if step.kind != StepKind.OBSERVE or not isinstance(step.content, list):
                continue
            for item in step.content:
                if not isinstance(item, dict):
                    continue
                url = item.get("url", "")
                if url in seen:
                    continue
                seen.add(url)
                results.append(self._dict_to_result(item, trace.query))
        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:15]

    @staticmethod
    def _dict_to_result(d: dict, query: str) -> AgentResult:
        authority = float(d.get("authority_score", 0.5))
        # Simple keyword relevance
        q_terms = set(re.findall(r"\w+", query.lower()))
        text = (d.get("title", "") + " " + d.get("snippet", "")).lower()
        txt_terms = set(re.findall(r"\w+", text))
        relevance = min(1.0, len(q_terms & txt_terms) / max(len(q_terms), 1) + 0.1)
        pdf_bonus = 0.05 if d.get("is_pdf") else 0.0
        combined = 0.45 * relevance + 0.45 * authority + 0.10 * pdf_bonus

        return AgentResult(
            result_id=str(uuid.uuid4()),
            title=d.get("title", "Untitled"),
            url=d.get("url", ""),
            snippet=d.get("snippet", "")[:300],
            source_name=d.get("source_name", ""),
            source_domain=d.get("source_domain", ""),
            doc_type=d.get("doc_type", "unknown"),
            authority_score=authority,
            relevance_score=round(relevance, 3),
            combined_score=round(combined, 3),
            is_pdf=bool(d.get("is_pdf")),
        )
