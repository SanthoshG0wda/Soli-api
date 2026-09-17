"""
RAG Legal Synthesis Generator
=============================
Assembles retrieved legal chunk passages into a structured legal prompt and
queries `meta/muse-glimmer-30b` via NVIDIA NIM to generate an authoritative,
cited legal explanation referencing exact statutory sections and provisions.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import settings
from app.services.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are Soli Legal Assistant, an expert Indian legal analysis AI.
Your job is to answer legal queries using ONLY the authoritative context passages provided below.

Guidelines:
1. Ground your answer strictly in the provided statutory text and passages.
2. Explicitly cite the statutory Act name, Section number, and clause for every legal proposition (e.g. "Under Section 73 of the Indian Contract Act, 1872...").
3. If the context does not contain sufficient details to answer the query completely, clearly acknowledge what is covered and what is missing.
4. Maintain a formal, objective, and precise legal tone.
"""


async def generate_rag_answer(
    query: str,
    chunks: list[RetrievedChunk],
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """
    Synthesize an authoritative legal answer grounded in retrieved chunks.

    Returns
    -------
    dict with 'answer', 'sources', 'query', and 'model_used'.
    """
    if not chunks:
        return {
            "answer": "No relevant documents or statutory sections were found in the knowledge base for this query.",
            "sources": [],
            "query": query,
            "model_used": settings.NVIDIA_LLM_MODEL,
        }

    # 1. Format context passages
    context_blocks = []
    for i, c in enumerate(chunks, 1):
        header = f"[Source {i}: {c.document_title}"
        if c.section_title:
            header += f" | {c.section_title}"
        header += f" (Score: {c.score})]"
        context_blocks.append(f"{header}\n{c.content}\n")

    full_context = "\n---\n".join(context_blocks)

    user_message = f"""\
LEGAL CONTEXT:
{full_context}

USER QUESTION:
{query}

Please provide a detailed, accurate legal answer with exact section citations based on the context above:
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    payload = {
        "model": settings.NVIDIA_LLM_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                f"{settings.NVIDIA_BASE_URL}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        raw_content = choice["message"].get("content") or ""
        # If content has think tags, clean them
        if "<think>" in raw_content and "</think>" in raw_content:
            raw_content = raw_content.split("</think>")[-1].strip()
        answer = raw_content or choice["message"].get("reasoning_content") or ""

    except Exception as exc:
        logger.error("LLM synthesis failed: %s", exc)
        answer = ""

    # If LLM answer is empty or failed, generate a grounded extraction from the top retrieved chunks
    if not answer.strip():
        top_chunk = chunks[0]
        sec = f" ({top_chunk.section_title})" if top_chunk.section_title else ""
        answer = (
            f"Based on **{top_chunk.document_title}**{sec}, the relevant statutory provision provides:\n\n"
            f"> {top_chunk.content.strip()}\n\n"
            f"Refer to the retrieved source chunks below for the complete legislative provisions."
        )

    sources = [
        {
            "source_number": i + 1,
            "document_title": c.document_title,
            "section_title": c.section_title,
            "act_number": c.act_number,
            "similarity_score": c.score,
            "chunk_id": c.chunk_id,
            "content_preview": c.content[:250] + ("..." if len(c.content) > 250 else ""),
        }
        for i, c in enumerate(chunks)
    ]

    return {
        "answer": answer.strip(),
        "query": query,
        "sources": sources,
        "model_used": settings.NVIDIA_LLM_MODEL,
    }
