"""
Legal Semantic Chunker
======================
Splits legal documents (Acts, Codes, Judgments, Reports) into semantic chunks.
Understands Indian statutory structures:
- Separates 'Arrangement of Sections' (Table of Contents) from substantive sections.
- Identifies numbered sections (e.g., '1. Short title...', '2. Definitions...', 'Section 73...').
- Breaks large Definition/Interpretation sections (Section 2) into dedicated clause chunks (e.g., Clause (n) 'sexual harassment').
- Preserves statutory titles and boundaries while keeping bounded window sizes for embedding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Regex to match Indian Statute Section headers:
# Matches single and multi-line statutory headings across Indian Gazettes:
# '1. Short title...' or '1.Short title...'
# '2. Definitions. —'
# '43. Penalty and compensation for damage to computer, computer system,\netc . -'
# '67. Punishment for publishing or transmitting obscene material in electronic\nform. -'
# 'Section 73. Compensation...'
# 'Article 21. ...'
INDIAN_SECTION_PATTERN = re.compile(
    r'(?m)^(?:\s*)(?:(?:SECTION|Section|SEC\.|Sec\.|ARTICLE|Article|Art\.|ORDER|Order)\s*([0-9]+[A-Za-z]*)|([0-9]+[A-Za-z]*)\.)\s*([A-Z][^\n\r\.\:\-\—]{1,120}?(?:\n\s*[a-zA-Z][^\n\r\.\:\-\—]{1,120}?)?)[\.\:\-\—]'
)

# Clause pattern inside Definitions / Section 2:
# Matches primary clauses: '(a) "aggrieved woman" means...', '(n) "sexual harassment" includes...'
DEF_CLAUSE_PATTERN = re.compile(
    r'(?m)^(?:\s*)(?:[0-9]+\s+)?\(([a-z])\)\s*[“"\'‘\’\s]*([a-zA-Z\s\-]+?)[”"\'‘\’\s]*(?:means|includes)',
    re.I
)


@dataclass
class TextChunk:
    chunk_index: int
    content: str
    section_title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_legal_text(
    text: str,
    act_title: str | None = None,
    max_words: int = 400,
    overlap_words: int = 50,
) -> list[TextChunk]:
    """
    Splits legal text into semantically coherent statutory chunks.
    """
    if not text or not text.strip():
        return []

    chunks: list[TextChunk] = []

    # 1. Detect and separate Arrangement of Sections (Table of Contents)
    toc_match = re.search(r'(?i)\bARRANGEMENT\s+OF\s+SECTIONS\b', text)
    enactment_match = re.search(r'(?i)\bBE\s+it\s+enacted\b.*?(?:as\s+follows\s*[:\—\-]*)?', text)

    substantive_text = text
    if toc_match and enactment_match and enactment_match.start() > toc_match.start():
        toc_text = text[toc_match.start():enactment_match.start()].strip()
        substantive_text = text[enactment_match.end():]

        # Add TOC as an explicitly tagged chunk (low priority for substantive queries)
        toc_title = f"{act_title} - Arrangement of Sections (Table of Contents)" if act_title else "Arrangement of Sections"
        chunks.append(TextChunk(
            chunk_index=len(chunks),
            content=f"[{toc_title}]\n{toc_text}",
            section_title=toc_title,
            metadata={
                "is_toc": True,
                "word_count": len(toc_text.split()),
                "char_length": len(toc_text),
            }
        ))
    elif toc_match:
        # If no enactment clause found, look for Chapter I or Section 1 after TOC
        sec1_match = re.search(r'(?m)^(?:\s*)(?:1\.|Section\s+1\.)\s+Short\s+title', text)
        if sec1_match and sec1_match.start() > toc_match.start() + 200:
            toc_text = text[toc_match.start():sec1_match.start()].strip()
            substantive_text = text[sec1_match.start():]
            toc_title = f"{act_title} - Arrangement of Sections (Table of Contents)" if act_title else "Arrangement of Sections"
            chunks.append(TextChunk(
                chunk_index=len(chunks),
                content=f"[{toc_title}]\n{toc_text}",
                section_title=toc_title,
                metadata={
                    "is_toc": True,
                    "word_count": len(toc_text.split()),
                    "char_length": len(toc_text),
                }
            ))

    # 2. Extract sections from substantive text
    raw_matches = list(INDIAN_SECTION_PATTERN.finditer(substantive_text))
    # Filter out footnotes (e.g. '1. Subs. by Act...', '2. Ins. by Act...')
    matches = [
        m for m in raw_matches
        if not (m.group(3) or "").strip().lower().startswith(("subs", "ins", "omitted", "rep.", "see "))
    ]

    if len(matches) >= 2:
        for idx, match in enumerate(matches):
            sec_num = match.group(1) or match.group(2)
            sec_heading = re.sub(r'\s+', ' ', (match.group(3) or "")).strip()
            start_pos = match.start()
            end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(substantive_text)

            sec_content = substantive_text[start_pos:end_pos].strip()
            full_sec_title = f"Section {sec_num}"
            if sec_heading:
                full_sec_title += f": {sec_heading}"
            if act_title:
                full_sec_title = f"{act_title} - {full_sec_title}"

            # 3. Special handling for Section 2 (Definitions / Interpretation clause)
            if sec_num in ("2", "3") and ("definition" in sec_heading.lower() or "interpretation" in sec_heading.lower() or len(sec_content) > 1500):
                clause_chunks = _chunk_definition_clauses(
                    sec_content=sec_content,
                    base_title=full_sec_title,
                    sec_num=sec_num,
                    sec_heading=sec_heading,
                    start_index=len(chunks)
                )
                if clause_chunks:
                    chunks.extend(clause_chunks)
                    continue

            # Standard section chunking
            words = sec_content.split()
            if len(words) <= max_words:
                chunks.append(TextChunk(
                    chunk_index=len(chunks),
                    content=f"[{full_sec_title}]\n{sec_content}",
                    section_title=full_sec_title,
                    metadata={
                        "section_number": sec_num,
                        "heading": sec_heading,
                        "is_toc": False,
                        "word_count": len(words),
                        "char_length": len(sec_content),
                    }
                ))
            else:
                # Long section: split into sliding windows with overlap
                sub_chunks = _split_sliding_window(
                    words,
                    max_words=max_words,
                    overlap=overlap_words,
                    prefix=f"[{full_sec_title}]\n"
                )
                for sc in sub_chunks:
                    chunks.append(TextChunk(
                        chunk_index=len(chunks),
                        content=sc,
                        section_title=full_sec_title,
                        metadata={
                            "section_number": sec_num,
                            "heading": sec_heading,
                            "is_toc": False,
                            "word_count": len(sc.split()),
                            "char_length": len(sc),
                        }
                    ))
    else:
        # Fallback: sliding window chunking across entire text
        words = substantive_text.split()
        sub_chunks = _split_sliding_window(
            words,
            max_words=max_words,
            overlap=overlap_words,
            prefix=f"[{act_title}]\n" if act_title else ""
        )
        for sc in sub_chunks:
            chunks.append(TextChunk(
                chunk_index=len(chunks),
                content=sc,
                section_title=act_title,
                metadata={
                    "is_toc": False,
                    "word_count": len(sc.split()),
                    "char_length": len(sc),
                }
            ))

    return chunks


def _chunk_definition_clauses(
    sec_content: str,
    base_title: str,
    sec_num: str,
    sec_heading: str,
    start_index: int,
) -> list[TextChunk]:
    """
    Splits Section 2 definitions by clause (a), (b), (c) ... (n) so each key legal term
    becomes a dedicated, highly retrievable semantic chunk.
    """
    clauses = list(DEF_CLAUSE_PATTERN.finditer(sec_content))
    if len(clauses) < 2:
        return []

    chunks: list[TextChunk] = []
    
    # Header intro (e.g. '2. Definitions. — In this Act, unless the context otherwise requires, —')
    intro_text = sec_content[:clauses[0].start()].strip()
    if intro_text:
        chunks.append(TextChunk(
            chunk_index=start_index + len(chunks),
            content=f"[{base_title} - Preamble]\n{intro_text}",
            section_title=f"{base_title} - Preamble",
            metadata={
                "section_number": sec_num,
                "heading": f"{sec_heading} (Intro)",
                "is_toc": False,
            }
        ))

    for idx, c in enumerate(clauses):
        cl_letter = c.group(1)
        cl_title = c.group(2).strip().strip("“”\"'‘’—–-")
        start_p = c.start()
        end_p = clauses[idx + 1].start() if idx + 1 < len(clauses) else len(sec_content)
        
        cl_body = sec_content[start_p:end_p].strip()
        cl_full_title = f"{base_title} - Clause ({cl_letter}): {cl_title}"

        chunks.append(TextChunk(
            chunk_index=start_index + len(chunks),
            content=f"[{cl_full_title}]\n{cl_body}",
            section_title=cl_full_title,
            metadata={
                "section_number": sec_num,
                "clause": cl_letter,
                "heading": f"Definitions ({cl_letter}) - {cl_title}",
                "is_toc": False,
                "word_count": len(cl_body.split()),
                "char_length": len(cl_body),
            }
        ))

    return chunks


def _split_sliding_window(
    words: list[str],
    max_words: int,
    overlap: int,
    prefix: str = "",
) -> list[str]:
    """Splits a list of words into overlapping windows of at most max_words."""
    windows: list[str] = []
    start = 0
    stride = max(1, max_words - overlap)

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        text_body = " ".join(chunk_words)
        windows.append(f"{prefix}{text_body}" if prefix else text_body)
        if end >= len(words):
            break
        start += stride

    return windows
