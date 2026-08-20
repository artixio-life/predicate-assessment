"""
Token-aware chunking for drug.products.document_text -> drug.product_chunks.

Chunks are consumed in order by the AI extraction stage's fold: each chunk
is a unit of "new evidence" shown to the LLM alongside the JSON accumulated
from prior chunks. Paragraph boundaries are preserved where possible so a
chunk boundary doesn't split a table row or a sentence mid-thought; a small
overlap keeps continuity across the cut.
"""
import re
from typing import List

import os

# Chunk size in tokens. Lowering it makes each excerpt smaller, which is the
# lever for a small-context model: the fold's per-call prompt is the system
# prompt + accumulator + CHUNKS_PER_CALL excerpts, so halving this roughly
# halves the excerpt half of that budget.
DEFAULT_TARGET_TOKENS = int(os.getenv("CHUNK_TARGET_TOKENS", "1800"))
# Fraction of the previous chunk prefixed onto each chunk, so a fact split
# across a boundary is still readable in one of them.
DEFAULT_OVERLAP_RATIO = float(os.getenv("CHUNK_OVERLAP_RATIO", "0.15"))

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODING = None


def _count_tokens(text: str) -> int:
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    # Conservative chars-per-token fallback for regulatory prose.
    return max(1, len(text) // 4)


def count_tokens(text: str) -> int:
    """Public wrapper around `_count_tokens`, for callers outside this module
    that need to size their own content the same way (e.g. ai_extraction.py
    packing json_data's list-valued fields into chunk-sized groups)."""
    return _count_tokens(text)


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts or ([text] if text else [])


def _split_oversized_paragraph(paragraph: str, target_tokens: int) -> List[str]:
    """A single paragraph bigger than target_tokens (e.g. a giant table dump)
    gets split on sentence boundaries instead of being force-fit whole."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces, current = [], []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence)
        if current and current_tokens + sentence_tokens > target_tokens:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(text: str, target_tokens: int = DEFAULT_TARGET_TOKENS,
               overlap_ratio: float = DEFAULT_OVERLAP_RATIO) -> List[str]:
    """
    Split `text` into chunks of roughly `target_tokens` tokens, packing whole
    paragraphs greedily and only splitting a paragraph internally if it alone
    exceeds the target. Each chunk after the first is prefixed with the tail
    of the previous chunk (~overlap_ratio of it) so context isn't lost at the
    cut. Returns [] for empty input, or a single-element list if the whole
    document already fits in one chunk.
    """
    normalized = _normalize(text or "")
    if not normalized:
        return []

    if _count_tokens(normalized) <= target_tokens:
        return [normalized]

    units: List[str] = []
    for paragraph in _split_paragraphs(normalized):
        if _count_tokens(paragraph) > target_tokens:
            units.extend(_split_oversized_paragraph(paragraph, target_tokens))
        else:
            units.append(paragraph)

    chunks: List[str] = []
    current_units: List[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = _count_tokens(unit)
        if current_units and current_tokens + unit_tokens > target_tokens:
            chunks.append("\n\n".join(current_units))
            current_units, current_tokens = [], 0
        current_units.append(unit)
        current_tokens += unit_tokens
    if current_units:
        chunks.append("\n\n".join(current_units))

    if len(chunks) <= 1 or overlap_ratio <= 0:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        overlap_tokens = max(1, int(_count_tokens(prev) * overlap_ratio))
        if _ENCODING is not None:
            prev_token_ids = _ENCODING.encode(prev)
            tail = _ENCODING.decode(prev_token_ids[-overlap_tokens:])
        else:
            tail = prev[-overlap_tokens * 4:]
        overlapped.append(f"{tail.strip()}\n\n{chunks[i]}")

    return overlapped
