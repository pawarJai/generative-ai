"""Semantic fallback — the last resort before telling a user we found nothing.

Chroma is on disk and survives every restart (24,067 chunks at the time this
was written), while app/state.py does not. So whenever an exact lookup dead-ends
— a page with no separately extractable content, a table query that matched no
rows — the indexed text for that file is still there and worth searching before
the answer degrades into "your file appears to be empty or corrupted".

Kept in its own module so both the tools layer and the agent backstop use one
implementation, per the project's one-responsibility-per-file rule.
"""
import re
from typing import Optional
from app.config import vector_db


def _has_signal(chunk: str, min_words: int = 5) -> bool:
    """Whether a chunk carries real text rather than table scaffolding."""
    if not chunk:
        return False
    words = [w for w in re.split(r"\W+", chunk) if len(w) > 1 and not w.isdigit()]
    return len(words) >= min_words


def semantic_fallback(file_id: str, query: str, k: int = 5,
                      max_chars: int = 2500) -> Optional[str]:
    """Top matching indexed chunks for one file, or None if there are none.

    Scoped to a single file_id on purpose: a fallback that silently pulled
    content from a *different* document would be worse than no answer.
    """
    if not file_id or not query or not query.strip():
        return None
    # Over-fetch. In table-heavy documents the nearest neighbours are often
    # runs of markdown pipe separators, and filtering those out of a plain
    # top-k left nothing at all — turning a file with 174 indexed chunks into
    # "no similar content found".
    try:
        results = vector_db.similarity_search(query, k=k * 5, filter={"file_id": file_id})
    except Exception as e:
        print(f"[semantic_fallback] search failed for {file_id}: {e}")
        return None
    if not results:
        return None

    parts, total = [], 0
    for r in results:
        if len(parts) >= k:
            break
        chunk = (r.page_content or "").strip()
        if not _has_signal(chunk):
            # Markdown tables chunk into runs of "|" and "---" separators that
            # match almost any query and carry no information. Returning them
            # looks like an answer while saying nothing.
            continue
        if total + len(chunk) > max_chars:
            chunk = chunk[: max_chars - total]
        parts.append(chunk)
        total += len(chunk)
        if total >= max_chars:
            break
    return "\n\n---\n\n".join(parts) if parts else None


def has_indexed_content(file_id: str) -> bool:
    """Whether this file has any chunks in the vector store at all — the
    evidence needed to contradict a claim that a file is empty."""
    try:
        got = vector_db.get(where={"file_id": file_id}, limit=1)
        return bool(got and got.get("ids"))
    except Exception:
        return False
