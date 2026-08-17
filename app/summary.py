"""Overview generation — one LLM call per file, cached in FILE_META."""
from typing import List, Dict, Any
from app import state
from app.config import llm


def looks_malformed(text: str) -> bool:
    """Catches raw tool-call leakage / obviously broken model output before
    it gets cached as permanent file metadata."""
    if not text or len(text.strip()) < 20:
        return True
    if "<tool_call>" in text or "<|" in text:
        return True
    return False


def label_str(item) -> str:
    lbl = getattr(item, "label", None)
    return getattr(lbl, "value", str(lbl)) if lbl is not None else ""


def extract_toc(doc) -> List[Dict[str, Any]]:
    """Flat, reading-order heading list. Docling doesn't reliably expose
    heading LEVEL, so this is intentionally flat, not nested."""
    toc = []
    for item in getattr(doc, "texts", []):
        if label_str(item) in ("title", "section_header"):
            page = item.prov[0].page_no if getattr(item, "prov", None) else None
            text = (getattr(item, "text", "") or "").strip()
            if text:
                toc.append({"text": text, "page": page})
    return toc


def build_summary(file_id: str) -> None:
    # Re-uploading identical content under a new file_id (common when a
    # frontend re-uploads on every page load) previously re-ran this LLM
    # call every time even though the answer can't have changed.
    content_hash = state.FILE_META.get(file_id, {}).get("content_hash")
    if content_hash and content_hash in state.SUMMARY_CACHE:
        state.FILE_META[file_id]["summary"] = state.SUMMARY_CACHE[content_hash]
        return

    toc = state.FILE_META.get(file_id, {}).get("toc", [])
    toc_text = "\n".join(f"- {t['text']} (p.{t['page']})" for t in toc[:60]) or "(no headings detected)"
    if state.FILE_KIND.get(file_id) == "tabular":
        # Same rule for spreadsheets: describe the sheets that exist, and
        # say so plainly when there are none.
        if not toc:
            state.FILE_META[file_id]["summary"] = (
                "This spreadsheet has no readable sheets or tables — nothing "
                "was extracted from it, so there is nothing to describe.")
            return
        sheets = state.FILE_META.get(file_id, {}).get("sheets") or []
        prompt = (f"This is a spreadsheet. Its sheets are: {list(sheets)}\n"
                  f"Its extracted tables are:\n{toc_text}\n"
                  f"Write a 3-4 sentence overview of what it contains. Describe "
                  f"ONLY the sheets and tables listed above — do not guess at "
                  f"columns, subject matter or purpose that is not shown.")
    else:
        doc = state.DOCLING_DOCS[file_id]
        first_page = ""
        try:
            first_page = doc.export_to_markdown(page_no=1)[:2000]
        except Exception:
            pass

        # Nothing to summarise is not the same as "summarise nothing". Given
        # no headings and no first-page text, the model happily produced
        # "a technical specification or standard, likely related to a system
        # or process involving data handling and configuration" — for a blank
        # image. A file overview is exactly where an invented description
        # does the most damage, because it gets cached as this file's
        # permanent metadata and every later answer builds on it.
        if not toc and not first_page.strip():
            pages = len(getattr(doc, "pages", {}) or {})
            pics = len(getattr(doc, "pictures", []) or [])
            tables = len(getattr(doc, "tables", []) or [])
            state.FILE_META[file_id]["summary"] = (
                f"No text could be extracted from this file "
                f"({pages} page(s), {pics} image(s), {tables} table(s)). "
                f"It is most likely a scanned or purely graphical document. "
                f"Nothing is being described here because nothing was read — "
                f"ask about a specific page to see what OCR recovered.")
            return

        prompt = (f"Document heading list (flat, in reading order):\n{toc_text}\n\n"
                  f"First page content:\n{first_page}\n\n"
                  f"Write a 4-6 sentence overview of what this document is and how it's "
                  f"organized. Base it only on the material shown; don't invent chapter "
                  f"numbers not listed above.")
    try:
        summary = llm.invoke(prompt).content
        if looks_malformed(summary):
            summary = llm.invoke(prompt).content
        if looks_malformed(summary):
            summary = ("Overview generation produced malformed output twice. "
                       f"Try re-running ingestion for '{file_id}' or check the model config.")
        state.FILE_META[file_id]["summary"] = summary
        if content_hash and not looks_malformed(summary):
            state.SUMMARY_CACHE[content_hash] = summary
    except Exception as e:
        state.FILE_META[file_id]["summary"] = f"(overview unavailable: {e})"
