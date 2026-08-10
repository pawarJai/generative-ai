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
    toc = state.FILE_META.get(file_id, {}).get("toc", [])
    toc_text = "\n".join(f"- {t['text']} (p.{t['page']})" for t in toc[:60]) or "(no headings detected)"
    if state.FILE_KIND.get(file_id) == "tabular":
        prompt = f"This is a spreadsheet with:\n{toc_text}\nWrite a 3-4 sentence overview of what it contains."
    else:
        doc = state.DOCLING_DOCS[file_id]
        first_page = ""
        try:
            first_page = doc.export_to_markdown(page_no=1)[:2000]
        except Exception:
            pass
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
    except Exception as e:
        state.FILE_META[file_id]["summary"] = f"(overview unavailable: {e})"
