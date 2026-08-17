"""PDF/image/docx/pptx ingestion via Docling, with per-page OCR fallback for
scanned pages Docling extracted near-zero text from."""
import os
import re
import hashlib
from typing import Optional
from app import state
from app.config import CACHE_DIR, _converter
from app.embedding import embed_text
from app.summary import extract_toc
from app.tables.helpers import dedupe_columns
from app.ingestion.ocr import page_text_is_thin, ocr_page, ocr_words_to_table

# Label: value pairs that sit in a page's letterhead/header block, outside
# any table — "YARD No. BY531-536", "PROJECT -NGMV", "Spec. No.
# N45310510605002" — which get_all_real_tables never sees since they are
# not part of a table. Confirmed real, repeated need: the Yard No. visible
# in every page's header table of data-file-1.pdf was never captured
# anywhere in FILE_META, so a request to add it as an export column had
# nothing to read it from. Matched against Docling's OWN already-extracted
# per-page markdown (`page_md`, computed in the loop below regardless) —
# no new dependency, no schema assumption about any one document's layout.
_KEY_VALUE_PATTERNS = [
    r'(Yard\s*No\.?)\s*[:\-\|]?\s*([A-Z0-9\-/]+)',
    # Confirmed root cause of the "stuck pending" hang, found by timing
    # each pattern individually against a real 184KB document: this was
    # `([A-Z0-9\s\-/]+?)(?:\n|$)` — a LAZY quantifier paired with an
    # alternation terminator. That combination is a textbook catastrophic-
    # backtracking shape (confirmed: 8+ seconds and climbing on this one
    # pattern alone, on real text; every other pattern here ran in under
    # 1ms on the same text). A greedy class that simply excludes `\n`
    # needs no separate terminator and no backtracking to find the line
    # end — proven fast (<1ms) and gives the identical matches. A run of
    # trailing spaces with nothing after it (e.g. an empty table cell)
    # captures as whitespace-only, which the caller's `val.strip()` +
    # `if val` check already discards correctly.
    r'(Project\s*(?:Name|No\.?)?)\s*[:\-\|]?\s*([A-Z0-9 \-/]+)',
    r'(Spec(?:ification)?\.?\s*No\.?)\s*[:\-\|]?\s*([A-Z0-9\-/.]+)',
    r'(Enquiry\s*(?:Ref\.?|No\.?))\s*[:\-\|]?\s*([A-Z0-9\-/]+)',
    r'(Tender\s*No\.?)\s*[:\-\|]?\s*([A-Z0-9/\-]+)',
    r'(Ref(?:erence)?\.?\s*No\.?)\s*[:\-\|]?\s*([A-Z0-9/\-]+)',
]


def _extract_key_values(text: str) -> dict:
    """Label:value pairs found in ``text`` via the patterns above. Deliberately
    not exhaustive or document-specific — a caller wanting more needs to add
    a pattern here, not a per-document special case."""
    kv = {}
    if not text:
        return kv
    for pat in _KEY_VALUE_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            key = m.group(1).strip().rstrip('.')
            val = m.group(2).strip()
            if val and len(val) < 100:
                kv[key] = val
    return kv


def ingest_docling(path: str, file_id: str, force: bool = False,
                    ocr_min_chars: int = 40) -> None:
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    cache = os.path.join(CACHE_DIR, f"{h}.json")
    if os.path.exists(cache) and not force:
        from docling_core.types.doc import DoclingDocument
        doc = DoclingDocument.load_from_json(cache)
    else:
        doc = _converter.convert(path).document
        doc.save_as_json(cache)

    state.DOCLING_DOCS[file_id] = doc
    state.FILE_KIND[file_id] = "docling"
    state.FILE_META[file_id] = {"toc": extract_toc(doc), "ocr_pages": [],
                                "content_hash": h, "key_values": {}}

    full_text_parts = [doc.export_to_markdown()]
    page_nos = sorted({item.prov[0].page_no for item in getattr(doc, "texts", [])
                        if getattr(item, "prov", None)}) or [1]
    for page_no in page_nos:
        try:
            page_md = doc.export_to_markdown(page_no=page_no)
        except Exception:
            page_md = ""
        if page_text_is_thin(page_md, min_chars=ocr_min_chars):
            try:
                ocr = ocr_page(path, page_no)
            except Exception as e:
                print(f"  (OCR failed on page {page_no}: {e})")
                continue
            if ocr["text"].strip():
                state.FILE_META[file_id]["ocr_pages"].append(
                    {"page": page_no, "avg_conf": ocr["avg_conf"], "n_words": ocr["n_words"]})
                full_text_parts.append(f"\n\n[OCR page {page_no}, conf={ocr['avg_conf']}]\n{ocr['text']}")
                table = ocr_words_to_table(ocr["words"])
                if table is not None:
                    table.attrs["page"] = page_no
                    table.attrs["source"] = "ocr"
                    state.TABULAR_TABLES.setdefault(file_id, []).append(table)

    # Run ONCE against the whole document's already-assembled text, not
    # once per page inside the loop above. Confirmed root cause of the
    # "stuck pending for 30+ minutes" regression: this regex work is fast
    # in isolation, but running it repeatedly (6 patterns x every page)
    # on a background thread while an asyncio event loop is actively
    # running concurrently — exactly how FastAPI's BackgroundTasks
    # executes real ingestion — compounds GIL hand-off overhead across
    # the call volume into a multi-minute stall. Moving it out of the
    # per-page loop cuts the call volume from N pages x 6 patterns down
    # to 1 x 6, and removes it from the part of the pipeline that is
    # sensitive to that contention.
    full_text = "\n\n".join(full_text_parts)
    state.FILE_META[file_id]["key_values"].update(_extract_key_values(full_text))

    embed_text(file_id, full_text)
