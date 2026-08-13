"""PDF/image/docx/pptx ingestion via Docling, with per-page OCR fallback for
scanned pages Docling extracted near-zero text from."""
import os
import hashlib
from typing import Optional
from app import state
from app.config import CACHE_DIR, _converter
from app.embedding import embed_text
from app.summary import extract_toc
from app.tables.helpers import dedupe_columns
from app.ingestion.ocr import page_text_is_thin, ocr_page, ocr_words_to_table


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
    state.FILE_META[file_id] = {"toc": extract_toc(doc), "ocr_pages": [], "content_hash": h}

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

    embed_text(file_id, "\n\n".join(full_text_parts))
