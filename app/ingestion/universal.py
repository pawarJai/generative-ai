"""Single entry point for ingesting any file — routes by extension."""
import os
import time
from app import state
from app.ingestion.docling_ingest import ingest_docling
from app.ingestion.tabular_ingest import ingest_tabular
from app.ingestion.text_ingest import ingest_plain_text
from app.summary import build_summary
from app.tables.helpers import get_all_real_tables
from app.logging_utils import log_ingestion

_TABULAR_EXTS = {".xlsx", ".xls", ".csv"}
_TEXT_EXTS = {".txt"}


def universal_ingest(path: str, file_id: str, force: bool = False) -> dict:
    """Routes by extension:
    - XLSX/XLS/CSV → tabular (DuckDB-queryable)
    - TXT           → plain text (vector-search only)
    - everything else → Docling (PDF/PNG/DOCX/PPTX/HTML)
    Returns a small summary dict for the API layer."""
    t0 = time.time()
    ext = os.path.splitext(path)[1].lower()
    if ext in _TABULAR_EXTS:
        kind = "tabular"
    elif ext in _TEXT_EXTS:
        kind = "text"
    else:
        kind = "docling"
    try:
        if kind == "tabular":
            ingest_tabular(path, file_id)
        elif kind == "text":
            ingest_plain_text(path, file_id)
        else:
            ingest_docling(path, file_id, force=force)

        # Invalidate the SQL agent's per-file DuckDB cache so the next query
        # rebuilds against the freshly ingested tables.
        try:
            from app.query.sql_agent import invalidate_cache
            invalidate_cache(file_id)
        except Exception:
            pass  # sql_agent may not be imported yet at startup

        state.set_active_file_id(file_id)
        if file_id not in state.FILE_ORDER:
            state.FILE_ORDER.append(file_id)
        state.FILE_ORIGINAL_NAME[file_id] = os.path.basename(path)
        build_summary(file_id)

        n_tables = len(get_all_real_tables(file_id))
        latency = time.time() - t0
        log_ingestion(file_id, path, kind, success=True, latency=latency, tables_found=n_tables)
        return {
            "file_id": file_id,
            "original_filename": os.path.basename(path),
            "kind": kind,
            "tables_found": n_tables,
            "ocr_pages": len(state.FILE_META.get(file_id, {}).get("ocr_pages", [])),
            "latency_sec": round(latency, 2),
        }
    except Exception as e:
        log_ingestion(file_id, path, kind, success=False, error=str(e), latency=time.time() - t0)
        raise
