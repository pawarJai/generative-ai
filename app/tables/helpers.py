"""
Table access layer — every other module reads tables through these functions,
never by touching app.state.DOCLING_DOCS / TABULAR_TABLES directly. Keeping
one access point means a future storage-backend swap (e.g. moving cached
tables to Postgres) only requires changing this file.
"""
import re
from typing import List, Dict, Optional
import pandas as pd
from app import state


def dedupe_columns(cols) -> List[str]:
    """Guarantees unique column names per table — required before any
    pd.concat, since duplicate/blank headers (common in real PDFs/xlsx)
    otherwise crash with InvalidIndexError."""
    seen: Dict[str, int] = {}
    out = []
    for i, c in enumerate(cols):
        name = str(c).strip() or f"col_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def looks_like_real_table(df: pd.DataFrame) -> bool:
    """Content-based junk filter for Docling's occasional false-positive
    'table' detections on glossary/index pages (a column of sorted single
    words, no numbers)."""
    sample = df.astype(str).values.flatten()
    if len(sample) == 0:
        return False
    single_alpha_words = sum(1 for v in sample if v.strip().isalpha() and " " not in v.strip())
    has_digit = any(any(ch.isdigit() for ch in v) for v in sample)
    if single_alpha_words / len(sample) > 0.8 and not has_digit:
        return False
    return True


def get_page_markdown(file_id: str, page_no: int) -> str:
    doc = state.DOCLING_DOCS[file_id]
    return doc.export_to_markdown(page_no=page_no)


def get_tables_on_page(file_id: str, page_no: int) -> List[pd.DataFrame]:
    doc = state.DOCLING_DOCS[file_id]
    out = []
    for t in doc.tables:
        if t.prov and t.prov[0].page_no == page_no:
            out.append(t.export_to_dataframe())
    return out


def get_all_real_tables(file_id: str, min_cols: int = 3, min_rows: int = 1) -> List[pd.DataFrame]:
    """All real tables for a file — native Docling tables + OCR-derived +
    directly-loaded tabular tables, whichever apply."""
    if state.FILE_KIND.get(file_id) == "tabular":
        return state.TABULAR_TABLES.get(file_id, [])

    doc = state.DOCLING_DOCS[file_id]
    dfs = []
    for t in doc.tables:
        df = t.export_to_dataframe(doc)
        df.columns = dedupe_columns(df.columns)
        if df.shape[1] >= min_cols and df.shape[0] >= min_rows and looks_like_real_table(df):
            page = t.prov[0].page_no if t.prov else None
            df.attrs["page"] = page
            dfs.append(df)

    for df in state.TABULAR_TABLES.get(file_id, []):
        if df.shape[1] >= min_cols and df.shape[0] >= min_rows:
            dfs.append(df)
    return dfs


def get_tables_for_scope(file_ids: List[str]) -> List[pd.DataFrame]:
    """Tables from several files at once, each tagged with its source file_id."""
    all_tables = []
    for fid in file_ids:
        for df in get_all_real_tables(fid):
            d = df.copy()
            d.attrs["source_file"] = fid
            all_tables.append(d)
    return all_tables


def resolve_file_scope(prompt: str) -> List[str]:
    """Deterministically parses 'last file' / 'the excel file' / 'all files'
    into concrete file_ids — never left to LLM guessing."""
    p = prompt.lower()
    if not state.FILE_ORDER:
        return []
    if re.search(r"\b(all files?|every file|across (all|every) (file|document)s?|combine all)\b", p):
        return list(state.FILE_ORDER)
    if re.search(r"\b(last|latest|most recent|newest) (file|upload|document|pdf)\b", p):
        return [state.FILE_ORDER[-1]]
    if re.search(r"\b(first|earliest) (file|upload|document|pdf)\b", p):
        return [state.FILE_ORDER[0]]
    # Collect EVERY filename mentioned, not just the first match — a prompt like
    # "combine file-A and file-B" previously returned only file-A because this
    # used to `return` on the first hit.
    named = [fid for fid in state.FILE_ORDER
             if state.FILE_ORIGINAL_NAME.get(fid, "") and
             state.FILE_ORIGINAL_NAME.get(fid, "").lower() in p]
    if named:
        return named
    ext_words = {"excel": (".xlsx", ".xls"), "csv": (".csv",), "pdf": (".pdf",),
                 "powerpoint": (".pptx",), "image": (".png", ".jpg", ".jpeg")}
    for kw, exts in ext_words.items():
        if kw in p:
            matches = [fid for fid in state.FILE_ORDER
                       if state.FILE_ORIGINAL_NAME.get(fid, "").lower().endswith(exts)]
            if matches:
                return matches
    return []
