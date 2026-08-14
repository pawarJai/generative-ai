"""XLSX/CSV ingestion: merged-cell unfilling -> multi-block splitting ->
per-block header detection. Fixes the two root causes of near-empty tables
on real-world messy spreadsheets: (1) pd.read_excel reads a merged cell's
value only into the top-left cell, NaN elsewhere; (2) it always assumes
row 0 is the header, which is wrong for sheets with a title/logo block above
the real header row."""
from typing import List, Optional, Tuple
import pandas as pd
import openpyxl
from app import state
from app.tables.helpers import dedupe_columns
from app.embedding import embed_text


def _df_to_text(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        vals = [str(v).strip() for v in row if pd.notna(v) and str(v).strip()]
        if vals:
            lines.append(" | ".join(vals))
    return "\n".join(lines)


def _clean_raw_sheet(df: pd.DataFrame, sheet_name: str) -> Optional[pd.DataFrame]:
    raw_cleaned = df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)
    if raw_cleaned.empty:
        return None
    cols = [f"col_{i}" for i in range(len(raw_cleaned.columns))]
    raw_cleaned.columns = dedupe_columns(cols)
    raw_cleaned.attrs["page"] = f"{sheet_name}_Raw"
    return raw_cleaned


def _unmerge_and_fill(path: str, sheet_name: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet_name]
    data = [[cell.value for cell in row] for row in ws.iter_rows()]
    df = pd.DataFrame(data)
    for merge in ws.merged_cells.ranges:
        val = df.iat[merge.min_row - 1, merge.min_col - 1]
        for r in range(merge.min_row - 1, merge.max_row):
            for c in range(merge.min_col - 1, merge.max_col):
                df.iat[r, c] = val
    return df


def _score_header_row(row: pd.Series) -> float:
    vals = [str(v).strip() for v in row if pd.notna(v)]
    if not vals:
        return -1.0
    non_null_frac = len(vals) / len(row)
    text_frac = sum(1 for v in vals if not v.replace(".", "", 1).replace("-", "", 1).isdigit()) / len(vals)
    unique_frac = len(set(vals)) / len(vals)
    avg_len = sum(len(v) for v in vals) / len(vals)
    return non_null_frac * 0.4 + text_frac * 0.3 + unique_frac * 0.2 + (0.1 if avg_len < 40 else 0.0)


def _detect_header_row(raw_df: pd.DataFrame, max_scan: int = 15) -> int:
    best_idx, best_score = 0, -1.0
    for i in range(min(max_scan, len(raw_df))):
        row = raw_df.iloc[i]
        if row.notna().sum() < 2:
            continue
        score = _score_header_row(row)
        if i + 1 < len(raw_df):
            next_fill = raw_df.iloc[i + 1].notna().mean()
            if next_fill > row.notna().mean() * 0.5:
                score += 0.15
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx


def _split_blocks(df: pd.DataFrame, min_blank_run: int = 2) -> List[pd.DataFrame]:
    is_blank = df.isna().all(axis=1)
    blocks, start, blank_run = [], None, 0
    for i, blank in enumerate(is_blank):
        if blank:
            blank_run += 1
            if start is not None and blank_run >= min_blank_run:
                blocks.append(df.iloc[start:i - blank_run + 1])
                start = None
        else:
            if start is None:
                start = i
            blank_run = 0
    if start is not None:
        blocks.append(df.iloc[start:])
    return [b for b in blocks if len(b) >= 2]


def _clean_block_to_table(block: pd.DataFrame) -> Optional[pd.DataFrame]:
    block = block.reset_index(drop=True)
    header_idx = _detect_header_row(block)
    header = block.iloc[header_idx]
    data = block.iloc[header_idx + 1:].dropna(axis=1, how="all").dropna(axis=0, how="all")
    if data.empty:
        return None
    cols = [str(header[c]).strip() if pd.notna(header.get(c)) else f"col_{c}" for c in data.columns]
    data.columns = dedupe_columns(cols)
    return data.reset_index(drop=True)


def _ingest_xls_legacy(path: str) -> Tuple[List[pd.DataFrame], List[str], List[str]]:
    """Read old-format .xls files via xlrd (openpyxl can't handle them)."""
    sheets = pd.read_excel(path, sheet_name=None, engine="xlrd", header=None)
    tables = []
    all_text = []
    sheet_names = list(sheets.keys())
    for sheet_name, raw_df in sheets.items():
        raw_df = raw_df.reset_index(drop=True)
        
        sheet_text = _df_to_text(raw_df)
        if sheet_text:
            all_text.append(f"--- Sheet: {sheet_name} ---\n{sheet_text}")
            
        raw_clean = _clean_raw_sheet(raw_df, sheet_name)
        if raw_clean is not None:
            tables.append(raw_clean)

        blocks = _split_blocks(raw_df)
        for bi, block in enumerate(blocks):
            cleaned = _clean_block_to_table(block)
            if cleaned is not None and cleaned.shape[1] >= 2:
                label = sheet_name if len(blocks) == 1 else f"{sheet_name} (block {bi + 1})"
                cleaned.attrs["page"] = label
                tables.append(cleaned)
    return tables, all_text, sheet_names


def ingest_tabular(path: str, file_id: str) -> None:
    import os
    ext = os.path.splitext(path)[1].lower()
    tables = []
    all_text = []
    # The workbook's OWN sheet names, in workbook order — recorded separately
    # from the extracted tables because they are not recoverable from them.
    # One sheet becomes several tables (a "_Raw" whole-sheet dump plus a
    # "(block N)" per detected sub-table), and a sheet holding nothing
    # tabular becomes none at all. Asked "how many sheets are there", the
    # only honest source is this list: counting tables answered 17 for a
    # 10-sheet workbook, and dropped 'Delivery' (no extractable table)
    # entirely. See app.graph.tools.list_sheets.
    sheet_names: List[str] = []

    if ext == ".xls":
        tables, all_text, sheet_names = _ingest_xls_legacy(path)
    elif ext == ".xlsx":
        sheet_names = list(pd.ExcelFile(path).sheet_names)
        for sheet_name in sheet_names:
            raw = _unmerge_and_fill(path, sheet_name)
            
            sheet_text = _df_to_text(raw)
            if sheet_text:
                all_text.append(f"--- Sheet: {sheet_name} ---\n{sheet_text}")
                
            raw_clean = _clean_raw_sheet(raw, sheet_name)
            if raw_clean is not None:
                tables.append(raw_clean)

            blocks = _split_blocks(raw)
            for bi, block in enumerate(blocks):
                cleaned = _clean_block_to_table(block)
                if cleaned is not None and cleaned.shape[1] >= 2:
                    label = sheet_name if len(blocks) == 1 else f"{sheet_name} (block {bi + 1})"
                    cleaned.attrs["page"] = label
                    tables.append(cleaned)
    else:
        df = pd.read_csv(path)
        sheet_text = _df_to_text(df)
        if sheet_text:
            all_text.append(f"--- CSV File ---\n{sheet_text}")
            
        df.columns = dedupe_columns(df.columns)
        df.attrs["page"] = "data"
        tables.append(df)
        sheet_names = ["data"]

    state.TABULAR_TABLES[file_id] = tables
    state.FILE_KIND[file_id] = "tabular"
    state.FILE_META[file_id] = {
        "toc": [{"text": f"Sheet/Block: {t.attrs['page']}", "page": t.attrs["page"]}
                for t in tables],
        "sheets": sheet_names,
    }

    # Generate description for embedding
    desc_parts = []
    if tables:
        desc_parts.append("Data Tables:\n" + "\n\n".join(
            f"'{t.attrs['page']}' columns: {list(t.columns)}\n{t.head(5).to_string(index=False)}"
            for t in tables if not str(t.attrs['page']).endswith("_Raw")))
    
    if all_text:
        desc_parts.append("Full Text Content:\n" + "\n\n".join(all_text))
        
    if not desc_parts:
        desc_parts.append(f"Tabular file with no readable content: {os.path.basename(path)}")

    embed_text(file_id, "\n\n========================\n\n".join(desc_parts))
