"""Schema-driven table export: direct column-match first (deterministic,
can't misalign), LLM-reshape fallback with self-check + retry."""
import io
import re
from typing import List, Dict, Any, Optional
import pandas as pd
from app.config import llm
from app.models import QueryPlan
from app.tables.helpers import get_tables_for_scope
from app.export.text_export import clean_llm_csv


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def schema_map_export(file_id: str, plan: QueryPlan) -> pd.DataFrame:
    """Reshapes source rows into exactly the requested target headers via
    batched, grounded LLM calls (small batches — large ones reliably get cut
    off mid-output for verbose real-world text)."""
    file_ids = plan.file_scope or [file_id]
    tables = get_tables_for_scope(file_ids)
    if not tables:
        return pd.DataFrame(columns=plan.columns)

    BATCH_ROWS = 12

    def _trim(v, limit=300):
        s = str(v)
        return s[:limit] + "..." if len(s) > limit else s

    all_rows = []
    for df in tables:
        src = df.attrs.get("source_file", file_ids[0] if file_ids else "?")
        for _, row in df.iterrows():
            all_rows.append((src, {k: _trim(v) for k, v in row.items()}))

    if not all_rows:
        return pd.DataFrame(columns=plan.columns)

    sys = (
        f"Reshape the SOURCE ROWS below into a table with EXACTLY these headers, "
        f"in this order: {plan.columns}\n\n"
        "Rules:\n"
        "- Output ONLY CSV. No explanation, no markdown fences.\n"
        "- Map each target column from the best-matching source field when one "
        "clearly corresponds.\n"
        "- If no matching source data exists for a cell, leave it BLANK. Never "
        "invent, estimate, or guess a value that isn't actually present.\n"
        "- Output exactly one row per SOURCE ROW given — do not skip any."
    )

    results = []
    for i in range(0, len(all_rows), BATCH_ROWS):
        batch = all_rows[i:i + BATCH_ROWS]
        batch_text = "\n".join(
            f"[from {src}] " + ", ".join(f"{k}={v}" for k, v in rec.items())
            for src, rec in batch
        )
        try:
            raw = llm.invoke(f"{sys}\n\nSOURCE ROWS:\n{batch_text}").content
            df_batch = pd.read_csv(io.StringIO(clean_llm_csv(raw)))
            results.append(df_batch)
        except Exception as e:
            print(f"  (batch failed, skipping {len(batch)} rows: {e})")

    if not results:
        raise RuntimeError("Every batch failed — no rows could be reshaped.")

    combined = pd.concat(results, ignore_index=True, sort=False)
    for c in plan.columns:
        if c not in combined.columns:
            combined[c] = ""
    return combined[plan.columns]


def best_matching_table(tables: List[pd.DataFrame], target_columns: List[str],
                         threshold: float = 0.6) -> Optional[pd.DataFrame]:
    """If a source table already covers most of the target schema by column
    name, use it DIRECTLY — deterministic pandas extraction can't misalign a
    column the way free-text generation can."""
    target_norm = {_norm(c) for c in target_columns}
    best, best_score = None, 0.0
    for df in tables:
        overlap = len(target_norm & {_norm(c) for c in df.columns}) / len(target_norm)
        if overlap > best_score:
            best, best_score = df, overlap
    if best is not None and best_score >= threshold:
        return best
    return None


def validate_export(df: pd.DataFrame, target_columns: List[str]) -> Dict[str, Any]:
    """Self-check: does this actually look like real data, not just 'parsed
    without a python exception'?"""
    if df is None or df.empty:
        return {"ok": False, "problems": ["no rows produced"], "fill_rate": 0.0}

    problems = []
    missing = [c for c in target_columns if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")

    as_str = df.fillna("").astype(str)
    total_cells = df.shape[0] * df.shape[1]
    non_blank = (as_str.apply(lambda col: col.str.strip().ne("") & col.str.strip().str.lower().ne("nan"))).sum().sum()
    fill_rate = non_blank / total_cells if total_cells else 0.0
    is_blank_row = as_str.apply(lambda row: all(v.strip() == "" or v.strip().lower() == "nan" for v in row), axis=1)
    blank_row_rate = is_blank_row.mean()

    if fill_rate < 0.15:
        problems.append(f"suspiciously low fill rate ({fill_rate:.0%})")
    if blank_row_rate > 0.4:
        problems.append(f"{blank_row_rate:.0%} of rows are entirely blank")
    if len(df) > 3 and df.duplicated().mean() > 0.8:
        problems.append("most rows are exact duplicates")

    return {"ok": len(problems) == 0, "problems": problems, "fill_rate": fill_rate,
            "blank_row_rate": blank_row_rate}


def validation_note(tables: List[pd.DataFrame]) -> str:
    v = tables[0].attrs.get("validation") if tables else None
    if not v:
        return ""
    if v.get("ok"):
        return f" [self-check passed via {v['method']}, fill_rate={v.get('fill_rate', 0):.0%}]"
    return (f" [WARNING: self-check found issues after {len(v.get('attempts', []))} "
            f"attempt(s): {v['problems']} — inspect before relying on this]")


def prep_tables(file_id: str, plan: QueryPlan) -> List[pd.DataFrame]:
    """Long explicit target-column list -> reshape with retry loop:
    direct-match first, LLM-reshape fallback, honestly flag if unverified."""
    if plan.columns and len(plan.columns) >= 4:
        file_ids = plan.file_scope or [file_id]
        tables_in_scope = get_tables_for_scope(file_ids)
        attempts = []

        direct = best_matching_table(tables_in_scope, plan.columns)
        if direct is not None:
            renamed = direct.rename(columns={c: t for c in direct.columns for t in plan.columns
                                              if _norm(c) == _norm(t)})
            for c in plan.columns:
                if c not in renamed.columns:
                    renamed[c] = ""
            result = renamed[plan.columns]
            check = validate_export(result, plan.columns)
            attempts.append(("direct_match", check))
            if check["ok"]:
                result.attrs["validation"] = {"method": "direct_match", "attempts": attempts, **check}
                return [result]

        MAX_LLM_ATTEMPTS = 2
        result, check = None, None
        for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
            result = schema_map_export(file_id, plan)
            check = validate_export(result, plan.columns)
            attempts.append((f"llm_reshape_try{attempt}", check))
            if check["ok"]:
                result.attrs["validation"] = {"method": f"llm_reshape (attempt {attempt})",
                                               "attempts": attempts, **check}
                return [result]

        result.attrs["validation"] = {"method": "llm_reshape (exhausted retries, unverified)",
                                       "attempts": attempts, **check}
        return [result]

    file_ids = plan.file_scope or [file_id]
    tables = get_tables_for_scope(file_ids)
    rename_map = {_norm(k): v for k, v in (plan.rename or {}).items()}
    targets = {_norm(k) for k in rename_map} | {_norm(c) for c in (plan.columns or [])}

    prepared = []
    for df in tables:
        if targets and not (targets & {_norm(c) for c in df.columns}):
            continue
        renamed = {c: rename_map[_norm(c)] for c in df.columns if _norm(c) in rename_map}
        d = df.rename(columns=renamed)
        if plan.columns:
            keep = [c for c in d.columns if _norm(c) in {_norm(x) for x in plan.columns}]
            if keep:
                d = d[keep]
        prepared.append(d)
    return prepared
