"""
Text-to-SQL query engine — simple, fast, reliable.
Single-shot LLM SQL generation + DuckDB execution.
"""
import re
from typing import List, Optional, Tuple
import duckdb
import pandas as pd
from app.config import llm
from app.tables.helpers import get_all_real_tables

WIDE_TABLE_COLUMN_THRESHOLD = 12


def format_as_record_cards(df: pd.DataFrame) -> str:
    cards = []
    for _, row in df.iterrows():
        lines = [f"  {col}: {val}" for col, val in row.items() if pd.notna(val)]
        cards.append("---\n" + "\n".join(lines))
    return "\n".join(cards)


def _register_tables(file_id: str) -> Tuple[duckdb.DuckDBPyConnection, List[str]]:
    con = duckdb.connect(database=":memory:")
    tables = get_all_real_tables(file_id)
    schema_lines = []
    for i, df in enumerate(tables):
        raw_name = str(df.attrs.get("page", f"table_{i}"))
        tname = re.sub(r"\W+", "_", raw_name).strip("_") or f"table_{i}"
        base_tname, n = tname, 1
        existing = {row[0] for row in con.execute(
            "select table_name from information_schema.tables").fetchall()}
        while tname in existing:
            n += 1
            tname = f"{base_tname}_{n}"
        con.register(tname, df)
        schema_lines.append(f"{tname}({', '.join(str(c) for c in df.columns)})")
    return con, schema_lines


def run_data_query(file_id: str, prompt: str, history_messages=None) -> dict:
    """Returns {"text": str, "table": {"columns": [...], "rows": [[...]]} | None, "sql": str | None}."""
    con, schema_lines = _register_tables(file_id)
    if not schema_lines:
        return {"text": "No tables found for this file.", "table": None, "sql": None}

    schema_text = "\n".join(schema_lines)
    sys = (
        f"You have access to these DuckDB tables:\n{schema_text}\n\n"
        f"Write ONE DuckDB SQL query that answers the user's question.\n"
        f"Rules:\n"
        f"- Output ONLY the SQL query, no explanation, no markdown fences, no LIMIT unless user asked.\n"
        f"- Use exact table/column names from the schema above.\n"
        f"- If the question asks for a sample of rows, use LIMIT (default 5 if unspecified).\n"
        f"- If the question asks for a specific value/row, use WHERE with the exact match.\n"
        f"- If counting, use COUNT(*). Quote column names with spaces using double quotes.\n"
        f"- Always fetch the FULL result, not a sample, unless explicitly asked for a sample."
    )
    sql = llm.invoke(f"{sys}\n\nQUESTION: {prompt}").content
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql.strip())

    try:
        result_df = con.execute(sql).fetchdf()
    except Exception as e:
        return {"text": f"Query failed ({e}). Generated SQL was:\n{sql}", "table": None, "sql": sql}

    if result_df.empty:
        return {
            "text": "Query ran successfully but returned no matching rows.",
            "table": None,
            "sql": sql,
        }

    if result_df.shape[1] > WIDE_TABLE_COLUMN_THRESHOLD:
        text = (f"Query result ({len(result_df)} row(s)):\n\n"
                f"{format_as_record_cards(result_df)}")
        return {"text": text, "table": None, "sql": sql}

    table = {
        "columns": list(result_df.columns),
        "rows": [[None if pd.isna(v) else v for v in row] for _, row in result_df.iterrows()],
    }
    return {
        "text": f"Query result ({len(result_df)} row(s), {len(result_df.columns)} column(s)).",
        "table": table,
        "sql": sql,
    }
