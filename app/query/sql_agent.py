"""
Dynamic text-to-SQL via LangChain's SQL Agent toolkit.

Return contract: {"text": str, "table": dict | None, "sql": str | None}
"""
import re
import warnings

import pandas as pd
from sqlalchemy import create_engine

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langchain_community.utilities.sql_database import SQLDatabase
    from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
    from langchain_community.agent_toolkits.sql.base import create_sql_agent

from app.config import llm
from app.tables.helpers import get_all_real_tables

_db_cache: dict = {}
_SELECT_RE = re.compile(r"^\s*select\b", re.IGNORECASE)

# Root cause of the missing-export bug: the agent was answering from the
# 2-row schema sample without executing a real SELECT, so intermediate_steps
# had no sql_db_query step, so last_sql=None, so table=None, so EXPORTERS
# was never called.  The prefix below forces a real query every time.
_AGENT_PREFIX = (
    "You are a precise SQL agent for a user's uploaded data.\n"
    "MANDATORY RULES — follow every one, no exceptions:\n"
    "1. ALWAYS call sql_db_list_tables, then sql_db_schema, then "
    "sql_db_query_checker, then sql_db_query — in that order. "
    "You MUST run sql_db_query to get real data; answering from schema "
    "samples alone is FORBIDDEN.\n"
    "2. Only SELECT queries are allowed — no INSERT/UPDATE/DELETE/DROP/ALTER.\n"
    "3. When the user mentions a column name (e.g. 'column name = Group'), "
    "write SELECT \"column\" FROM table to fetch ALL values of that column.\n"
    "4. Do not add LIMIT unless the user explicitly asked for a sample.\n"
    "5. After running the query, summarise the result in plain text.\n"
    "{history_block}"
)


def _sanitise(raw: str, existing: list) -> str:
    name = re.sub(r"\W+", "_", raw).strip("_") or "table_0"
    base, n = name, 1
    while name in existing:
        n += 1
        name = f"{base}_{n}"
    return name


def _build_db(file_id: str) -> SQLDatabase:
    if file_id in _db_cache:
        return _db_cache[file_id]
    engine = create_engine("duckdb:///:memory:")
    tables_df = get_all_real_tables(file_id)
    registered: list = []
    with engine.begin() as conn:
        for i, df in enumerate(tables_df):
            tname = _sanitise(str(df.attrs.get("page", f"table_{i}")), registered)
            registered.append(tname)
            df.to_sql(tname, conn, if_exists="replace", index=False)
    db = SQLDatabase(engine=engine, include_tables=registered, sample_rows_in_table_info=3)
    _db_cache[file_id] = db
    return db


def invalidate_cache(file_id: str) -> None:
    _db_cache.pop(file_id, None)


def _extract_last_sql(intermediate_steps: list) -> str | None:
    """Pull the last executed SELECT from agent intermediate_steps.
    Handles both string and dict tool_input (varies by agent_type version)."""
    last = None
    for step in intermediate_steps:
        action = step[0] if isinstance(step, (list, tuple)) else step
        if getattr(action, "tool", "") != "sql_db_query":
            continue
        ti = getattr(action, "tool_input", None)
        if isinstance(ti, str):
            candidate = ti
        elif isinstance(ti, dict):
            candidate = ti.get("query") or ti.get("input") or next(iter(ti.values()), "")
        else:
            candidate = None
        if candidate and _SELECT_RE.match(str(candidate).strip()):
            last = str(candidate).strip()
    return last


def _extract_sql_from_text(text: str) -> str | None:
    """Fallback: parse a SELECT out of the agent's verbose text output."""
    # Try fenced code block first
    m = re.search(r"```(?:sql)?\s*(SELECT.+?)```", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Then bare SELECT ... ; or SELECT ... \n\n
    m = re.search(r"(SELECT\s+.+?)(?:;|\n{2,}|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        if _SELECT_RE.match(candidate):
            return candidate
    return None


def _direct_sql_fallback(db: SQLDatabase, prompt: str) -> str | None:
    """Last-resort: ask the LLM to write one SELECT using the real schema,
    then return it.  Equivalent to the old data_query.py single-shot path."""
    schema = db.get_table_info()
    sys_msg = (
        f"DuckDB schema:\n{schema}\n\n"
        "Write ONE DuckDB SELECT query that answers the question.\n"
        "Rules: output ONLY the SQL, no fences, no explanation.\n"
        "Use double-quoted column names for any column with spaces/special chars.\n"
        "Do NOT add LIMIT unless the user explicitly asked for a sample."
    )
    try:
        raw = llm.invoke(f"{sys_msg}\n\nQUESTION: {prompt}").content
        sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", raw.strip())
        if _SELECT_RE.match(sql):
            return sql
    except Exception as e:
        print(f"(sql_agent fallback SQL gen failed: {e})")
    return None


def _run_sql(db: SQLDatabase, sql: str) -> dict | None:
    """Execute sql against db, return structured table dict or None on error."""
    try:
        with db._engine.begin() as conn:
            df = pd.read_sql(sql, conn)
        safe = df.astype(object).where(df.notna(), None)
        return {
            "columns": [str(c) for c in safe.columns],
            "rows": safe.values.tolist(),
        }
    except Exception as e:
        print(f"(sql_agent: re-fetch failed: {e})")
        return None


def run_sql_agent_query(file_id: str, prompt: str,
                        history_messages: list | None = None) -> dict:
    """Returns {"text": str, "table": dict | None, "sql": str | None}."""
    db = _build_db(file_id)
    if not db.get_usable_table_names():
        return {"text": "No tables found for this file.", "table": None, "sql": None}

    # Inject last 4 conversation turns so follow-up messages like
    # "i need to export this column data" have context.
    history_block = ""
    if history_messages:
        recent = history_messages[-8:]  # last 4 turns
        lines = [f"{m.type.upper()}: {m.content}" for m in recent
                 if hasattr(m, "content") and m.content]
        if lines:
            history_block = "\nRecent conversation:\n" + "\n".join(lines) + "\n"

    prefix = _AGENT_PREFIX.format(history_block=history_block)

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            agent_type="tool-calling",
            verbose=True,
            max_iterations=10,
            top_k=1000,   # don't silently add LIMIT 10
            prefix=prefix,
        )

    try:
        result = agent.invoke({"input": prompt})
    except Exception as e:
        return {"text": f"SQL agent error: {e}", "table": None, "sql": None}

    answer_text: str = result.get("output", "")
    steps = result.get("intermediate_steps", [])

    # --- SQL extraction with three fallback levels ---
    last_sql = _extract_last_sql(steps)

    if not last_sql:
        last_sql = _extract_sql_from_text(answer_text)
        if last_sql:
            print(f"(sql_agent: extracted SQL from text output)")

    if not last_sql:
        last_sql = _direct_sql_fallback(db, prompt)
        if last_sql:
            print(f"(sql_agent: used direct SQL fallback)")

    table = _run_sql(db, last_sql) if last_sql else None

    return {"text": answer_text, "table": table, "sql": last_sql}
