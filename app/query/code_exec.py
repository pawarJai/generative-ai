"""
General-purpose data tool: gives the LLM a real, restricted Python sandbox
with the current file's tables already loaded as pandas DataFrames. This
replaces per-question-type intent categories (list_sheets, data_query,
row_sample) with one tool that generalizes to any file/question, because
it runs real code against real data instead of matching a phrasing regex.

SANDBOX SCOPE — what is and isn't guaranteed:
  - No imports beyond pandas (already provided in-namespace)
  - Hard wall-clock timeout per execution attempt
  - No filesystem writes of any kind from within the sandboxed code
  - Only pre-loaded DataFrames + pandas + a small numeric/string builtin
    allowlist are reachable

This is a process-level sandbox (signal-based timeout, restricted
namespace), not OS-level isolation (no container/VM boundary). That is an
acceptable tradeoff ONLY because the executed code originates from your
own LLM call (not directly from raw user input) and the namespace has no
I/O primitives. If you later let users paste code directly, or your LLM
provider changes, re-evaluate whether you need real OS-level sandboxing
(gVisor, Firecracker, a subprocess with seccomp) instead of this in-process
restriction.
"""
import re
import signal
import types
import builtins as _builtins
import pandas as pd
from typing import Dict, Any, List, Optional
from app import state
from app.config import llm
from app.tables.helpers import get_all_real_tables

_ALLOWED_BUILTIN_NAMES = {
    "len", "range", "sum", "min", "max", "sorted", "list", "dict", "set",
    "str", "int", "float", "bool", "round", "abs", "enumerate", "zip",
    "isinstance", "type", "print",
}
_SAFE_BUILTINS = {name: getattr(_builtins, name) for name in _ALLOWED_BUILTIN_NAMES
                  if hasattr(_builtins, name)}

# Module-level pandas surface the sandbox may reach. Everything not listed —
# notably every pd.read_* reader — is absent, so the model cannot open files.
_ALLOWED_PD_ATTRS = {
    "DataFrame", "Series", "Index", "MultiIndex", "concat", "merge", "melt",
    "pivot_table", "crosstab", "cut", "qcut", "to_numeric", "to_datetime",
    "to_timedelta", "isna", "isnull", "notna", "notnull", "unique", "factorize",
    "date_range", "NA", "NaT", "Timestamp", "Timedelta", "options",
}


def _make_safe_pandas() -> types.SimpleNamespace:
    """A pandas facade with no file I/O. `pd.read_excel(...)` must not exist:
    the DataFrames are already loaded, and there is no file for sandboxed code
    to legitimately open."""
    return types.SimpleNamespace(**{
        name: getattr(pd, name) for name in _ALLOWED_PD_ATTRS if hasattr(pd, name)
    })


# Static denylist applied to generated code BEFORE exec. Defence in depth: the
# restricted namespace already removes the module-level entry points, but
# DataFrame instance methods (df.to_csv) and dunder traversal
# (().__class__.__bases__) are still reachable from a live object, so reject
# those textually and let the model retry.
_FORBIDDEN_CODE = [
    (re.compile(r"\bto_(csv|excel|pickle|parquet|sql|hdf|feather|clipboard|json)\s*\("),
     "writing files is not allowed"),
    (re.compile(r"\bread_\w+\s*\("),
     "reading files is not allowed — the DataFrames are already loaded"),
    (re.compile(r"__\w+__"),
     "dunder attribute access is not allowed"),
    (re.compile(r"\b(open|eval|exec|compile|globals|locals|vars|getattr|setattr|delattr)\s*\("),
     "that builtin is not allowed"),
    (re.compile(r"\bimport\b"),
     "imports are not allowed — pd is already available"),
]


def _reject_unsafe_code(code: str) -> Optional[str]:
    """Return a specific rejection reason, naming the offending token so the
    retry prompt can actually act on it. A generic "not allowed" made the
    model retry the same construct until it ran out of attempts."""
    for pattern, reason in _FORBIDDEN_CODE:
        m = pattern.search(code)
        if m:
            return f"{reason} (found {m.group(0).strip()!r})"
    return None


class _ExecTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _ExecTimeout("execution exceeded time limit")


def _install_alarm(timeout_sec: int) -> bool:
    """signal.signal() only works on the main thread of the main interpreter.
    FastAPI runs sync endpoints in a threadpool, so this can legitimately fail —
    fall back to running without a timeout rather than crashing the request."""
    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout_sec)
        return True
    except (ValueError, AttributeError):
        return False


def _clear_alarm(installed: bool) -> None:
    if installed:
        try:
            signal.alarm(0)
        except (ValueError, AttributeError):
            pass


def _build_dataframe_map(file_ids: List[str]) -> Dict[str, pd.DataFrame]:
    """Loads tables from one or more files into a single sandbox namespace.

    When more than one file is in scope, each variable is prefixed with the
    source filename (`SalesQ1__Working_Sheet`) so that a sheet named
    "Working Sheet" in file A can't silently collide with — or shadow — a
    same-named sheet in file B. With a single file, names stay unprefixed
    (`Working_Sheet`) to match prior behaviour and existing tests."""
    multi = len(file_ids) > 1
    df_map: Dict[str, pd.DataFrame] = {}
    for fid in file_ids:
        tables = get_all_real_tables(fid)
        file_label = state.FILE_ORIGINAL_NAME.get(fid, fid)
        clean_file = "".join(c if c.isalnum() else "_" for c in file_label).strip("_") or fid
        for i, df in enumerate(tables):
            raw_name = str(df.attrs.get("page", f"table_{i}"))
            clean_page = "".join(c if c.isalnum() else "_" for c in raw_name).strip("_") or f"table_{i}"
            name = f"{clean_file}__{clean_page}" if multi else clean_page
            base, n = name, 1
            while name in df_map:
                n += 1
                name = f"{base}_{n}"
            df_map[name] = df
    return df_map


def _schema_text(df_map: Dict[str, pd.DataFrame]) -> str:
    return "\n".join(
        f"- {name}: columns={list(df.columns)}, {len(df)} rows"
        for name, df in df_map.items()
    )


_FENCE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL)


def _strip_fences(raw: str) -> str:
    """Extract runnable Python from an LLM reply.

    Models frequently emit a prose or comment line BEFORE the fence, so
    matching only a leading ``` leaves the prose in the code and raises
    SyntaxError. Prefer the contents of the first fenced block wherever it
    appears; fall back to stripping bare leading/trailing fences."""
    text = raw.strip()
    m = _FENCE_BLOCK.search(text)
    if m:
        return m.group(1).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("python"):
            text = text[len("python"):]
        elif text.startswith("py"):
            text = text[len("py"):]
    return text.strip()


def run_code_on_files(file_ids: List[str], prompt: str, max_iters: int = 3,
                      timeout_sec: int = 5) -> Dict[str, Any]:
    """Returns {"text": str, "table": {"columns","rows"} | None, "code": str | None}.
    Same return contract as the SQL-based data_query path, so dispatch.py's
    export-handling and record-card formatting logic doesn't need to change.

    Accepts a LIST of file_ids so a single question — or export — can span
    several uploaded files at once (e.g. "combine file A and file B into one
    CSV"), instead of being locked to whichever file was uploaded last."""
    df_map = _build_dataframe_map(file_ids)
    if not df_map:
        return {"text": "No tables found for the selected file(s).", "table": None, "code": None}

    schema = _schema_text(df_map)
    sheet_names = list(df_map.keys())
    n_files = len(file_ids)
    file_note = (
        f"These variables come from {n_files} different uploaded files — names are "
        f"prefixed with the source filename (e.g. `SalesQ1__Working_Sheet`) so sheets "
        f"with the same name in different files don't collide. To combine data across "
        f"files, use pd.concat([...]) or pd.merge(...) directly on these variables.\n\n"
        if n_files > 1 else ""
    )
    sys_prompt = (
        f"You are working with {'one or more already-loaded files' if n_files > 1 else 'an already-loaded spreadsheet/document'}.\n\n"
        f"Each of the following variables IS one sheet/table, "
        f"already loaded as a pandas DataFrame:\n{schema}\n\n"
        f"{file_note}"
        f"There are {len(sheet_names)} sheet(s)/table(s) total, named: "
        f"{sheet_names}\n\n"
        f"A dict `SHEETS` is also available, mapping each of those names to "
        f"its DataFrame. Use `SHEETS` when you need to iterate over sheets "
        f"(e.g. 'which sheet has the most columns'). Never use globals() or "
        f"locals() — they are blocked; `SHEETS` is the supported way.\n\n"
        f"Write Python code to answer the question. Rules:\n"
        f"- Assign your final answer to a variable named `result`.\n"
        f"- `result` should be a pandas DataFrame when the answer is tabular "
        f"data (a column, filtered rows, an aggregation table), or a plain "
        f"string/number for a single-value answer.\n"
        f"- There is NO file on disk to open. Never call pd.read_excel, "
        f"pd.read_csv, or any read_*/to_* function — they do not exist here. "
        f"Use the DataFrame variables listed above directly.\n"
        f"- If the user asks for data from a specific sheet, or asks to export a sheet, assign the DataFrame itself to `result` (e.g. `result = SHEETS['sheet_name']`), NOT the string name of the sheet.\n"
        f"- If the question is purely about the structure itself (e.g. 'what are the sheet names', 'how many sheets'), answer from the variable names listed above.\n"
        f"- Tables ending in `_Raw` are unlabeled full-fidelity fallbacks: generic "
        f"col_0/col_1/... column names, with the original header row left in as "
        f"ordinary data. Only use a `_Raw` table if no cleaned (non-`_Raw`) version "
        f"of that sheet exists. For any real question, prefer the cleaned table.\n"
        f"- For a 2-column key/value style table (e.g. a field-name column next to a "
        f"value column), find the answer by filtering for the row whose key column "
        f"matches the requested field (e.g. "
        f"`df[df.iloc[:, 0].astype(str).str.contains('Total Quantity', case=False, na=False)].iloc[0, 1]`). "
        f"Never guess a fixed row position like `.iloc[0, 0]`.\n"
        f"- If the question does NOT name a specific sheet and looks like a key/value "
        f"fact lookup (e.g. 'what is the total quantity', 'what is X'), do NOT guess a "
        f"single sheet to search. Loop over every 2-column table in SHEETS and check "
        f"each one's first column for a matching row before trying anything else:\n"
        f"  for _name, _df in SHEETS.items():\n"
        f"      if _df.shape[1] == 2:\n"
        f"          _match = _df[_df.iloc[:, 0].astype(str).str.contains('Total Quantity', case=False, na=False)]\n"
        f"          if not _match.empty:\n"
        f"              result = _match.iloc[0, 1]\n"
        f"              break\n"
        f"  Only fall back to column-name matching in a wide table if no 2-column "
        f"table contains a matching key.\n"
        f"- When the question asks to extract, list, combine, or export data (not "
        f"a single-value lookup), `result` must contain EVERY matching row. Never "
        f"use `.head()`, `.tail()`, a slice like `df[:5]`, or otherwise limit the "
        f"row count unless the user explicitly asked for a specific number of rows.\n"
        f"- Never invent or reference a column name you have not seen in the schema "
        f"above (including any the user typed in their question) — always select "
        f"columns by the real names/positions shown in the schema, never by a name "
        f"you're assuming should exist.\n"
        f"- When the question names a specific page number, the variable/sheet name "
        f"IS that exact page number (e.g. page 7 is the variable `7`, not `6` or `8`). "
        f"If the user asks for a page number that has no exact-matching variable name "
        f"in the schema above, that page has NO table — do NOT substitute a "
        f"different, differently-numbered table and claim it represents that page. "
        f"Instead set `result` to a short string saying exactly which page(s) were "
        f"asked for, which of those have no table, and which real page numbers ARE "
        f"available nearby, so the caller can tell the user the truth instead of "
        f"getting data silently mislabeled as the wrong page.\n"
        f"- No imports, no file I/O, no os/subprocess/network access.\n"
        f"- Output ONLY the Python code. No explanation, no markdown fences."
    )

    last_error: Optional[str] = None
    last_code: Optional[str] = None
    for _attempt in range(max_iters):
        retry_note = (f"\n\nYour previous attempt failed with this error:\n{last_error}\n"
                      f"Fix the code and try again." if last_error else "")
        raw = llm.invoke(f"{sys_prompt}{retry_note}\n\nQUESTION: {prompt}").content
        code = _strip_fences(raw)
        last_code = code

        rejection = _reject_unsafe_code(code)
        if rejection:
            last_error = f"code rejected by sandbox: {rejection}"
            continue

        namespace: Dict[str, Any] = {
            **df_map,
            "SHEETS": dict(df_map),   # supported way to iterate sheets
            "pd": _make_safe_pandas(),
            "__builtins__": _SAFE_BUILTINS,
        }
        installed = _install_alarm(timeout_sec)
        try:
            exec(code, namespace)  # noqa: S102 -- restricted namespace, see module docstring
        except _ExecTimeout:
            last_error = f"execution exceeded {timeout_sec}s"
            continue
        except Exception as e:  # noqa: BLE001 -- deliberately broad: any failure feeds back to the LLM
            last_error = f"{type(e).__name__}: {e}"
            continue
        finally:
            _clear_alarm(installed)

        result = namespace.get("result")
        if result is None:
            last_error = "code ran but did not assign a `result` variable"
            continue

        if isinstance(result, pd.DataFrame):
            safe = result.astype(object).where(result.notna(), None)
            return {"text": f"Found {len(safe)} row(s).",
                    "table": {"columns": [str(c) for c in safe.columns],
                              "rows": safe.values.tolist()},
                    "code": code}
        if isinstance(result, pd.Series):
            safe = result.astype(object).where(result.notna(), None)
            return {"text": f"Found {len(safe)} value(s).",
                    "table": {"columns": [str(result.name or "value")],
                              "rows": [[v] for v in safe.tolist()]},
                    "code": code}
        return {"text": str(result), "table": None, "code": code}

    return {"text": f"Couldn't answer after {max_iters} attempts. Last error: {last_error}",
            "table": None, "code": last_code}


def run_code_on_file(file_id: str, prompt: str, max_iters: int = 3,
                     timeout_sec: int = 5) -> Dict[str, Any]:
    """Single-file convenience wrapper around run_code_on_files. Kept for callers
    and tests that only ever had one file in scope."""
    return run_code_on_files([file_id], prompt, max_iters=max_iters, timeout_sec=timeout_sec)
