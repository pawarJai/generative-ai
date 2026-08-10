
# Implementation Brief Part 4 — Code-Execution Tool (replaces hand-rolled intent routing)

## Which model to run this with, and why

This task is architecturally significant (building a sandboxed code-execution
tool, changing the core answer-generation path, security-sensitive) and this
project has already hit several regressions from smaller/faster models
missing edge cases. Use **Opus** for the implementation itself, and drop to
Sonnet only for pure mechanical cleanup afterward.

```bash
# Start Claude Code, then switch models mid-session as needed:
claude
/model opus      # use for this whole brief — architecture + security-sensitive
# once the core implementation is done and reviewed, for routine follow-up:
/model sonnet
```

Or launch directly on Opus for the whole session:

```bash
claude --model opus
```

To make Opus your default going forward (optional — costs more per token,
worth it while this project is still finding regressions):

```bash
export ANTHROPIC_MODEL="claude-opus-4-8"   # add to ~/.zshrc or ~/.bashrc
```

Check what you're actually running at any point with `/status`. If Claude
Code is genuinely stuck on something Opus itself can't resolve after a
couple of tries, that's a signal to come back here and paste the actual
error, not a signal to try a smaller model.

---

## Context for Claude Code

Read `app/query/planner.py`, `app/query/dispatch.py`, `app/query/data_query.py`,
`app/tables/helpers.py`, and `app/export/exporters.py` before starting.

**The core problem being fixed:** the current architecture hand-codes a
separate intent category (`list_sheets`, `data_query`, `row_sample`, etc.)
per *type of question*, with regex to route between them. Every new
phrasing of an existing question type risks landing in the wrong category
or getting shadowed by an earlier, broader regex check (confirmed
regression: a `list_sheets` fast path was added but never reached because
an earlier `data_fact_cues` check matched first for any prompt containing
the word "sheet"). This does not generalize — it requires a new regex for
every new way someone phrases a question, forever.

**The fix:** replace hand-coded intent categories for data questions with
ONE tool that lets the model write and execute real pandas code against
the actual loaded data — the same mechanism ChatGPT/Claude's own code
interpreter uses for file analysis. This generalizes to any file and any
phrasing by construction, because the model is writing code against
whatever the real schema is, not matching a category.

---

## STEP 1 — Build the code-execution tool

Create `app/query/code_exec.py`:

```python
"""
General-purpose data tool: gives the LLM a real, restricted Python sandbox
with the current file's tables already loaded as pandas DataFrames. This
replaces per-question-type intent categories (list_sheets, data_query,
row_sample) with one tool that generalizes to any file/question, because
it runs real code against real data instead of matching a phrasing pattern.

SECURITY: executes model-written code. This is NOT optional to restrict —
every line of model-written code is untrusted input from the moment a user
can influence the prompt. Restrictions enforced here:
  - No os/sys/subprocess/socket/open/eval/exec/importlib access
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
import signal
import builtins as _builtins
import pandas as pd
from typing import Dict, Any, Optional
from app.config import llm
from app.tables.helpers import get_all_real_tables

_ALLOWED_BUILTIN_NAMES = {
    "len", "range", "sum", "min", "max", "sorted", "list", "dict", "set",
    "str", "int", "float", "bool", "round", "abs", "enumerate", "zip",
    "isinstance", "type", "print",
}
_SAFE_BUILTINS = {name: getattr(_builtins, name) for name in _ALLOWED_BUILTIN_NAMES
                  if hasattr(_builtins, name)}


class _ExecTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _ExecTimeout("execution exceeded time limit")


def _build_dataframe_map(file_id: str) -> Dict[str, pd.DataFrame]:
    tables = get_all_real_tables(file_id)
    df_map: Dict[str, pd.DataFrame] = {}
    for i, df in enumerate(tables):
        raw_name = str(df.attrs.get("page", f"table_{i}"))
        name = "".join(c if c.isalnum() else "_" for c in raw_name).strip("_") or f"table_{i}"
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


def run_code_on_file(file_id: str, prompt: str, max_iters: int = 3,
                      timeout_sec: int = 5) -> Dict[str, Any]:
    """Returns {"text": str, "table": {"columns","rows"} | None, "code": str | None}.
    Same return contract as the SQL-based data_query path, so dispatch.py's
    export-handling and record-card formatting logic doesn't need to change."""
    df_map = _build_dataframe_map(file_id)
    if not df_map:
        return {"text": "No tables found for this file.", "table": None, "code": None}

    schema = _schema_text(df_map)
    sys_prompt = (
        f"You have these pandas DataFrames already loaded by name:\n{schema}\n\n"
        f"Write Python code to answer the question. Rules:\n"
        f"- Assign your final answer to a variable named `result`.\n"
        f"- `result` should be a pandas DataFrame when the answer is tabular "
        f"data (a column, filtered rows, an aggregation table), or a plain "
        f"string/number for a single-value answer.\n"
        f"- Only use pandas operations on the given DataFrames. `pd` is "
        f"already available. No imports, no file I/O, no os/subprocess/"
        f"network access -- none of that is available in this sandbox anyway.\n"
        f"- Output ONLY the Python code. No explanation, no markdown fences."
    )

    last_error: Optional[str] = None
    last_code: Optional[str] = None
    for _attempt in range(max_iters):
        retry_note = (f"\n\nYour previous attempt failed with this error:\n{last_error}\n"
                       f"Fix the code and try again." if last_error else "")
        raw = llm.invoke(f"{sys_prompt}{retry_note}\n\nQUESTION: {prompt}").content
        code = raw.strip()
        if code.startswith("```"):
            code = code.strip("`")
            if code.startswith("python"):
                code = code[len("python"):]
        code = code.strip()
        last_code = code

        namespace: Dict[str, Any] = {**df_map, "pd": pd, "__builtins__": _SAFE_BUILTINS}
        try:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout_sec)
            exec(code, namespace)  # noqa: S102 -- restricted namespace, see module docstring
            signal.alarm(0)
        except _ExecTimeout:
            last_error = f"execution exceeded {timeout_sec}s"
            continue
        except Exception as e:  # noqa: BLE001 -- deliberately broad: any failure feeds back to the LLM
            signal.alarm(0)
            last_error = f"{type(e).__name__}: {e}"
            continue

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
            safe = result.where(result.notna(), None)
            return {"text": f"Found {len(safe)} value(s).",
                     "table": {"columns": [str(result.name or "value")],
                               "rows": [[v] for v in safe.tolist()]},
                     "code": code}
        return {"text": str(result), "table": None, "code": code}

    return {"text": f"Couldn't answer after {max_iters} attempts. Last error: {last_error}",
             "table": None, "code": last_code}
```

---

## STEP 2 — Wire it into dispatch.py

Replace the `list_sheets`, `data_query`, and `row_sample` intent branches
with a single call to `run_code_on_file`. Keep the existing `verify_export`
disk-verification logic from the last brief unchanged for the sink/save path.

```python
from app.query.code_exec import run_code_on_file

if plan.intent == "data_query":   # this now handles what list_sheets/row_sample used to
    result = run_code_on_file(fid, prompt)
    if plan.sink and result.get("table"):
        df = pd.DataFrame(result["table"]["rows"], columns=result["table"]["columns"])
        export_plan = QueryPlan(intent="export", filename=plan.filename, sink=plan.sink)
        return EXPORTERS[plan.sink](fid, export_plan, tables=[df])
    if result.get("table") and len(result["table"]["columns"]) > WIDE_TABLE_COLUMN_THRESHOLD:
        return format_as_record_cards(result["table"])
    return result["text"]
```

In `planner.py`, simplify: any tabular-file prompt that isn't clearly
`overview`/`generate`/`export-with-explicit-schema` should route to
`data_query` and let the code-execution tool figure out what's actually
being asked — you no longer need separate regex for "list sheets" vs "give
me rows" vs "filter by column" vs "count something." Keep the deterministic
`greeting` and `chat_history` fast paths from the previous brief; those
aren't data questions and don't benefit from code execution.

---

## STEP 3 — Replace static routing tests with dynamic, real-data integration tests

The old `tests/test_routing.py` (checking prompt → intent category) is now
mostly obsolete, since there are fewer categories to route between. Replace
it with tests that ingest REAL files and assert on REAL computed values —
this is what actually generalizes to a new file, because it never hardcodes
an expected answer, only a way to independently verify the answer against
the same data through a second, trusted path (direct pandas).

Create `tests/test_code_exec.py`:

```python
import pandas as pd
import pytest
from app.query.code_exec import run_code_on_file
from app.ingestion.universal import universal_ingest
from app import state

@pytest.fixture(scope="module")
def clean_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "clean.csv"
    pd.DataFrame({"Name": ["A", "B", "C"], "Value": [10, 20, 30]}).to_csv(path, index=False)
    universal_ingest(str(path), "test_clean_csv")
    return "test_clean_csv"

@pytest.fixture(scope="module")
def messy_xlsx(tmp_path_factory):
    # Point this at a REAL messy file from your actual data set (multi-sheet,
    # merged cells) -- not a synthetic clean one. This is the file class that
    # broke previous fixes.
    path = "path/to/a/real/messy/working-sheet.xlsx"
    universal_ingest(path, "test_messy_xlsx")
    return "test_messy_xlsx"

def test_sheet_count_matches_real_ingestion(messy_xlsx):
    from app.tables.helpers import get_all_real_tables
    real_count = len(get_all_real_tables(messy_xlsx))
    result = run_code_on_file(messy_xlsx, "how many sheets does this file have")
    assert str(real_count) in result["text"], (
        f"Expected real sheet count {real_count} to appear in: {result['text']}")

def test_column_extraction_matches_real_data(clean_csv):
    from app.tables.helpers import get_all_real_tables
    real_df = get_all_real_tables(clean_csv)[0]
    result = run_code_on_file(clean_csv, "give me all the values in the Value column")
    assert result["table"] is not None
    returned_values = sorted(v[0] for v in result["table"]["rows"])
    real_values = sorted(real_df["Value"].tolist())
    assert returned_values == real_values, (
        f"Returned {returned_values}, real data has {real_values}")

def test_malformed_question_gives_honest_failure_not_fabrication(clean_csv):
    result = run_code_on_file(clean_csv, "what is the average of the NonexistentColumn field")
    assert result["table"] is None
    assert "couldn't" in result["text"].lower() or "error" in result["text"].lower()

def test_sandbox_blocks_filesystem_access(clean_csv):
    # Directly probe the sandbox with code that WOULD write a file if the
    # restriction were broken -- this test should show the attempt failed,
    # not that a file was created.
    from app.query.code_exec import run_code_on_file
    import os
    marker = "/tmp/sandbox_escape_test_marker"
    if os.path.exists(marker):
        os.remove(marker)
    run_code_on_file(clean_csv, f"write the text 'escaped' to a file at {marker}")
    assert not os.path.exists(marker), "Sandbox failed to block filesystem write"
```

Run:

```bash
pytest tests/test_code_exec.py -v
```

Every test must pass against **both** a clean synthetic file and a real
messy file from your actual data set before this is considered done — the
messy-file fixture is not optional, since that's the file class that has
broken every previous fix in this project.

---

## STEP 4 — Report back

Show me:

1. Real `pytest` output for all four tests, against the real messy file
2. One live example: a genuinely new phrasing you haven't tested before
   (something not in this brief), run against a real file, with the actual
   generated code and actual result shown
3. Confirmation the sandbox-escape test actually attempted and failed the
   escape, not just that the test file exists

Do not report this as done without that real output.
