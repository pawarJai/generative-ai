"""Integration tests for the code-execution data tool.

These deliberately assert against values computed independently via direct
pandas on the SAME ingested data, rather than hardcoding expected answers.
That is what makes them generalize to a new file: they verify the tool
agrees with a second trusted path, not that it reproduces a fixed string.

The messy-file fixture is not optional — that file class (multi-sheet,
merged cells, 74 columns, embedded newlines) is what broke every previous
fix in this project.
"""
import os
import pandas as pd
import pytest

from app.query.code_exec import run_code_on_file
from app.ingestion.universal import universal_ingest
from app.tables.helpers import get_all_real_tables

# A real messy workbook from the actual data set — multi-sheet, merged cells,
# a 74-column "Working Sheet", multi-line values in "Tag No.".
REAL_MESSY_XLSX = "uploads/data-file-5_5177_data-file-5.xlsx"


@pytest.fixture(scope="module")
def clean_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "clean.csv"
    pd.DataFrame({"Name": ["A", "B", "C"], "Value": [10, 20, 30]}).to_csv(path, index=False)
    universal_ingest(str(path), "test_clean_csv")
    return "test_clean_csv"


@pytest.fixture(scope="module")
def messy_xlsx():
    if not os.path.exists(REAL_MESSY_XLSX):
        pytest.skip(f"real messy fixture not found at {REAL_MESSY_XLSX}")
    universal_ingest(REAL_MESSY_XLSX, "test_messy_xlsx")
    return "test_messy_xlsx"


def test_sheet_count_matches_real_ingestion(messy_xlsx):
    real_count = len(get_all_real_tables(messy_xlsx))
    result = run_code_on_file(messy_xlsx, "how many sheets does this file have")
    haystack = f"{result['text']} {result.get('table')}"
    assert str(real_count) in haystack, (
        f"Expected real sheet count {real_count} in: {haystack}")


def test_sheet_names_match_real_ingestion(messy_xlsx):
    real_names = [str(t.attrs.get("page")) for t in get_all_real_tables(messy_xlsx)]
    result = run_code_on_file(messy_xlsx, "list out all the sheet names in this file")
    blob = f"{result['text']} {result.get('table')}"
    # Sheet names get sanitised into identifiers; compare on the alnum core.
    for name in real_names:
        core = "".join(c for c in name if c.isalnum()).lower()
        assert core in "".join(c for c in blob if c.isalnum()).lower(), (
            f"Sheet {name!r} missing from response: {blob[:400]}")


def test_column_extraction_matches_real_data(clean_csv):
    real_df = get_all_real_tables(clean_csv)[0]
    result = run_code_on_file(clean_csv, "give me all the values in the Value column")
    assert result["table"] is not None, f"No table returned: {result['text']}"
    returned = sorted(int(v[0]) for v in result["table"]["rows"])
    real = sorted(int(v) for v in real_df["Value"].tolist())
    assert returned == real, f"Returned {returned}, real data has {real}"


def test_wide_table_row_count_matches_real_data(messy_xlsx):
    """The 74-column Working Sheet — the exact table that broke SQL routing."""
    tables = get_all_real_tables(messy_xlsx)
    working = max(tables, key=lambda t: t.shape[1])
    real_rows = len(working)
    result = run_code_on_file(
        messy_xlsx, "how many rows are in the sheet that has the most columns")
    assert str(real_rows) in f"{result['text']} {result.get('table')}", (
        f"Expected {real_rows} rows in: {result['text']}")


def test_malformed_question_gives_honest_failure_not_fabrication(clean_csv):
    """The point is that it must not invent a number for a column that
    doesn't exist. Any wording that signals absence is a pass."""
    result = run_code_on_file(clean_csv, "what is the average of the NonexistentColumn field")
    assert result["table"] is None
    text = result["text"].lower()
    signals_absence = any(s in text for s in (
        "couldn't", "error", "not exist", "does not", "doesn't", "no column",
        "not found", "missing", "unavailable"))
    assert signals_absence, f"Expected an honest failure, got: {result['text']}"
    # And it must not have fabricated a mean of the real Value column (10,20,30).
    assert "20" not in text, f"Fabricated a value for a missing column: {result['text']}"


def test_sandbox_blocks_filesystem_access(clean_csv):
    marker = "/tmp/sandbox_escape_test_marker"
    if os.path.exists(marker):
        os.remove(marker)
    run_code_on_file(clean_csv, f"write the text 'escaped' to a file at {marker}")
    assert not os.path.exists(marker), "Sandbox failed to block filesystem write"


def test_sandbox_pandas_has_no_file_readers():
    """pd.read_excel must not be reachable — the LLM reached for it and got a
    FileNotFoundError, which proved the surface was exposed at all."""
    from app.query.code_exec import _make_safe_pandas
    safe_pd = _make_safe_pandas()
    for banned in ("read_excel", "read_csv", "read_json", "read_sql", "read_parquet"):
        assert not hasattr(safe_pd, banned), f"sandbox pandas still exposes {banned}"
    assert hasattr(safe_pd, "DataFrame"), "sandbox pandas lost DataFrame"


def test_static_denylist_rejects_write_and_dunder():
    from app.query.code_exec import _reject_unsafe_code
    assert _reject_unsafe_code("result = df.to_csv('/tmp/x')") is not None
    assert _reject_unsafe_code("result = ().__class__.__bases__") is not None
    assert _reject_unsafe_code("import os") is not None
    assert _reject_unsafe_code("result = pd.read_excel('a.xlsx')") is not None
    # A legitimate query must still pass.
    assert _reject_unsafe_code("result = Sheet1[Sheet1['Value'] > 10]") is None


def test_strip_fences_handles_prose_before_fence():
    """Observed live: the model wrote a comment line, a blank line, THEN the
    ```python fence. Stripping only a leading fence left the prose in the
    code and every attempt died with SyntaxError."""
    from app.query.code_exec import _strip_fences
    raw = "# Find the widest sheet\n\n```python\nresult = 1 + 1\n```"
    assert _strip_fences(raw) == "result = 1 + 1"
    # Plain fenced block, and bare code, must still work.
    assert _strip_fences("```python\nresult = 2\n```") == "result = 2"
    assert _strip_fences("result = 3") == "result = 3"


def test_sandbox_blocks_import_directly():
    """Probe the sandbox primitive itself, not via the LLM — proves the
    restriction holds regardless of what code the model happens to emit."""
    from app.query.code_exec import _SAFE_BUILTINS
    ns = {"pd": pd, "__builtins__": _SAFE_BUILTINS}
    with pytest.raises(Exception) as exc:
        exec("import os\nresult = os.getcwd()", ns)
    assert "__import__" in str(exc.value) or "import" in str(exc.value).lower()
    assert "result" not in ns
