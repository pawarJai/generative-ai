"""Named-sheet export from a tabular (xlsx) file.

Confirmed production failure (session 477b296a, 14:11:03). The user, having
data-file-5.xlsx (a real multi-sheet workbook: Cover Sheet, Terms, Sheet1,
Working Sheet, Other, DropDown, Enquiry amendment form, Deviation List,
RFQ-Authorities, Delivery — all real, none corrupted, confirmed by opening
the file directly) asked:

    "in data-file-5 check data like want to export sheet name = cover sheet
     in to new file sheet-01.xlsx"

and got: "The file `data-file-5_9912` could not be parsed to retrieve any
sheet names or data. It may be corrupted..." — a fabricated file_id, for a
file that was never broken.

Root cause, traced by hand: _clean_export_question(prompt, "sheet-01.xlsx")
strips the filename, then applies _EXPORT_DIRECTIVE_RE, which matches
"export ... file" across up to 40 characters. Between those two words sat
"sheet name = cover sheet in to new " — the ENTIRE data request — and it was
deleted along with the phrasing it was meant to strip:

    'in data-file-5 check data like want to export sheet name = cover sheet
     in to new file sheet-01.xlsx'
    -> 'in data-file-5 check data like want to'

That garbage question went to the code-exec LLM sandbox, which sometimes
answered honestly ("incomplete question") and sometimes got lucky — the same
prompt in a fresh session against a fresh sandbox call produced a correct
27-row export. Not deterministic, so not something a prompt tweak fixes.

The fix bypasses the sandbox for this shape of request entirely: a sheet
name is matched against the file's OWN real sheet names (recovered by
ingestion, stored as each table's attrs["page"]) before the question is
ever cleaned or handed to an LLM.
"""
import glob
import os

import pandas as pd
import pytest

from app.config import OUTPUT_DIR
from app.graph import agent as A
from app.graph import tools as T


@pytest.fixture
def outdir():
    """The real OUTPUT_DIR, matching this suite's own convention
    (test_export_context.py writes real files there rather than
    monkeypatching app.config.OUTPUT_DIR) — exporters.py imports OUTPUT_DIR
    once at module load, so patching app.config.OUTPUT_DIR afterwards is
    invisible to it. Files this test creates are removed afterwards."""
    from pathlib import Path
    before = set(glob.glob(os.path.join(OUTPUT_DIR, "*")))
    yield Path(OUTPUT_DIR)
    for f in glob.glob(os.path.join(OUTPUT_DIR, "*")):
        if f not in before:
            os.remove(f)


def _table(page_label, cols_rows):
    df = pd.DataFrame(cols_rows)
    df.attrs["page"] = page_label
    return df


@pytest.fixture(autouse=True)
def _isolated_user_prompt(monkeypatch):
    """CURRENT_USER_PROMPT is a module-level global with no built-in reset.
    _resolve_source_specs (reached from export_data's multi-document check,
    which runs before this file's sheet-name check) prefers it over whatever
    `question` a test passes — a leak from an earlier test file left this
    stale exactly once and hijacked the routing here. monkeypatch guarantees
    a revert regardless of what any other test does."""
    from app import state as app_state
    monkeypatch.setattr(app_state, "CURRENT_USER_PROMPT", None)


@pytest.fixture
def workbook(monkeypatch):
    """A tabular file shaped like the real data-file-5.xlsx: one sheet with
    a Raw whole-sheet table plus structured sub-tables split out of it, and
    other, unrelated sheets."""
    tables = [
        _table("Cover Sheet_Raw", {
            "col_0": ["QUEST FLOW CONTROLS LTD", "Gat No - 324", None],
            "col_1": [None, None, "Offer Date :"],
        }),
        _table("Cover Sheet (block 2)", {
            "Customer Name & Address": ["Space for Customer Name"],
            "Enquiry Ref. & Date": ["Email:"],
        }),
        _table("Terms_Raw", {"col_0": ["Commercial Terms and Conditions"]}),
        _table("Working Sheet", {
            "Category": ["Ball Valve"], "Rating": ["PN10"]}),
    ]
    monkeypatch.setattr("app.tables.helpers.get_all_real_tables",
                        lambda fid, **kw: tables)
    from app import state as app_state
    monkeypatch.setitem(app_state.FILE_KIND, "df5", "tabular")
    monkeypatch.setattr(app_state, "FILE_ORIGINAL_NAME",
                        {"df5": "data-file-5.xlsx"})
    return tables


# --- the exact failing prompt ----------------------------------------------

REAL_PROMPT = ("in data-file-5 check data like want to export sheet name = "
              "cover sheet in to new file sheet-01.xlsx")


def test_the_real_failing_prompt_exports_the_real_sheet(outdir, workbook):
    out = T.export_data.invoke({
        "question": REAL_PROMPT, "format": "excel",
        "filename": "sheet-01.xlsx", "file_id": "df5"})
    assert "Verified" in out
    assert "data-file-5_9912" not in out  # the fabricated id must not appear
    df = pd.read_excel(outdir / "sheet-01.xlsx")
    assert df.shape == (3, 2)  # the Cover Sheet Raw table, not garbage


def test_clean_export_question_would_have_destroyed_this_request():
    """Proves the bug existed independent of LLM non-determinism: the
    sandbox never had a chance, regardless of which model or temperature
    answered it."""
    cleaned = A._clean_export_question(REAL_PROMPT, "sheet-01.xlsx")
    assert "cover sheet" not in cleaned.lower()
    assert cleaned == "in data-file-5 check data like want to"


def test_the_recovery_backstop_also_exports_the_real_sheet(outdir, workbook):
    """THE actual failing path. _attempt_recovery_export has its own copy of
    the dispatch and never calls the export_data tool — the same shape of
    bug as the side-by-side merge fix earlier in this file."""
    out = A._attempt_recovery_export(REAL_PROMPT, "df5")
    assert "Verified" in out
    df = pd.read_excel(outdir / "sheet-01.xlsx")
    assert df.shape == (3, 2)


# --- sheet name matching ----------------------------------------------------

@pytest.mark.parametrize("prompt,expected", [
    ("export sheet name = cover sheet into x.xlsx", "Cover Sheet"),
    ("export the cover sheet into x.xlsx", "Cover Sheet"),
    ("give me sheet named Working Sheet", "Working Sheet"),
    ('export "Terms" sheet', "Terms"),
    ("export sheet: Working Sheet to excel", "Working Sheet"),
])
def test_sheet_name_recognised_in_various_phrasings(prompt, expected, workbook):
    assert A._extract_requested_sheet(prompt, "df5") == expected


def test_no_sheet_named_returns_none(workbook):
    assert A._extract_requested_sheet("export page 8 to excel", "df5") is None


def test_docling_file_never_matches_a_sheet_name(monkeypatch):
    """A PDF has pages, not sheets — this must never fire for FILE_KIND ==
    'docling', however sheet-name-shaped the prompt looks."""
    from app import state as app_state
    monkeypatch.setitem(app_state.FILE_KIND, "pdf1", "docling")
    assert A._extract_requested_sheet(
        "export sheet name = cover sheet", "pdf1") is None


def test_longer_sheet_name_does_not_get_shadowed(monkeypatch):
    """'Sheet1' must not falsely match inside a prompt naming 'Working
    Sheet1-extended' or similar — longest name wins."""
    tables = [_table("Sheet1", {"a": [1]}),
             _table("Sheet1 Extended", {"a": [1], "b": [2]})]
    monkeypatch.setattr("app.tables.helpers.get_all_real_tables",
                        lambda fid, **kw: tables)
    from app import state as app_state
    monkeypatch.setitem(app_state.FILE_KIND, "df6", "tabular")
    assert A._extract_requested_sheet(
        "export the sheet1 extended data", "df6") == "Sheet1 Extended"


# --- _sheet_base_name --------------------------------------------------------

@pytest.mark.parametrize("label,expected", [
    ("Cover Sheet_Raw", "Cover Sheet"),
    ("Cover Sheet (block 2)", "Cover Sheet"),
    ("Working Sheet", "Working Sheet"),
    (None, ""),
])
def test_sheet_base_name_strips_ingestion_suffixes(label, expected):
    assert A._sheet_base_name(label) == expected


# --- _deterministic_sheet_export --------------------------------------------

def test_raw_table_is_preferred_over_blocks(outdir, workbook):
    """Asked for 'the sheet' rather than a specific table within it, the Raw
    whole-sheet dump is the faithful answer — not one arbitrary sub-table."""
    out = A._deterministic_sheet_export("df5", "Cover Sheet", "out.xlsx", "excel")
    assert "Verified" in out
    df = pd.read_excel(outdir / "out.xlsx")
    assert list(df.columns) == ["col_0", "col_1"]  # the Raw table's columns


def test_sheet_with_no_raw_table_unions_its_blocks(outdir, monkeypatch):
    tables = [_table("Other (block 1)", {"X": [1, 2]}),
             _table("Other (block 2)", {"X": [3]})]
    monkeypatch.setattr("app.tables.helpers.get_all_real_tables",
                        lambda fid, **kw: tables)
    from app import state as app_state
    monkeypatch.setitem(app_state.FILE_KIND, "df5", "tabular")
    out = A._deterministic_sheet_export("df5", "Other", "out.xlsx", "excel")
    assert "Verified" in out
    assert len(pd.read_excel(outdir / "out.xlsx")) == 3


def test_unknown_sheet_name_creates_no_file_and_says_so(outdir, workbook):
    out = A._deterministic_sheet_export("df5", "Nonexistent Sheet",
                                        "out.xlsx", "excel")
    assert "No sheet named" in out
    assert not (outdir / "out.xlsx").exists()


def test_source_sheet_is_named_in_the_result(outdir, workbook):
    out = A._deterministic_sheet_export("df5", "Working Sheet", "w.xlsx", "excel")
    assert "data-file-5.xlsx" in out
    assert "Working Sheet" in out


# --- must not affect docling / page-based exports ---------------------------

def test_docling_page_export_is_unaffected(outdir, monkeypatch):
    """The sheet-name check must be a no-op for ordinary PDF page exports —
    guarding against the new check swallowing an unrelated request shape."""
    from app import state as app_state
    monkeypatch.setitem(app_state.FILE_KIND, "pdf1", "docling")
    called = []
    monkeypatch.setattr(A, "_deterministic_page_export",
                        lambda *a, **k: called.append(1) or "page export ran")
    out = T.export_data.invoke({
        "question": "export page 8 to excel", "format": "excel",
        "filename": "p.xlsx", "file_id": "pdf1"})
    assert called == [1]
    assert out == "page export ran"
