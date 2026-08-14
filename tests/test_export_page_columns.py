"""A second round of export bugs found live in the actual running server
after the first fix (app.export.spec, tests/test_export_spec.py) landed —
confirmed via interaction_log.jsonl and by reading the real .xlsx bytes on
disk, not by trusting the assistant's own response text.

Bug 1 — sheet-name collision (session 477b296a, 2026-08-13 20:01:16 and
again 2026-08-14 05:56:22): a CSV's single "sheet" is a synthetic
placeholder always named literally "data" (a CSV has no real sheets). A
prompt in this domain routinely contains the word "data" several times
("data file 3", "column data"), so app.graph.agent._extract_requested_sheet
matched it as if the user had named a real sheet, and routed to
_deterministic_sheet_export — which wrote the WHOLE sheet and never
consulted app.export.spec at all, silently bypassing the column-selection
fix for every tabular file. Confirmed: outputs/f1-exp-06.xlsx still had all
7 columns after the first fix supposedly landed.

Bug 2 — the docling (PDF) page-export path never applied column selection
either. "create excel file based on data-file one page number 6,7,8,9,10,
columns we need is = Yard No., Material code, Item" wrote all 6 real
columns from the assembled table, none of them the 3 requested, and the
model's own final answer FABRICATED a description ("Columns exported: Yard
No., Material code, Item... value from first occurrence in top section
used for all rows") of a file that did not match what it said — confirmed
by reading outputs/T1-01.xlsx directly with openpyxl/pandas.

Bug 3 — the same turn's response then had a bogus "Correction... there is
no column named `yard number will come same export`" banner appended,
because _COLUMN_ASK_RE's unquoted branch captured an entire clause of the
user's prompt (everything between the word "column" and a much later "in")
as though it were a claimed column name, then listed all 60-odd real
columns of an unrelated raw multi-header table — which is what made the
response LOOK like it had dumped every column, on top of the real file
actually doing so.

Bug 4/5 — while fixing bug 2, "columns we need is = 1.Yard No.,2.matrial
code ,3 item" broke the new requested-column-list parser (the numbered
item's own '1.' looked like an end-of-list period), and the typo "matrial"
for "Material" silently dropped a real, requested column instead of
including it.
"""
import os

import pandas as pd
import pytest

from app.config import OUTPUT_DIR
from app.export import spec as export_spec
from app.graph import agent as A
from app.graph import tools as T


@pytest.fixture(autouse=True)
def _isolated_user_prompt(monkeypatch):
    from app import state as app_state
    monkeypatch.setattr(app_state, "CURRENT_USER_PROMPT", None)


@pytest.fixture
def outdir():
    import glob
    from pathlib import Path
    before = set(glob.glob(os.path.join(OUTPUT_DIR, "*")))
    yield Path(OUTPUT_DIR)
    for f in glob.glob(os.path.join(OUTPUT_DIR, "*")):
        if f not in before:
            os.remove(f)


# --- bug 1: a CSV's synthetic "data" sheet name must not swallow column ----
# selection ---------------------------------------------------------------

REAL_PROMPT = ("in uploaded data file 3 we need to export two column data in "
              "f1-exp-06.xlsx , this column we need to export= Item Title, "
              "Item Quantity , do it")


@pytest.fixture
def csv_like_workbook(monkeypatch):
    table = pd.DataFrame({
        "Item Number": [1, 2, 3],
        "Item Title": ["Valve A", "Valve B", "Valve C"],
        "Item Quantity": [10, 20, 30],
        "Unit of Measure": ["EA", "EA", "EA"],
    })
    table.attrs["page"] = "data_Raw"  # matches tabular_ingest's csv naming
    monkeypatch.setattr("app.tables.helpers.get_all_real_tables",
                        lambda fid, **kw: [table])
    from app import state as app_state
    monkeypatch.setitem(app_state.FILE_KIND, "df3", "tabular")
    monkeypatch.setitem(app_state.FILE_META, "df3", {"sheets": ["data"]})
    monkeypatch.setattr(app_state, "FILE_ORIGINAL_NAME", {"df3": "data-file-3.csv"})
    monkeypatch.setattr(app_state, "FILE_ORDER", ["df3"])
    return table


def test_the_word_data_in_the_prompt_still_matches_the_synthetic_sheet(csv_like_workbook):
    """This IS the collision — confirms the test fixture reproduces the
    exact condition that broke in production, so the assertions below are
    proving something real."""
    assert A._extract_requested_sheet(REAL_PROMPT, "df3") == "data"


def test_sheet_routed_export_still_selects_only_the_requested_columns(
        outdir, csv_like_workbook):
    out = T.export_data.invoke({
        "question": "Item Title, Item Quantity", "format": "excel",
        "filename": "f1-exp-06.xlsx", "file_id": "df3"})
    assert "Applied" in out
    assert "kept only columns" in out
    written = pd.read_excel(outdir / "f1-exp-06.xlsx")
    assert list(written.columns) == ["Item Title", "Item Quantity"]
    assert len(written) == 3


def test_deterministic_sheet_export_without_a_prompt_is_unaffected(
        outdir, csv_like_workbook):
    """No prompt passed (the existing call shape from before this fix) must
    still export the whole sheet, unchanged — prompt-driven filtering is
    additive, not a behaviour change for callers that don't opt in."""
    out = A._deterministic_sheet_export("df3", "data", "whole.xlsx", "excel")
    assert "Verified" in out
    written = pd.read_excel(outdir / "whole.xlsx")
    assert list(written.columns) == ["Item Number", "Item Title",
                                     "Item Quantity", "Unit of Measure"]


# --- bug 2: docling page export must also apply column selection -----------

@pytest.fixture
def docling_pages(monkeypatch):
    from app import state as app_state

    def _table(page, cols_rows):
        df = pd.DataFrame(cols_rows)
        df.attrs["page"] = page
        return df

    tables = [
        _table(6, {"SL no": [1, 2], "Item": ["Globe Valve", "Gate Valve"],
                   "Spec No.": ["A.1", "A.2"], "Material code": ["PP1", "PP2"]}),
        _table(8, {"SL no": [3], "Item": ["Ball Valve"],
                   "Spec No.": ["A.3"], "Material code": ["PP3"]}),
    ]
    monkeypatch.setattr("app.tables.helpers.get_all_real_tables",
                        lambda fid, **kw: tables)
    monkeypatch.setitem(app_state.FILE_KIND, "sbs-f1", "docling")
    monkeypatch.setattr(app_state, "FILE_ORIGINAL_NAME", {"sbs-f1": "data-file-1.pdf"})
    return tables


REAL_PAGE_PROMPT = (
    "create excel file based on data-file one page number 6,8 , columns we "
    "need is = 1.Yard No.,2.matrial code ,3 item, Yard number is there in "
    "table top section in provided data file one check if any table top "
    "section yard number change so we need to change other vise every row "
    "column yard number will come same export in T1-01.xlsx")


def test_page_export_selects_only_matched_real_columns(outdir, docling_pages, monkeypatch):
    from app import state as app_state
    monkeypatch.setattr(app_state, "CURRENT_USER_PROMPT", REAL_PAGE_PROMPT)
    out = A._deterministic_page_export("sbs-f1", [6, 8], "T1-01.xlsx", "excel")
    assert "Applied" in out
    path = str(outdir / "T1-01.xlsx")
    from app.export.exporters import band_offset
    written = pd.read_excel(path, skiprows=band_offset(path))
    # "Item" matches exactly; "matrial code" typo-matches "Material code";
    # "Yard No." is not a real column and must not appear.
    assert set(written.columns) == {"Item", "Material code"}
    assert len(written) == 3


def test_page_export_reports_a_requested_column_that_is_not_real_data(
        outdir, docling_pages, monkeypatch):
    from app import state as app_state
    monkeypatch.setattr(app_state, "CURRENT_USER_PROMPT", REAL_PAGE_PROMPT)
    out = A._deterministic_page_export("sbs-f1", [6, 8], "T1-02.xlsx", "excel")
    assert "Yard No." in out
    assert "NOT in this file" in out


def test_page_export_without_a_current_prompt_is_unaffected(outdir, docling_pages):
    """No CURRENT_USER_PROMPT set (e.g. a direct call outside a chat turn)
    must behave exactly as before this fix — export everything."""
    out = A._deterministic_page_export("sbs-f1", [6, 8], "whole.xlsx", "excel")
    assert "Verified" in out
    path = str(outdir / "whole.xlsx")
    from app.export.exporters import band_offset
    written = pd.read_excel(path, skiprows=band_offset(path))
    assert set(written.columns) == {"SL no", "Item", "Spec No.", "Material code"}


# --- bug 3: the invented-column guard must not eat a whole clause ----------

def test_a_long_clause_containing_the_word_column_is_not_read_as_a_name(monkeypatch):
    from app import state as app_state
    monkeypatch.setitem(app_state.FILE_KIND, "sbs-f1", "docling")
    response = "The Excel file `T1-01.xlsx` has been successfully created."
    out = A._catch_invented_column(response, REAL_PAGE_PROMPT, "sbs-f1")
    assert out == response


def test_a_short_real_column_ask_is_still_caught(monkeypatch):
    """The guard this fix narrowed must still do its actual job."""
    monkeypatch.setattr("app.tables.helpers.get_all_real_tables",
                        lambda fid, **kw: [pd.DataFrame({"Item": [1]})])
    monkeypatch.setattr(A, "_header_only_column_names", lambda fid: set())
    from app import state as app_state
    monkeypatch.setitem(app_state.FILE_KIND, "fid1", "docling")
    out = A._catch_invented_column(
        "Schedule 1, Schedule 2.",
        "give me unique data of column EvaluationSchedules", "fid1")
    assert "there is no column named `EvaluationSchedules`" in out


# --- bug 4/5: numbered list parsing and typo-tolerant column matching -----

def test_numbered_list_items_are_not_cut_at_their_own_period():
    tokens = export_spec.extract_requested_column_tokens(REAL_PAGE_PROMPT)
    assert tokens == ["Yard No.", "matrial code", "item"]


def test_trailing_filler_after_a_column_list_is_not_a_token():
    tokens = export_spec.extract_requested_column_tokens(
        "this column we need to export= Item Title, Item Quantity , do it")
    assert tokens == ["Item Title", "Item Quantity"]


@pytest.mark.parametrize("typo,real", [
    ("matrial code", "Material code"),   # one letter dropped
    ("quntity", "Quantity"),              # one letter dropped
])
def test_a_one_character_typo_still_matches_the_real_column(typo, real):
    cols = export_spec.extract_requested_columns(
        f"export the {typo} column", [real, "Item", "Spec No."])
    assert cols == [real]


def test_a_typo_matched_column_is_not_also_reported_missing():
    selected = export_spec.extract_requested_columns(
        "export matrial code and item", ["Material code", "Item", "Spec No."])
    missing = export_spec.unmatched_requested_columns(
        "columns = matrial code, item", selected)
    assert missing == []


def test_two_typos_in_one_word_are_not_matched():
    """max_dist=1 is deliberate, same reasoning as the file-handle fuzzy
    matcher — beyond one typo, guessing becomes riskier than asking."""
    cols = export_spec.extract_requested_columns(
        "export the mtrail cod column", ["Material code"])
    assert cols == []


def test_short_column_names_are_not_fuzzy_matched():
    """The <5-char guard on fuzzy matching must still hold — 'Item' is
    short enough that a 1-edit fuzzy scan would match almost anything."""
    assert not export_spec._fuzzy_contains("item", "random unrelated text")
