"""create_excel_template: build a file from an explicit column list, with
or without an uploaded document — and export_data's own fallback to it.

Confirmed production failure (session 7e43a023, 2026-08-14 08:21): "create
excel file based on this columsn = 1.Sr No 2.Group 3.Tag No. 4.Yard No. ...
export in w1-02.xlsx" with nothing uploaded got "No files have been
uploaded, so I cannot generate the quotation or extract data" — twice,
worded slightly differently each time. That answer is correct as far as it
goes (export_data genuinely has nothing to pull FROM) but is not what the
request needed: a blank template with those 31 headers is a completely
reasonable thing to hand back, and the same request against an UPLOADED
document that has no such columns either (session 477b296a, same day)
produced multi-minute hangs trying the LLM sandbox instead of the same
honest, fast, blank-cells-explained answer.

Two bugs were found and fixed while building this: the requested-column-
list parser choked on this user's actual, repeated typo "columsn" for
"columns" (never matched at all, so the fallback never triggered), and once
that was fixed, a second bug split the numbered marker "10." into "1" +
"0.TestingStd" (a digit run inside a digit run) and ALSO fired inside the
filename "w1-02.xlsx" at "02." — both silently corrupting the column list.
"""
import os

import pandas as pd
import pytest

from app.config import OUTPUT_DIR
from app.export import spec as export_spec
from app.graph.tools import create_excel_template, export_data

REAL_31_COLUMN_PROMPT = (
    "create excel file based on this columsn = 1.Sr No 2.Group 3.Tag No. "
    "4.Yard No. 5.Customer Item/Material Code 6.Customer Description "
    "7.Service 8.Product_Type 9.Design Std 10.TestingStd 11.Category "
    "12.Body Type 13.Rating 14.Series 15.Size 16.Qty/SS 17.6 Ship Set "
    "18.Body 19.Disc/Ball/Wedge/Plug 20.Stem 21.Seat 22.End Connection "
    "23.Flange Drilling 24.Operator 25.Leakage_Rate 26.Paint_Finish "
    "27.Certification 28.Special Req 29.Unit Rate (INR) "
    "30.Total Value for 1 Shipset 31.Total Value for All Shipset, "
    "export in w1-02.xlsx")

REAL_31_COLUMNS = [
    "Sr No", "Group", "Tag No.", "Yard No.", "Customer Item/Material Code",
    "Customer Description", "Service", "Product_Type", "Design Std",
    "TestingStd", "Category", "Body Type", "Rating", "Series", "Size",
    "Qty/SS", "6 Ship Set", "Body", "Disc/Ball/Wedge/Plug", "Stem", "Seat",
    "End Connection", "Flange Drilling", "Operator", "Leakage_Rate",
    "Paint_Finish", "Certification", "Special Req", "Unit Rate (INR)",
    "Total Value for 1 Shipset", "Total Value for All Shipset",
]


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    from app import state as app_state
    monkeypatch.setattr(app_state, "CURRENT_USER_PROMPT", None)
    monkeypatch.setattr(app_state, "FILE_ORDER", [])
    monkeypatch.setattr(app_state, "FILE_KIND", {})
    monkeypatch.setattr(app_state, "FILE_ORIGINAL_NAME", {})
    # A bare module-level global, not session-scoped — a different test
    # elsewhere in the full suite can leave it pointing at a real file_id,
    # which resolve_target_file then trusts as "the active file" even with
    # FILE_ORDER empty.
    monkeypatch.setattr(app_state, "ACTIVE_FILE_ID", None)


@pytest.fixture
def outdir():
    import glob
    from pathlib import Path
    before = set(glob.glob(os.path.join(OUTPUT_DIR, "*")))
    yield Path(OUTPUT_DIR)
    for f in glob.glob(os.path.join(OUTPUT_DIR, "*")):
        if f not in before:
            os.remove(f)


# --- the numbered-list token parser, on the exact real (typo'd) prompt ----

def test_the_typo_columsn_still_triggers_the_list_parser():
    tokens = export_spec.extract_requested_column_tokens(REAL_31_COLUMN_PROMPT)
    assert tokens == REAL_31_COLUMNS


def test_a_two_digit_marker_is_not_split_on_its_own_trailing_digit():
    """'10.TestingStd' must not become '1' + '0.TestingStd'."""
    tokens = export_spec.extract_requested_column_tokens(
        "columns = 9.Design Std 10.TestingStd 11.Category")
    assert tokens == ["Design Std", "TestingStd", "Category"]


def test_a_numbered_filename_does_not_leak_into_the_list():
    """'w1-02.xlsx' contains '02.' — a lookalike marker — right after where
    the real list ends."""
    tokens = export_spec.extract_requested_column_tokens(
        "columns = 1.Item 2.Quantity, export in w1-02.xlsx")
    assert tokens == ["Item", "Quantity"]
    assert "xlsx" not in tokens


def test_a_comma_separated_list_still_works_unchanged():
    tokens = export_spec.extract_requested_column_tokens(
        "columns we need is = 1.Yard No.,2.matrial code ,3 item")
    assert tokens == ["Yard No.", "matrial code", "item"]


# --- create_excel_template: works with no document at all -----------------

def test_builds_a_blank_template_with_no_document_uploaded(outdir):
    out = create_excel_template.invoke({
        "columns": REAL_31_COLUMNS, "filename": "t-blank.xlsx"})
    assert "Created" in out
    assert "Left blank" in out
    written = pd.read_excel(outdir / "t-blank.xlsx")
    assert list(written.columns) == REAL_31_COLUMNS
    assert len(written) == 0


def test_a_constant_is_broadcast_into_every_row(outdir):
    out = create_excel_template.invoke({
        "columns": ["Yard No.", "Item"], "filename": "t-const.xlsx",
        "constants": {"Yard No.": "BY531-536"}})
    assert "Set to a fixed value" in out
    written = pd.read_excel(outdir / "t-const.xlsx")
    assert len(written) == 1
    assert written.iloc[0]["Yard No."] == "BY531-536"
    assert pd.isna(written.iloc[0]["Item"]) or written.iloc[0]["Item"] == ""


def test_no_column_names_is_a_clear_error_not_a_blank_success():
    out = create_excel_template.invoke({"columns": [], "filename": "t-empty.xlsx"})
    assert "No column names" in out


# --- create_excel_template: fills what it can from a real document --------

def test_matching_columns_are_filled_from_a_real_document(outdir, monkeypatch):
    from app import state as app_state
    table = pd.DataFrame({
        "Item": ["Globe Valve", "Gate Valve"],
        "Material code": ["PP1", "PP2"],
    })
    monkeypatch.setattr("app.tables.helpers.get_all_real_tables",
                        lambda fid, **kw: [table] if fid == "df1" else [])
    monkeypatch.setitem(app_state.FILE_KIND, "df1", "docling")
    monkeypatch.setattr(app_state, "FILE_ORIGINAL_NAME", {"df1": "spec.pdf"})

    out = create_excel_template.invoke({
        "columns": ["Item", "Material code", "Yard No."],
        "filename": "t-mixed.xlsx", "file_id": "df1"})
    assert "Filled from" in out
    assert "Left blank" in out
    written = pd.read_excel(outdir / "t-mixed.xlsx")
    assert list(written["Item"]) == ["Globe Valve", "Gate Valve"]
    assert list(written["Material code"]) == ["PP1", "PP2"]
    assert written["Yard No."].isna().all() or (written["Yard No."] == "").all()


def test_a_constant_overrides_a_real_document_match(outdir, monkeypatch):
    """The user stating a fixed value directly ('set Yard No. to X for all
    rows') is a deliberate override, not a fallback only used when nothing
    real matches."""
    from app import state as app_state
    table = pd.DataFrame({"Item": ["A", "B"], "Yard No.": ["OLD1", "OLD2"]})
    monkeypatch.setattr("app.tables.helpers.get_all_real_tables",
                        lambda fid, **kw: [table])
    monkeypatch.setitem(app_state.FILE_KIND, "df1", "docling")
    monkeypatch.setattr(app_state, "FILE_ORIGINAL_NAME", {"df1": "spec.pdf"})

    out = create_excel_template.invoke({
        "columns": ["Item", "Yard No."], "filename": "t-override.xlsx",
        "constants": {"Yard No.": "BY531-536"}, "file_id": "df1"})
    written = pd.read_excel(outdir / "t-override.xlsx")
    assert list(written["Yard No."]) == ["BY531-536", "BY531-536"]
    assert list(written["Item"]) == ["A", "B"]


def test_omitting_file_id_resolves_the_same_document_as_passing_it_explicitly(
        outdir, monkeypatch):
    """Confirmed production failure (session 7e43a023, 6 files uploaded):
    the model called this tool twice for one request, passing file_id the
    first time and omitting it the second. With several files loaded, the
    old resolution only auto-picked a document when EXACTLY ONE file was
    uploaded, so the second call silently fell back to no document at all
    and wrote a fully blank template where the first call had filled real
    data — same request, same session, two contradictory files."""
    from app import state as app_state
    table = pd.DataFrame({
        "Category": ["Ball Valve", "Gate Valve"],
        "Rating": ["150#", "300#"],
    })
    monkeypatch.setattr("app.tables.helpers.get_all_real_tables",
                        lambda fid, **kw: [table] if fid == "df2" else [])
    monkeypatch.setattr(app_state, "FILE_ORDER", ["df1", "df2", "df3"])
    monkeypatch.setitem(app_state.FILE_KIND, "df2", "docling")
    monkeypatch.setattr(app_state, "FILE_ORIGINAL_NAME", {"df2": "data-file-2.pdf"})
    monkeypatch.setattr(app_state, "ACTIVE_FILE_ID", "df2")
    monkeypatch.setattr(app_state, "CURRENT_USER_PROMPT",
                        "create excel with columns = Category, Rating")

    with_id = create_excel_template.invoke({
        "columns": ["Category", "Rating"], "filename": "t-with-id.xlsx",
        "file_id": "df2"})
    without_id = create_excel_template.invoke({
        "columns": ["Category", "Rating"], "filename": "t-without-id.xlsx"})

    assert "Filled from" in with_id
    assert "Filled from" in without_id
    w1 = pd.read_excel(outdir / "t-with-id.xlsx")
    w2 = pd.read_excel(outdir / "t-without-id.xlsx")
    assert list(w1["Category"]) == list(w2["Category"]) == ["Ball Valve", "Gate Valve"]
    assert list(w1["Rating"]) == list(w2["Rating"]) == ["150#", "300#"]


# --- export_data's own fallback, end to end --------------------------------

def test_export_data_falls_back_to_a_template_with_nothing_uploaded(outdir, monkeypatch):
    from app import state as app_state
    monkeypatch.setattr(app_state, "CURRENT_USER_PROMPT", REAL_31_COLUMN_PROMPT)
    out = export_data.invoke({
        "question": "create excel with these columns",
        "format": "excel", "filename": "w1-fallback.xlsx"})
    assert "No files uploaded" not in out
    assert "Created" in out
    written = pd.read_excel(outdir / "w1-fallback.xlsx")
    assert list(written.columns) == REAL_31_COLUMNS


def test_export_data_falls_back_to_a_template_when_nothing_in_the_document_matches(
        outdir, monkeypatch):
    from app import state as app_state
    # Deliberately no overlap at all with any of REAL_31_COLUMNS (unlike a
    # column such as "Description", which would legitimately fuzzy-match
    # "Customer Description" and take the existing column-selection path
    # instead of this fallback — a real, separate feature, not a bug).
    table = pd.DataFrame({"SKU": ["V001"], "Notes": ["n/a"]})
    monkeypatch.setattr("app.tables.helpers.get_all_real_tables",
                        lambda fid, **kw: [table])
    monkeypatch.setitem(app_state.FILE_KIND, "df1", "docling")
    monkeypatch.setattr(app_state, "FILE_ORDER", ["df1"])
    monkeypatch.setattr(app_state, "FILE_ORIGINAL_NAME", {"df1": "spec.pdf"})
    monkeypatch.setattr(app_state, "CURRENT_USER_PROMPT", REAL_31_COLUMN_PROMPT)

    out = export_data.invoke({
        "question": "create excel with these columns",
        "format": "excel", "filename": "w1-fallback2.xlsx", "file_id": "df1"})
    assert "Could not extract data" not in out
    written = pd.read_excel(outdir / "w1-fallback2.xlsx")
    assert list(written.columns) == REAL_31_COLUMNS
    assert "Left blank" in out


def test_export_data_still_refuses_plainly_for_a_short_ambiguous_request(monkeypatch):
    """Fewer than 3 column-like tokens is not confidently 'an explicit
    schema to build' — export_data's original, honest refusal must still
    apply rather than guessing at a one-word 'template'."""
    from app import state as app_state
    # FILE_ORDER alone isn't enough isolation here: resolve_target_file
    # also consults the durable, disk-backed registry, which can hold real
    # entries left by manual/live testing outside pytest — that made this
    # test order-dependent (passed alone, failed in the full suite) until
    # the registry itself was mocked empty too.
    monkeypatch.setattr("app.persistence.get_all_files", lambda session_id=None: [])
    # ACTIVE_FILE_ID is a bare module-level global, not session-scoped —
    # a different test earlier in the full suite can leave it pointing at
    # a real file_id, which resolve_target_file then trusts as "the active
    # file" even with FILE_ORDER empty. Confirmed: this test passed alone
    # and failed only in the full run, resolving to a leaked 'test_clean_csv'.
    monkeypatch.setattr(app_state, "ACTIVE_FILE_ID", None)
    monkeypatch.setattr(app_state, "CURRENT_USER_PROMPT", "export the summary")
    out = export_data.invoke({"question": "export the summary", "format": "excel"})
    assert out == "No files uploaded yet."
