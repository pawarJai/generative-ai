"""Tests for writing document context into an export, and for changing an
export that already exists.

Both capabilities trace to the same three turns in interaction_log.jsonl on
2026-08-12, where the user asked for the tender's header block to be added to
a sheet, supplied every value by hand, and was told the request was ambiguous —
because no exporter could write above a header row and no tool could touch a
file that already existed.
"""
import os

import pandas as pd
import pytest
from openpyxl import load_workbook

from app import state
from app.config import OUTPUT_DIR
from app.export.exporters import band_offset, export_excel, verify_export
from app.export.modify import apply_ops, modify, parse_instruction
from app.models import QueryPlan
from app.tables.context import TableContext

BAND = TableContext(
    document=["ACME SHIPYARD LTD", "PROJECT -NGMV", "Spec. No. N453105 REV.00"],
    section="1. ITEM AND QUANTITY REQUIRED:",
    footer=["UNCLASSIFIED"],
    source="Source: data-file-1.pdf — pages 8-10",
)


def _frame(rows=6, with_context=True):
    df = pd.DataFrame({
        "SL no": [str(i + 1) for i in range(rows)],
        "Item": [f"{20 + i}NB PN10 GLOBE VALVE" for i in range(rows)],
        "Spec No.": ["A.1"] * rows,
        "Quantity": [str(6 * (i + 1)) for i in range(rows)],
    })
    if with_context:
        df.attrs["context"] = BAND
    return df


def _write(name, df, **plan_kwargs):
    plan = QueryPlan(intent="export", sink="excel", filename=name, **plan_kwargs)
    message = export_excel("test_file", plan, tables=[df])
    return os.path.join(OUTPUT_DIR, name), message


# --- Step 2: the band reaches the file ---------------------------------------

def test_band_is_written_above_the_header_row():
    path, message = _write("t-band.xlsx", _frame())
    sheet = load_workbook(path).active

    written = [sheet.cell(row=r, column=1).value for r in range(1, 6)]
    assert written[0] == "ACME SHIPYARD LTD"
    assert "1. ITEM AND QUANTITY REQUIRED:" in written
    # The table itself is untouched: header row intact, rows counted honestly.
    assert "6 rows" in message
    body = pd.read_excel(path, skiprows=band_offset(path))
    assert list(body.columns) == ["SL no", "Item", "Spec No.", "Quantity"]
    assert len(body) == 6


def test_band_rows_are_never_counted_as_data():
    """A band must not be able to disguise a short export as a complete one."""
    path, _ = _write("t-short.xlsx", _frame(3))
    ok, message = verify_export(path, 60, "excel")

    assert not ok
    assert "3 rows" in message


def test_no_context_writes_a_bare_table():
    path, _ = _write("t-bare.xlsx", _frame(), no_context=True)

    assert band_offset(path) == 0
    assert list(pd.read_excel(path).columns)[0] == "SL no"


def test_a_table_with_no_context_is_unchanged():
    """Documents without a letterhead must export exactly as they did before."""
    path, _ = _write("t-none.xlsx", _frame(with_context=False))

    assert band_offset(path) == 0


def test_band_offset_is_detected_not_declared():
    """Every reader of an exported workbook has to agree where the table
    starts, including the callers that re-verify after the exporter already
    did — so the offset is read off the sheet, not passed around."""
    with_band, _ = _write("t-off1.xlsx", _frame())
    without, _ = _write("t-off2.xlsx", _frame(), no_context=True)

    assert band_offset(with_band) == len(BAND.styled()) + 1
    assert band_offset(without) == 0


# --- Step 3: changing a file that already exists ------------------------------

def test_the_prompt_that_failed_three_times_now_works():
    """Verbatim from interaction_log.jsonl, id=b39decb0."""
    path, _ = _write("t-mod.xlsx", _frame(), no_context=True)
    state.WRITTEN_EXPORTS["t-mod.xlsx"]["context"] = BAND
    assert band_offset(path) == 0

    result = modify("t-mod.xlsx", "we need to add this Full header section data "
                                  "also in this excel top section we need to add "
                                  "this thing and other make as it is as per last "
                                  "given file")

    assert "✓" in result
    assert band_offset(path) > 0
    assert load_workbook(path).active.cell(row=1, column=1).value == "ACME SHIPYARD LTD"
    assert len(pd.read_excel(path, skiprows=band_offset(path))) == 6


def test_renaming_the_first_two_columns():
    path, _ = _write("t-ren.xlsx", _frame())

    modify("t-ren.xlsx", "rename the first two columns to Sr. No. and Description")
    columns = list(pd.read_excel(path, skiprows=band_offset(path)).columns)

    assert columns == ["Sr. No.", "Description", "Spec No.", "Quantity"]


def test_a_column_name_ending_in_a_period_can_be_renamed():
    """'Spec No.' lost its full stop to name-cleaning, so the lookup missed
    and the rename silently did nothing while reporting success."""
    path, _ = _write("t-dot.xlsx", _frame())

    result = modify("t-dot.xlsx", "rename 'Spec No.' to Drawing Ref")

    assert "Drawing Ref" in list(pd.read_excel(path, skiprows=band_offset(path)).columns)
    assert "no column named" not in result


def test_dropping_a_column():
    path, _ = _write("t-drop.xlsx", _frame())

    modify("t-drop.xlsx", "remove the Quantity column")

    assert "Quantity" not in list(pd.read_excel(path, skiprows=band_offset(path)).columns)


def test_removing_the_band_leaves_the_rows_alone():
    path, _ = _write("t-unband.xlsx", _frame())

    modify("t-unband.xlsx", "remove the header section, i want only the table")

    assert band_offset(path) == 0
    assert len(pd.read_excel(path)) == 6


def test_a_rename_and_a_band_request_in_one_sentence():
    path, _ = _write("t-both.xlsx", _frame(), no_context=True)
    state.WRITTEN_EXPORTS["t-both.xlsx"]["context"] = BAND

    modify("t-both.xlsx", "add the header details and drop the Spec No. column")
    body = pd.read_excel(path, skiprows=band_offset(path))

    assert band_offset(path) > 0
    assert "Spec No." not in list(body.columns)
    assert len(body) == 6


def test_an_unknown_column_is_reported_not_invented():
    path, _ = _write("t-unknown.xlsx", _frame())

    result = modify("t-unknown.xlsx", "rename 'Widget' to Gadget")

    assert "no column named" in result
    assert list(pd.read_excel(path, skiprows=band_offset(path)).columns) == \
        ["SL no", "Item", "Spec No.", "Quantity"]


def test_a_missing_file_is_not_silently_created():
    result = modify("does-not-exist.xlsx", "add the header")

    assert "no file called" in result
    assert not os.path.exists(os.path.join(OUTPUT_DIR, "does-not-exist.xlsx"))


def test_an_instruction_with_no_operation_asks_instead_of_guessing():
    _write("t-vague.xlsx", _frame())

    result = modify("t-vague.xlsx", "make it look better")

    assert "rename columns" in result


def test_adding_a_band_we_cannot_authenticate_is_refused():
    """Better to say the header cannot be recovered than to invent one."""
    _write("t-noctx.xlsx", _frame(with_context=False))
    state.WRITTEN_EXPORTS["t-noctx.xlsx"] = {"file_id": None, "context": None,
                                             "pages": None, "sheets": 1}

    result = modify("t-noctx.xlsx", "add the header details")

    assert "nothing authentic" in result


# --- instruction parsing ------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("add the header details", True),
    ("i need the company name and project on top", True),
    ("remove the header", False),
    ("just the table please, no title block", False),
    ("rename the first column to X", None),
])
def test_band_intent_is_read_from_the_users_words(text, expected):
    assert parse_instruction(text)["context"] is expected


def test_a_column_name_is_not_a_request_about_the_header():
    """'Spec No.' contains a word this parser also uses for the letterhead."""
    ops = parse_instruction("rename 'Spec No.' to Drawing Ref")

    assert ops["context"] is None
    assert ops["rename"] == {"Spec No.": "Drawing Ref"}


def test_positional_rename_counts_words_and_digits():
    assert parse_instruction("rename the first 2 columns to A and B")["positional_rename"] == ["A", "B"]
    assert parse_instruction("rename the first three columns to A, B and C")["positional_rename"] == ["A", "B", "C"]


def test_apply_ops_never_touches_rows():
    df = _frame(5)
    out, changes = apply_ops(df, parse_instruction("rename the first column to X"))

    assert len(out) == len(df)
    assert changes
