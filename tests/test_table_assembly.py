"""Tests for assembling one table out of fragments from several pages/files.

Every case here is taken from a real defect in outputs/f1-66.xlsx, an export
the agent reported as "65 rows, 5 columns, export complete" while it was
misaligned from row 19 onward. The synthetic frames reproduce the exact
shapes Docling produced for pages 6-10 of data-file-2: a named header page,
continuation pages with no names at all, and a page whose merged left column
was dropped.
"""
import json
import os

import pandas as pd
import pytest

from app.tables.assembly import (assemble, align_columns, column_key,
                                 drop_label_rows, signature)

ADDRESS = "Cochin Shipyard Limited, Perumanoor, Kochi-682304 682015"
OFFICER = "Dijeev D"


def _named_page(n=6):
    """A header page: five real column names, prose/constant/numeric values."""
    return pd.DataFrame({
        "Evaluation Schedules": ["Foot Valve"] + ["Drain Valves"] * (n - 1),
        "Item/Category": [f"{20 + i}nb Pn10 En Ff Nab_nab St Valve With Lever"
                          for i in range(n)],
        "Consignee/Reporting Officer": [OFFICER] * n,
        "Consignee Address": [ADDRESS] * n,
        "Quantity": [str(6 * (i + 1)) for i in range(n)],
    })


def _continuation_page(n=6):
    """A continuation page: no header at all, and the merged left column
    dropped by Docling — four columns against the reference's five."""
    return pd.DataFrame({
        "col_0": [f"{50 + i}nb Pn10 En Ff Nab_nab_nab Gate Valve Handwheel"
                  for i in range(n)],
        "col_1": [OFFICER] * n,
        "col_2": [ADDRESS] * n,
        "col_3": [str(12 * (i + 1)) for i in range(n)],
    })


def test_dropped_column_becomes_a_gap_not_a_shift():
    """The core defect: a 4-wide continuation page aligned to a 5-name header
    put the item text under 'Evaluation Schedules', the officer under
    'Item/Category' and the quantity under 'Consignee Address'."""
    out, report = assemble([_named_page(), _continuation_page()])

    assert list(out.columns) == list(_named_page().columns)
    assert out["Quantity"].str.fullmatch(r"\d+").all()
    assert (out["Consignee Address"] == ADDRESS).all()
    assert (out["Consignee/Reporting Officer"] == OFFICER).all()
    # The dropped column is empty for the continuation rows — honestly blank,
    # never back-filled with a neighbouring group's label.
    assert out["Evaluation Schedules"].tail(6).isna().all()
    assert report.clean


def test_same_header_with_different_spacing_is_one_column():
    """'Consignee/Reporting Officer' and 'Consignee / Reporting Officer'
    became two half-empty columns in the shipped file."""
    a = _named_page()
    b = _named_page()
    b.columns = ["Evaluation Schedules", "Item/Category",
                 "Consignee / Reporting Officer", "Consignee Address", "Quantity"]
    out, _ = assemble([a, b])

    assert len(out.columns) == 5
    assert out["Consignee/Reporting Officer"].notna().all()


def test_split_words_in_headers_match():
    assert column_key("Q ua nti ty") == column_key("Quantity")
    assert column_key("Evaluat ion Schedu les") == column_key("Evaluation Schedules")


def test_positional_label_row_is_dropped():
    """Docling's column indices were written into the sheet as the data row
    ['0','1','','3','','2']."""
    df = _named_page(3)
    df.loc[len(df)] = ["0", "1", "2", "3", "4"]
    cleaned, dropped = drop_label_rows(df)

    assert dropped == 1
    assert len(cleaned) == 3


def test_header_echo_row_is_dropped():
    df = _named_page(3)
    df.loc[len(df)] = list(df.columns)
    cleaned, dropped = drop_label_rows(df)

    assert dropped == 1
    assert len(cleaned) == 3


def test_genuine_numeric_row_is_kept():
    """A row of real numbers must survive — only consecutive-from-zero index
    rows are labels."""
    df = pd.DataFrame({"a": ["10"], "b": ["25"], "c": ["7"]})
    cleaned, dropped = drop_label_rows(df)

    assert dropped == 0
    assert len(cleaned) == 1


def test_unmatched_column_is_kept_and_reported():
    """A column with no counterpart is never folded into a neighbour."""
    extra = _continuation_page()
    extra["col_4"] = ["2026-01-0%d" % (i + 1) for i in range(len(extra))]
    out, report = assemble([_named_page(), extra])

    assert not report.clean
    assert len(out) == 12
    assert (out["Consignee Address"].dropna() == ADDRESS).all()


def test_column_order_cannot_cross():
    """Alignment is monotone: matching must never reorder columns."""
    ref = [(c, signature(_named_page()[c])) for c in _named_page().columns]
    cand_df = _continuation_page()
    cand = [(c, signature(cand_df[c])) for c in cand_df.columns]
    mapping = align_columns(ref, cand)

    matched = [j for j in mapping if j is not None]
    assert matched == sorted(matched)


def test_empty_input_is_handled():
    out, report = assemble([])
    assert out is None
    assert report.rows_out == 0


def test_page_ranges_expand():
    """'page number 6 to 10' meant [6, 10] — so a five-page table exported as
    one page, and the export reported success."""
    from app.graph.agent import _extract_requested_pages

    assert _extract_requested_pages("page number 6 to 10 we have table") == [6, 7, 8, 9, 10]
    assert _extract_requested_pages("table on page 6-10") == [6, 7, 8, 9, 10]
    assert _extract_requested_pages("page 7 through 9") == [7, 8, 9]


def test_discrete_page_lists_do_not_expand():
    """'page 6 and 7' names two pages, not a range — and must stay two."""
    from app.graph.agent import _extract_requested_pages

    assert _extract_requested_pages("page 6 and 7 how many columns") == [6, 7]
    assert _extract_requested_pages("page 12") == [12]


def test_absurd_page_span_is_not_expanded():
    """A misparsed number must not generate thousands of pages."""
    from app.graph.agent import _extract_requested_pages

    assert _extract_requested_pages("page 2 to 2000") == [2, 2000]


CACHE = "docling_cache/1650a91c2ca11de0.json"


@pytest.mark.skipif(not os.path.exists(CACHE), reason="docling cache not present")
def test_real_tender_pages_6_to_10_assemble_correctly():
    """The exact export that shipped broken: pages 6-10 of data-file-2.

    Ground truth from the PDF — 64 body rows (17+17+17+13), five columns,
    every quantity numeric, every address the Cochin Shipyard consignee.
    """
    from docling_core.types.doc.document import DoclingDocument
    from app import state
    from app.tables.helpers import get_all_real_tables

    with open(CACHE) as fh:
        state.DOCLING_DOCS["assembly_test"] = DoclingDocument.model_validate(json.load(fh))
    state.FILE_KIND["assembly_test"] = "docling"

    frames = [df for df in get_all_real_tables("assembly_test")
              if df.attrs.get("page") in (6, 7, 8, 9, 10)]
    out, report = assemble(frames)

    assert len(out) == 64
    assert len(out.columns) == 5
    assert out["Quantity"].astype(str).str.replace(" ", "").str.fullmatch(r"\d+").all()
    assert out["Consignee Address"].astype(str).str.contains("Cochin Shipyard").all()
    assert report.rows_dropped == 0


@pytest.mark.skipif(not os.path.exists(CACHE), reason="docling cache not present")
def test_merged_group_labels_match_the_tenders_own_summary():
    """Every recovered group total must equal the RFQ-Authorities sheet.

    This is the strongest check available: those totals come from a different
    document (data-file-5.xlsx), written by the buyer, so agreeing with them
    cannot be an artefact of our own extraction. An earlier version of this
    code reported Globe Valve 46 rows / Gate Valve none at all, because a
    merged label was forward-filled across a page break.
    """
    from docling_core.types.doc.document import DoclingDocument
    from app import state
    from app.tables.helpers import assemble_pages
    from app.persistence import get_file

    if not get_file("data-file-2_0060"):
        pytest.skip("data-file-2 is not registered in this environment")

    with open(CACHE) as fh:
        state.DOCLING_DOCS["data-file-2_0060"] = DoclingDocument.model_validate(json.load(fh))
    state.FILE_KIND["data-file-2_0060"] = "docling"

    out, _report, _pages = assemble_pages("data-file-2_0060", [6, 7, 8, 9, 10])
    quantity = (out["Quantity"].astype(str)
                .str.replace(r"\D", "", regex=True).replace("", "0").astype(int))
    totals = out.assign(q=quantity).groupby("Evaluation Schedules")["q"].sum().to_dict()

    assert totals == {
        "Foot Valve": 72,
        "Drain Valves": 24,
        "Globe Valve": 228,
        "Gate Valve": 192,
        "Swing Check Valve": 306,
        "Ball Valve": 840,
        "Butterfly Valves": 672,
    }
