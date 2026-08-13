"""Tests for complex real-world Excel structures that the system must
handle without crashing or producing garbage output.

Based on the actual data-file-5 xlsx which contains:
- Cover Sheet: merged cells, logo, mixed form/table content
- Terms: multi-column terms with numbered items
- Sheet1: pivot-table style with Row Labels / Column Labels
- Working Sheet: 74 columns, real valve schedule data
- RFQ-Authorities: 2-column key-value lookup table
- Enquiry amendment form: mostly empty form with header block
"""
import os
import pytest
import pandas as pd
from app.ingestion.universal import universal_ingest
from app.tables.helpers import get_all_real_tables
from app.query.dispatch_legacy import chat

REAL_XLSX = "uploads/data-file-5_9451_data-file-5.xlsx"

@pytest.fixture(scope="module")
def complex_xlsx():
    if not os.path.exists(REAL_XLSX):
        pytest.skip(f"fixture not found: {REAL_XLSX}")
    universal_ingest(REAL_XLSX, "complex_xlsx_test")
    return "complex_xlsx_test"

def test_ingestion_does_not_crash_on_complex_xlsx(complex_xlsx):
    """The xlsx must ingest without exception regardless of structure."""
    tables = get_all_real_tables(complex_xlsx)
    assert len(tables) > 0

def test_working_sheet_has_tag_no_column(complex_xlsx):
    """Working Sheet is the most important table — must have Tag No.

    Every sheet produces two tables: a "_Raw" fallback (always generic
    col_0..col_N names, by design — see tabular_ingest.py) and a real
    header-detected table. Must check the header-detected one, not
    whichever "working"-named table happens to come first."""
    tables = get_all_real_tables(complex_xlsx)
    working = [t for t in tables
               if "working" in str(t.attrs.get("page","")).lower()
               and not str(t.attrs.get("page","")).endswith("_Raw")]
    assert len(working) > 0, "No header-detected Working Sheet found"
    cols_lower = [str(c).lower() for c in working[0].columns]
    assert any("tag" in c for c in cols_lower), (
        f"Tag No. column missing from Working Sheet. Columns: {working[0].columns.tolist()}")

def test_rfq_authorities_readable_as_key_value(complex_xlsx):
    """RFQ-Authorities is a 2-column key-value sheet — must be queryable."""
    result = chat("what is the total quantity from RFQ authorities sheet",
                   session_id="complex_test1", file_id=complex_xlsx)
    assert "2334" in result["response"] or result.get("table") is not None, (
        f"RFQ-Authorities data not readable: {result['response'][:200]}")

def test_pivot_sheet_does_not_produce_unnamed_columns(complex_xlsx):
    """Sheet1 is pivot-style — must not produce Unnamed: N columns."""
    tables = get_all_real_tables(complex_xlsx)
    sheet1 = [t for t in tables if "sheet1" in str(t.attrs.get("page","")).lower()]
    for t in sheet1:
        unnamed = [c for c in t.columns if str(c).startswith("Unnamed")]
        assert len(unnamed) == 0, (
            f"Sheet1 has {len(unnamed)} Unnamed columns — header detection failed")

def test_empty_form_sheet_does_not_crash(complex_xlsx):
    """Enquiry amendment form is mostly empty — must not crash, just
    return empty/minimal result."""
    tables = get_all_real_tables(complex_xlsx)
    # Should not raise, even if it finds no usable tables in that sheet
    assert tables is not None

def test_data_query_on_working_sheet_returns_real_values(complex_xlsx):
    """data_query against Working Sheet must return real Tag No values,
    not fabricated ones. 7210-V123 is a known real value."""
    result = chat("show me first 3 rows of the working sheet",
                   session_id="complex_test2", file_id=complex_xlsx)
    response = result["response"]
    # Must NOT contain known hallucinated values from earlier bugs
    assert "YD001" not in response, "Hallucinated Yard No. value returned"
    assert "Gate Valve" not in response or "7210" in response, (
        "Suspicious: 'Gate Valve' without any real Tag No.")

def test_cover_sheet_company_name_extractable(complex_xlsx):
    """Cover Sheet has 'QUEST FLOW CONTROLS LTD' — must be findable."""
    result = chat("what is the company name in this file",
                   session_id="complex_test3", file_id=complex_xlsx)
    assert "quest" in result["response"].lower() or "flow" in result["response"].lower(), (
        f"Company name not found: {result['response'][:200]}")

def test_total_quantity_from_rfq_sheet(complex_xlsx):
    """Total Quantity = 2334, directly visible in RFQ-Authorities sheet."""
    result = chat("what is the total quantity",
                   session_id="complex_test4", file_id=complex_xlsx)
    assert "2334" in result["response"], (
        f"Total quantity 2334 not found: {result['response'][:200]}")

def test_terms_sheet_delivery_info_readable(complex_xlsx):
    """Terms sheet has delivery info: '20-22 Weeks From Receipt of
    Approval' — must be findable via semantic search."""
    result = chat("what are the delivery terms",
                   session_id="complex_test5", file_id=complex_xlsx)
    assert "week" in result["response"].lower() or "delivery" in result["response"].lower(), (
        f"Delivery terms not found: {result['response'][:200]}")
