"""Live end-to-end demonstration of the exact STEP 7 scenario from the brief:
ingest 3 different real files (CSV, XLSX, PDF) and run 6 real chat() calls
through the actual dispatch pipeline — no mocks. Not a normal CI-style test
(the assertions ARE the report), run explicitly with -v -s to see the real
output:

    pytest tests/test_step7_live_scenario.py -v -s
"""
import os
import pytest

from app.ingestion.universal import universal_ingest
from app.query.dispatch import chat
from app.tables.helpers import get_all_real_tables

REAL_CSV = "uploads/data-file-3_0173_data-file-3.csv"
REAL_XLSX = "uploads/data-file-5_5177_data-file-5.xlsx"
REAL_PDF = "uploads/data-01_1067_data-01.pdf"


@pytest.fixture(scope="module")
def three_files():
    for p in (REAL_CSV, REAL_XLSX, REAL_PDF):
        if not os.path.exists(p):
            pytest.skip(f"fixture not found: {p}")
    universal_ingest(REAL_CSV, "step7_csv")
    universal_ingest(REAL_XLSX, "step7_xlsx")
    universal_ingest(REAL_PDF, "step7_pdf")
    return "step7_csv", "step7_xlsx", "step7_pdf"


def test_1_how_many_files_uploaded(three_files, capsys):
    result = chat("how many files have I uploaded", session_id="step7", file_id=None)
    with capsys.disabled():
        print(f"\n\n=== TEST 1: how many files have I uploaded ===\nintent={result['intent']}\n{result['response']}\n")
    assert result["intent"] == "list_files"
    for fid in three_files:
        assert fid in result["response"]


def test_2_sheet_names_from_excel_file(three_files, capsys):
    csv_fid, xlsx_fid, pdf_fid = three_files
    real_sheets = [str(t.attrs.get("page")) for t in get_all_real_tables(xlsx_fid)]
    result = chat("give me sheet names from the excel file", session_id="step7", file_id=None)
    with capsys.disabled():
        print(f"\n\n=== TEST 2: sheet names from the excel file (no file_id given) ===\n"
              f"intent={result['intent']}\n{result['response']}\n")
    assert result["intent"] == "data_query"
    core = "".join(c for c in result["response"] if c.isalnum()).lower()
    assert any("".join(c for c in n if c.isalnum()).lower() in core for n in real_sheets), (
        "None of the real XLSX sheet names appeared in the response")


def test_3_math_question(three_files, capsys):
    result = chat("what is 25 * 4", session_id="step7", file_id=None)
    with capsys.disabled():
        print(f"\n\n=== TEST 3: what is 25 * 4 ===\nintent={result['intent']}\n{result['response']}\n")
    assert result["intent"] == "general"
    assert "100" in result["response"]


def test_4_weather_question(three_files, capsys):
    result = chat("what is the weather today in Delhi", session_id="step7", file_id=None)
    with capsys.disabled():
        print(f"\n\n=== TEST 4: what is the weather today in Delhi ===\nintent={result['intent']}\n{result['response']}\n")
    assert result["intent"] == "general"
    lowered = result["response"].lower()
    assert "not in the uploaded document" not in lowered
    assert "uploaded document" not in lowered


def test_5_combine_csv_and_excel(three_files, capsys):
    csv_fid, xlsx_fid, pdf_fid = three_files
    real_csv_rows = len(get_all_real_tables(csv_fid)[0])
    csv_name = None
    from app import state
    csv_name = state.FILE_ORIGINAL_NAME[csv_fid]
    xlsx_name = state.FILE_ORIGINAL_NAME[xlsx_fid]
    prompt = f"combine data from {csv_name} and {xlsx_name} into one export as combined.csv"
    result = chat(prompt, session_id="step7", file_id=None)
    with capsys.disabled():
        print(f"\n\n=== TEST 5: combine {csv_name} and {xlsx_name} into one CSV export ===\n"
              f"intent={result['intent']} target_files={result.get('target_files')}\n{result['response']}\n")
    assert result["intent"] == "data_query"
    assert csv_fid in result.get("target_files", []) and xlsx_fid in result.get("target_files", [])


def test_6_pdf_only_question_does_not_pull_in_other_files(three_files, capsys):
    csv_fid, xlsx_fid, pdf_fid = three_files
    result = chat("what is this pdf document about, give me an overview",
                  session_id="step7", file_id=pdf_fid)
    with capsys.disabled():
        print(f"\n\n=== TEST 6: overview of the PDF (explicit file_id) ===\n"
              f"intent={result['intent']} file_id={result['file_id']}\n{result['response'][:400]}\n")
    assert result["file_id"] == pdf_fid
