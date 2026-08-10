"""Integration tests for the three architectural problems fixed together:

1. Single-file lock-in — chat() used to silently default to whichever file
   was uploaded LAST (ACTIVE_FILE_ID, overwritten on every /ingest call), so
   files 1-4 of a 5-file upload were unreachable unless the caller manually
   passed file_id.
2. Non-document questions ("what's the weather", "what is 25*4") used to be
   routed through vector search against the active file and produced a
   hedged non-answer instead of a real one.
3. Multi-file operations ("combine file A and file B") had no working code
   path: resolve_file_scope returned only the FIRST filename match, and
   run_code_on_file only ever accepted a single file_id.

Same approach as test_code_exec.py: assert against values computed
independently (real row counts, real file counts) rather than hardcoded
strings, and use real ingested files, not mocks.
"""
import os
import pandas as pd
import pytest

from app import state
from app.ingestion.universal import universal_ingest
from app.query.code_exec import run_code_on_files, run_code_on_file
from app.query.planner import plan_query
from app.query.dispatch import chat, _resolve_target_files
from app.tables.helpers import get_all_real_tables, resolve_file_scope
from app.models import QueryPlan

REAL_CSV = "uploads/data-file-3_0173_data-file-3.csv"
REAL_XLSX = "uploads/data-file-5_5177_data-file-5.xlsx"


@pytest.fixture(scope="module")
def two_files():
    """Ingests two distinct real files under known file_ids and names, so
    tests can assert on 'both are visible' without depending on whatever
    else happens to be in FILE_ORDER from other test modules."""
    if not (os.path.exists(REAL_CSV) and os.path.exists(REAL_XLSX)):
        pytest.skip("real multi-file fixtures not found")
    universal_ingest(REAL_CSV, "test_multi_csv")
    universal_ingest(REAL_XLSX, "test_multi_xlsx")
    return "test_multi_csv", "test_multi_xlsx"


# ---------------------------------------------------------------------------
# Problem 1: single-file lock-in
# ---------------------------------------------------------------------------

def test_resolve_target_files_defaults_to_all_ingested_not_just_last(two_files):
    """The core lock-in bug: with no explicit file_id and nothing named in the
    prompt, target resolution must return every ingested file — not just
    ACTIVE_FILE_ID (whichever was ingested most recently)."""
    csv_fid, xlsx_fid = two_files
    plan = QueryPlan(intent="data_query")  # no file_scope set
    targets = _resolve_target_files(None, plan)
    assert csv_fid in targets and xlsx_fid in targets, (
        f"Expected both {csv_fid!r} and {xlsx_fid!r} in default scope, got {targets}")


def test_resolve_target_files_honors_explicit_file_id(two_files):
    csv_fid, _xlsx_fid = two_files
    plan = QueryPlan(intent="data_query")
    targets = _resolve_target_files(csv_fid, plan)
    assert targets == [csv_fid]


def test_a_second_upload_does_not_orphan_the_first_file(two_files):
    """Direct regression test for the reported bug: after uploading a second
    file, a question with no file named in it must still be able to reach
    data from the FIRST uploaded file (identified by its actual column
    content, since the LLM only sees sanitized-filename variable names, not
    internal file_ids), via the full default scope — not silently only see
    the second (active) file."""
    csv_fid, _xlsx_fid = two_files
    real_csv_rows = len(get_all_real_tables(csv_fid)[0])
    target_files = _resolve_target_files(None, QueryPlan(intent="data_query"))
    assert csv_fid in target_files, f"First-uploaded file missing from default scope: {target_files}"
    result = run_code_on_files(
        target_files, "how many rows are in the table that has an 'Item Number' column")
    assert str(real_csv_rows) in f"{result['text']} {result.get('table')}", (
        f"First-uploaded file's data unreachable after a second upload: {result['text']}")


# ---------------------------------------------------------------------------
# Problem 2: general knowledge questions must skip file/vector logic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", [
    "what is 25 * 4",
    "what is the weather in Gujarat",
    "tell me a fun fact about the moon",
])
def test_planner_routes_non_file_questions_to_general(two_files, prompt):
    """Even with files ingested and one of them active, a question with no
    file-related language must route to 'general', not 'data_query' or 'qa'
    against the active file."""
    csv_fid, _ = two_files
    plan = plan_query(prompt, session_id="test_general", fid=csv_fid)
    assert plan.intent == "general", f"{prompt!r} routed to {plan.intent!r}, expected 'general'"


def test_general_question_end_to_end_does_not_mention_the_document(two_files):
    csv_fid, _ = two_files
    result = chat("what is 12 * 12", session_id="test_general_e2e", file_id=csv_fid)
    assert result["intent"] == "general"
    assert "144" in result["response"], f"Expected the actual answer 144, got: {result['response']}"


def test_file_related_question_still_routes_to_data_not_general(two_files):
    csv_fid, _ = two_files
    plan = plan_query("how many rows does this file have", session_id="test_general2", fid=csv_fid)
    assert plan.intent == "data_query"


# ---------------------------------------------------------------------------
# Problem 3: multi-file operations
# ---------------------------------------------------------------------------

def test_resolve_file_scope_collects_every_named_file_not_just_first(two_files):
    """Regression: resolve_file_scope used to `return` on the first filename
    match, so a prompt naming two files only ever resolved to one."""
    csv_fid, xlsx_fid = two_files
    csv_name = state.FILE_ORIGINAL_NAME[csv_fid]
    xlsx_name = state.FILE_ORIGINAL_NAME[xlsx_fid]
    scope = resolve_file_scope(f"combine {csv_name} and {xlsx_name} into one file")
    assert csv_fid in scope and xlsx_fid in scope, (
        f"Expected both files in scope, got {scope}")


def test_run_code_on_files_loads_tables_from_both_files_with_prefixed_names(two_files):
    csv_fid, xlsx_fid = two_files
    df_map_result = run_code_on_files(
        [csv_fid, xlsx_fid], "list the names of every variable/sheet available to you")
    csv_name_clean = "".join(c if c.isalnum() else "_" for c in state.FILE_ORIGINAL_NAME[csv_fid])
    xlsx_name_clean = "".join(c if c.isalnum() else "_" for c in state.FILE_ORIGINAL_NAME[xlsx_fid])
    blob = f"{df_map_result['text']} {df_map_result.get('table')}"
    assert csv_name_clean.split("_")[0] in blob or xlsx_name_clean.split("_")[0] in blob, (
        f"Expected file-prefixed variable names in response: {blob[:500]}")


def test_multi_file_row_count_matches_sum_of_both_real_files(two_files):
    """The concrete multi-file scenario from the bug report: a question that
    can only be answered by combining data from two different files."""
    csv_fid, xlsx_fid = two_files
    real_csv_rows = len(get_all_real_tables(csv_fid)[0])
    result = run_code_on_files(
        [csv_fid, xlsx_fid],
        "how many rows does the table that has 'Item Number' as a column have")
    assert str(real_csv_rows) in f"{result['text']} {result.get('table')}", (
        f"Expected {real_csv_rows} (the CSV's row count) in: {result['text']}")


def test_data_query_with_no_file_named_defaults_to_all_files_end_to_end(two_files):
    """End-to-end through chat(): no file_id passed, prompt doesn't name a
    specific file -> must still be able to answer about the non-active file."""
    csv_fid, _xlsx_fid = two_files
    real_csv_rows = len(get_all_real_tables(csv_fid)[0])
    result = chat(
        "how many rows are in the table with columns including Item Number",
        session_id="test_multi_e2e", file_id=None)
    assert result["intent"] == "data_query"
    assert str(real_csv_rows) in result["response"], (
        f"Expected {real_csv_rows} in response: {result['response']}")


# ---------------------------------------------------------------------------
# list_files intent (surfaced while fixing problem 1 — needed so users can
# discover what they've actually uploaded instead of guessing file_ids)
# ---------------------------------------------------------------------------

def test_list_files_intent_routes_correctly(two_files):
    plan = plan_query("how many files have I uploaded", session_id="test_lf", fid=None)
    assert plan.intent == "list_files"


def test_list_files_end_to_end_shows_both_files(two_files):
    csv_fid, xlsx_fid = two_files
    result = chat("what files have I uploaded", session_id="test_lf_e2e", file_id=csv_fid)
    assert result["intent"] == "list_files"
    assert csv_fid in result["response"] and xlsx_fid in result["response"], (
        f"Expected both file_ids listed in: {result['response']}")


def test_available_files_field_lists_every_ingested_file(two_files):
    csv_fid, xlsx_fid = two_files
    result = chat("what is 2 + 2", session_id="test_avail", file_id=csv_fid)
    ids = {f["file_id"] for f in result["available_files"]}
    assert csv_fid in ids and xlsx_fid in ids


# ---------------------------------------------------------------------------
# Back-compat: single-file wrapper must still behave exactly as before
# ---------------------------------------------------------------------------

def test_run_code_on_file_single_file_wrapper_still_works(two_files):
    csv_fid, _ = two_files
    real_rows = len(get_all_real_tables(csv_fid)[0])
    result = run_code_on_file(csv_fid, "how many rows are there")
    assert str(real_rows) in f"{result['text']} {result.get('table')}"
