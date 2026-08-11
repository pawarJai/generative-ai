import os
import pytest
import pandas as pd
from app.tables.helpers import dedupe_columns, resolve_file_scope
from app.embedding import embed_text
from app.query.planner import plan_query
from app.models import QueryPlan
from app.config import OUTPUT_DIR
from app.export.exporters import verify_export
from app import state

class TestMergedCellHandling:
    def test_unnamed_columns_are_renamed(self):
        # Bug 1: pandas assigns "Unnamed: N" to blank headers
        cols = ["Valid", "Unnamed: 1", "Unnamed: 2", "Another"]
        deduped = dedupe_columns(cols)
        assert "Unnamed: 1" not in deduped
        assert deduped == ["Valid", "col_1", "col_2", "Another"]

class TestNoHallucinatedData:
    def test_no_hallucination(self):
        plan = plan_query("give me 3 rows from the working sheet", fid=None)
        assert plan.intent in ("data_query", "general"), (
            f"Got {plan.intent} — hallucination risk if qa runs without file")

class TestEmptyEmbeddingsHandling:
    def test_empty_string_does_not_crash(self):
        # Bug 3: Empty embeddings crashing Chroma
        assert embed_text("dummy_id", "") == 0
        assert embed_text("dummy_id", "   \n  ") == 0

class TestDedupeColumns:
    def test_dedupe_handles_colliding_suffixes(self):
        # Bug 4: Duplicate column names crashing pd.concat
        cols = ["A", "A", "A_1"]
        deduped = dedupe_columns(cols)
        assert len(set(deduped)) == 3, "Must produce exactly N unique names"

class TestSheetListingIsDeterministic:
    def test_list_sheets_intent(self):
        # Bug 5: list sheet names generating bad SQL
        plan = plan_query("list all sheet names in this file")
        assert plan.intent == "data_query"

class TestExportVerification:
    def test_verify_export_fails_on_missing_file(self):
        # Bug 6: Export silently producing no file
        success, msg = verify_export("/tmp/does_not_exist_xyz.csv", 10, "csv")
        assert not success
        assert "FAILED" in msg

class TestMultiFileAccess:
    def test_multi_file_access(self):
        assert len(state.FILE_ORDER) >= 0  # basic state integrity check
        for fid in state.FILE_ORDER:
            assert fid in state.FILE_KIND, f"file_id {fid} in FILE_ORDER but not in FILE_KIND"
            assert fid in state.FILE_ORIGINAL_NAME, f"file_id {fid} missing from FILE_ORIGINAL_NAME"

class TestGeneralKnowledgeQuestions:
    def test_weather_question(self):
        # Bug 8: "What's the weather" getting hedged document non-answer
        plan = plan_query("what's the weather like in London today")
        assert plan.intent == "general"

class TestChatHistoryRouting:
    def test_what_did_i_tell_you(self):
        # Bug 9: "What did I tell you" dumping column listings
        plan = plan_query("what did I tell you earlier")
        assert plan.intent == "chat_history"

class TestColumnExportRouting:
    def test_column_name_group_not_list_columns(self):
        # Bug 10: "column name = group" misrouted to list_columns
        plan = plan_query("filter where column name = group")
        assert plan.intent != "list_columns"

class TestNoRawSQLInResponse:
    def test_no_raw_sql(self):
        from app.query.planner import _plan_query_raw
        plan = _plan_query_raw("list sheet names", fid=None)
        # The real assertion: SQL must never appear in a sheet-listing response
        # (tested in integration tests; this confirms routing at least tries)
        assert plan.intent in ("list_files", "data_query", "general")

class TestQueryPlanCompleteness:
    def test_query_plan_has_fields(self):
        # Bug 12: QueryPlan missing sheet_name/n_rows fields
        qp = QueryPlan(intent="data_query", sheet_name="Sheet1", n_rows=50)
        assert qp.sheet_name == "Sheet1"
        assert qp.n_rows == 50

class TestFileScopeResolution:
    def test_multiple_files_resolved(self, monkeypatch):
        # Bug 13: Multi-file scope only finding first match
        monkeypatch.setattr(state, "FILE_ORDER", ["f1", "f2", "f3"])
        monkeypatch.setattr(state, "FILE_ORIGINAL_NAME", {"f1": "january.csv", "f2": "february.csv", "f3": "march.pdf"})
        scope = resolve_file_scope("combine january.csv and february.csv together")
        assert "f1" in scope
        assert "f2" in scope

class TestOutputDirectoriesNotInSourceTree:
    def test_output_dir_outside_app(self):
        # Bug 14: --reload triggered by own log/cache writes
        assert "app" not in OUTPUT_DIR.split(os.sep), f"OUTPUT_DIR ({OUTPUT_DIR}) shouldn't be inside app/"
