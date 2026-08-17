"""Tool-call logging: every tool the agent calls in a turn, with the exact
arguments it passed and what the tool returned, alongside the user's own
prompt in the same log line.

Added because every export bug found in this project's history so far
("called export_data without file_id on a retry", "the model never called
modify_export at all despite claiming it had") could only be confirmed by
re-running the live server and reading the exported file's bytes back by
hand — this puts that first check directly into interaction_log.jsonl
instead of requiring a live reproduction every time.
"""
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graph import agent as A


def _thread_id(label):
    return f"pytest-toollog-{label}-{uuid.uuid4().hex[:8]}"


def test_tool_calls_are_logged_with_their_arguments_and_result(monkeypatch):
    def fake_invoke(payload, config=None):
        return {"messages": [
            HumanMessage(content="export it"),
            AIMessage(content="", tool_calls=[
                {"id": "call_1", "name": "export_data",
                 "args": {"question": "export it", "format": "excel",
                          "filename": "w1.xlsx"}}]),
            ToolMessage(content="Verified: wrote 5 rows to w1.xlsx",
                       tool_call_id="call_1", name="export_data"),
            AIMessage(content="Done — 5 rows exported to w1.xlsx."),
        ]}

    monkeypatch.setattr(A.agent, "invoke", fake_invoke)
    monkeypatch.setattr(A, "_restore_file_if_needed", lambda fid: None)

    captured = {}

    def fake_log_interaction(session_id, file_id, prompt, plan, response,
                             success, error=None, latency=None,
                             tool_calls=None):
        captured["tool_calls"] = tool_calls
        captured["prompt"] = prompt
        return "fake-log-id"

    monkeypatch.setattr("app.logging_utils.log_interaction", fake_log_interaction)

    A.run_agent("export it", session_id=_thread_id("basic"), file_id=None)

    assert captured["prompt"] == "export it"
    assert captured["tool_calls"] == [{
        "tool": "export_data",
        "args": {"question": "export it", "format": "excel",
                 "filename": "w1.xlsx"},
        "result": "Verified: wrote 5 rows to w1.xlsx",
    }]


def test_multiple_tool_calls_in_one_turn_are_all_logged_in_order(monkeypatch):
    """The exact shape of the bug this was built to catch: an export call
    followed by a second, separate tool call in the same turn — visible
    here as two entries instead of requiring a live re-run to notice one
    never happened."""
    def fake_invoke(payload, config=None):
        return {"messages": [
            HumanMessage(content="export and rename"),
            AIMessage(content="", tool_calls=[
                {"id": "call_1", "name": "export_data",
                 "args": {"question": "export and rename", "filename": "w1.xlsx"}}]),
            ToolMessage(content="Verified: wrote 5 rows to w1.xlsx",
                       tool_call_id="call_1", name="export_data"),
            AIMessage(content="", tool_calls=[
                {"id": "call_2", "name": "modify_export",
                 "args": {"filename": "w1.xlsx", "change": "rename X to Y"}}]),
            ToolMessage(content="Verified: 'X' -> 'Y'.",
                       tool_call_id="call_2", name="modify_export"),
            AIMessage(content="Done."),
        ]}

    monkeypatch.setattr(A.agent, "invoke", fake_invoke)
    monkeypatch.setattr(A, "_restore_file_if_needed", lambda fid: None)

    captured = {}

    def fake_log_interaction(session_id, file_id, prompt, plan, response,
                             success, error=None, latency=None,
                             tool_calls=None):
        captured["tool_calls"] = tool_calls
        return "fake-log-id"

    monkeypatch.setattr("app.logging_utils.log_interaction", fake_log_interaction)

    A.run_agent("export and rename", session_id=_thread_id("multi"), file_id=None)

    assert [tc["tool"] for tc in captured["tool_calls"]] == \
        ["export_data", "modify_export"]
    assert captured["tool_calls"][1]["args"] == \
        {"filename": "w1.xlsx", "change": "rename X to Y"}


def test_no_tool_calls_logs_an_empty_list_not_none(monkeypatch):
    def fake_invoke(payload, config=None):
        return {"messages": [
            HumanMessage(content="hello"),
            AIMessage(content="Hi there."),
        ]}

    monkeypatch.setattr(A.agent, "invoke", fake_invoke)
    monkeypatch.setattr(A, "_restore_file_if_needed", lambda fid: None)

    captured = {}

    def fake_log_interaction(session_id, file_id, prompt, plan, response,
                             success, error=None, latency=None,
                             tool_calls=None):
        captured["tool_calls"] = tool_calls
        return "fake-log-id"

    monkeypatch.setattr("app.logging_utils.log_interaction", fake_log_interaction)

    A.run_agent("hello", session_id=_thread_id("none"), file_id=None)
    assert captured["tool_calls"] == []


def test_log_interaction_writes_the_tool_calls_field_to_disk(tmp_path, monkeypatch):
    """End to end through the real log_interaction — not just the mock
    above — confirming the field actually lands in interaction_log.jsonl."""
    import json
    log_path = tmp_path / "interaction_log.jsonl"
    monkeypatch.setattr("app.logging_utils.LOG_PATH", str(log_path))

    from app.logging_utils import log_interaction
    from app.models import QueryPlan

    plan = QueryPlan(intent="export", sink="excel", filename="w1.xlsx")
    log_interaction(
        "sess1", "file1", "export it", plan, "Done.", success=True,
        latency=1.23,
        tool_calls=[{"tool": "export_data",
                    "args": {"filename": "w1.xlsx"},
                    "result": "Verified: wrote 5 rows."}])

    entry = json.loads(log_path.read_text().strip())
    assert entry["tool_calls"] == [{
        "tool": "export_data",
        "args": {"filename": "w1.xlsx"},
        "result": "Verified: wrote 5 rows.",
    }]
    assert entry["prompt"] == "export it"
