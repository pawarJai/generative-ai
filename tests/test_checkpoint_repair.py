"""Self-repair for a corrupted LangGraph checkpoint.

Confirmed production failure (session 477b296a): a call to export_data with
a very large `question` argument was interrupted between the model-call
step (checkpointed: an AIMessage with a tool_call) and the tool-execution
step (never ran: no matching ToolMessage). LangGraph then refused to call
the model again on that thread_id at all —

    ValueError: Found AIMessages with tool_calls that do not have a
    corresponding ToolMessage.

— which without a fix breaks every future /chat request in that session,
not just the one that hit it. The traceback showed the server itself
returning "POST /chat HTTP/1.1 200 OK" right after, because run_agent's
outer except catches the exception and answers "Something went wrong" —
but the underlying checkpoint stayed broken for the next turn too, since
nothing had repaired it.
"""
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt.chat_agent_executor import _validate_chat_history

from app.graph import agent as A


def _thread_id(label):
    return f"pytest-repair-{label}-{uuid.uuid4().hex[:8]}"


def _poison(thread_id, tool_call_id="call_orphan"):
    """Write a checkpoint for `thread_id` that reproduces the exact
    corruption: an AIMessage with a tool_call and nothing resolving it."""
    config = {"configurable": {"thread_id": thread_id}}
    orphan_call = {"id": tool_call_id, "name": "export_data",
                  "args": {"question": "x", "format": "excel"}}
    A.agent.update_state(config, {"messages": [
        HumanMessage(content="export something"),
        AIMessage(content="", tool_calls=[orphan_call]),
    ]})
    return config


def test_the_poisoned_fixture_actually_reproduces_the_real_failure():
    """Proves the setup below is testing the real bug, not a strawman."""
    config = _poison(_thread_id("proof"))
    snapshot = A.agent.get_state(config)
    with pytest.raises(ValueError, match="corresponding ToolMessage"):
        _validate_chat_history(snapshot.values["messages"])


def test_repair_patches_the_orphaned_call_and_the_thread_recovers():
    config = _poison(_thread_id("repair"))

    repaired = A._repair_orphaned_tool_calls(config)
    assert repaired is True

    snapshot = A.agent.get_state(config)
    _validate_chat_history(snapshot.values["messages"])  # must not raise


def test_repair_is_a_noop_on_a_healthy_thread():
    thread_id = _thread_id("healthy")
    config = {"configurable": {"thread_id": thread_id}}
    A.agent.update_state(config, {"messages": [
        HumanMessage(content="hello")]})
    assert A._repair_orphaned_tool_calls(config) is False


def test_the_synthetic_message_never_claims_success():
    """The repair must not let the model believe the interrupted tool call
    actually did something — that would risk it describing a file that was
    never written, the same class of fabrication found elsewhere in this
    project."""
    config = _poison(_thread_id("honesty"))
    A._repair_orphaned_tool_calls(config)
    snapshot = A.agent.get_state(config)
    tool_msgs = [m for m in snapshot.values["messages"]
                if getattr(m, "type", None) == "tool"]
    assert len(tool_msgs) == 1
    assert "interrupted" in tool_msgs[0].content.lower()
    assert "success" not in tool_msgs[0].content.lower()


# --- run_agent's own retry wiring -------------------------------------------

def test_run_agent_retries_once_after_repairing_and_recovers(monkeypatch):
    thread_id = _thread_id("run-agent")
    _poison(thread_id, tool_call_id="call_orphan_run")

    calls = {"n": 0}

    def fake_invoke(payload, config=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError(
                "Found AIMessages with tool_calls that do not have a "
                "corresponding ToolMessage. Here are the first few of "
                "those tool calls: [...]")
        return {"messages": [HumanMessage(content="hello again"),
                             AIMessage(content="Recovered fine.")]}

    monkeypatch.setattr(A.agent, "invoke", fake_invoke)
    monkeypatch.setattr(A, "_restore_file_if_needed", lambda fid: None)

    result = A.run_agent("hello again", session_id=thread_id, file_id=None)

    assert calls["n"] == 2
    assert result["response"] == "Recovered fine."
    assert result["intent"] != "error"


def test_run_agent_still_reports_a_clean_error_when_repair_cannot_help(monkeypatch):
    """A ValueError with this exact message but on a thread that turns out
    to have nothing to repair (e.g. some other cause) must not loop or
    crash uncontrolled — it still surfaces as the existing friendly error."""
    thread_id = _thread_id("unrepairable")

    def fake_invoke(payload, config=None):
        raise ValueError(
            "Found AIMessages with tool_calls that do not have a "
            "corresponding ToolMessage. Here are the first few: [...]")

    monkeypatch.setattr(A.agent, "invoke", fake_invoke)
    monkeypatch.setattr(A, "_restore_file_if_needed", lambda fid: None)

    result = A.run_agent("hello", session_id=thread_id, file_id=None)

    assert result["intent"] == "error"
    assert "Something went wrong" in result["response"]


def test_run_agent_does_not_swallow_unrelated_value_errors(monkeypatch):
    """Only the specific checkpoint-corruption message should trigger a
    repair attempt — any other ValueError must surface exactly as before."""
    thread_id = _thread_id("unrelated")

    def fake_invoke(payload, config=None):
        raise ValueError("some unrelated failure")

    monkeypatch.setattr(A.agent, "invoke", fake_invoke)
    monkeypatch.setattr(A, "_restore_file_if_needed", lambda fid: None)

    result = A.run_agent("hello", session_id=thread_id, file_id=None)

    assert result["intent"] == "error"
    assert "some unrelated failure" in result["response"]
