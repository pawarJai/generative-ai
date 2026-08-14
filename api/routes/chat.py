"""POST /chat — the main conversational endpoint, plus session history.

Every turn is already checkpointed to agent_checkpoints.db by LangGraph, but
until now nothing could read those checkpoints back, and the frontend never
sent a session_id — so all traffic piled into a single thread named "default"
(289 checkpoints deep) and closing the tab appeared to lose the conversation.
The history endpoints below replay a thread so a chat can be resumed.
"""
import sqlite3
from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool
from app.models import ChatRequest, ChatResponse
from app.graph.agent import (run_agent as chat_fn, thread_messages,
                             CHECKPOINT_DB_PATH)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # Confirmed production failure: chat_fn (run_agent) is a blocking sync
    # call — an LLM round-trip, sometimes several in sequence for a tool
    # that retries. Calling it directly from an `async def` route runs it
    # on the server's single event-loop thread, which FREEZES THE ENTIRE
    # SERVER for every other request — not just this one — for as long as
    # it takes. Confirmed live: while one slow /chat call was in flight, a
    # plain GET /files from a different session timed out completely. Every
    # sync route below that does real work needs this same treatment; this
    # one is fixed first because it is the one that runs the longest.
    result = await run_in_threadpool(
        chat_fn, req.prompt, session_id=req.session_id, file_id=req.file_id)
    allowed = {"response", "intent", "sink", "filename", "file_id",
               "table", "sql", "available_files", "target_files"}
    filtered = {k: v for k, v in result.items() if k in allowed}
    return ChatResponse(**filtered)


def _thread_messages(session_id: str) -> list:
    """User-visible turns for one thread, oldest first — the agent's own
    reader, so the history shown here and the history the agent recalls when
    asked "what did I ask before" can never diverge."""
    try:
        return thread_messages(session_id)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}")
async def chat_history(session_id: str):
    return {"session_id": session_id, "messages": _thread_messages(session_id)}


@router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete one conversation and everything LangGraph checkpointed for it.

    The checkpointer spreads a thread across several tables (checkpoints,
    writes, blobs — the exact set varies by langgraph version), so the tables
    are discovered by looking for a thread_id column rather than hardcoded.
    Missing one would leave the thread half-deleted and still listed.

    Documents uploaded into the chat are left on disk and in the registry;
    they are simply no longer shown, since they belong to a conversation that
    no longer exists. Deleting a chat never deletes the user's source files.
    """
    try:
        conn = sqlite3.connect(CHECKPOINT_DB_PATH)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        removed = 0
        for table in tables:
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            if "thread_id" in cols:
                cur = conn.execute(
                    f"DELETE FROM {table} WHERE thread_id=?", (session_id,))
                removed += cur.rowcount
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Could not delete session: {e}")

    if not removed:
        raise HTTPException(status_code=404, detail=f"No chat found for {session_id!r}")
    return {"session_id": session_id, "deleted_rows": removed}


@router.get("/sessions")
async def chat_sessions(limit: int = 20):
    """Known conversation threads, most recently active first, so a chat
    closed in another tab can be found and resumed."""
    try:
        conn = sqlite3.connect(CHECKPOINT_DB_PATH)
        rows = conn.execute(
            "SELECT thread_id, MAX(rowid) AS last_row FROM checkpoints "
            "GROUP BY thread_id ORDER BY last_row DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Could not list sessions: {e}")

    from app.persistence import get_all_files

    sessions = []
    for thread_id, _ in rows:
        try:
            msgs = _thread_messages(thread_id)
        except HTTPException:
            continue
        # A thread with no visible turns is a chat the user opened and never
        # used — listing it would fill the sidebar with blanks.
        if not msgs:
            continue
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        sessions.append({
            "session_id": thread_id,
            "title": (first_user[:70] + "…") if len(first_user) > 70 else first_user,
            "message_count": len(msgs),
            "file_count": len(get_all_files(thread_id)),
        })
    return sessions
