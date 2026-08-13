"""
Central in-memory state.

IMPORTANT: this is process-local, in-memory state — identical to the
notebook's global variables. It works for a single dev server / single
worker. It is NOT safe for multiple uvicorn workers or horizontal scaling
(each process would have its own copy). For production, replace these dicts
with a real store (Redis for session/active-file state, Postgres/S3 for
file metadata and cached tables) — the function signatures below are the
seam to do that swap without touching calling code.
"""
from typing import Dict, Optional, List, Any
import pandas as pd
from langchain_core.chat_history import InMemoryChatMessageHistory

ACTIVE_FILE_ID: Optional[str] = None
# The conversation the current request belongs to. Documents are registered
# per session, so "combine all my documents" has to mean the documents of
# THIS chat — not every file that happens to be loaded in the process.
ACTIVE_SESSION_ID: Optional[str] = None
# The user's own words for the current turn. Tools receive whatever the model
# chose to paraphrase into `question`, and a paraphrase like "all tables from
# all uploaded files" made an export fan out across every document when the
# user had asked about one. File scope is resolved from this, not the rewrite.
CURRENT_USER_PROMPT: Optional[str] = None
# The document each conversation is currently about. The UI selects the newest
# upload and keeps it selected (index.html:944), so a message that refers back
# to an earlier document — "export THIS table", "THAT page" — arrived carrying
# the wrong file_id and was answered from a spreadsheet nobody had mentioned
# for three turns. This is what the conversation established; the chip is only
# what was clicked last.
SESSION_FOCUS: Dict[str, str] = {}
# The file_id the UI sent on this session's previous turn. A chip the user just
# clicked is a deliberate act and must win; the same chip arriving unchanged
# for the fifth turn running is not evidence of anything.
LAST_SELECTION: Dict[str, str] = {}
# The output filename the user last named in this chat. "export into
# f1-14.xlsx" followed by "we need to export all data of this table" is one
# request for one file — the second message dropped the name and was read as
# not asking for a file at all, so nothing was exported and the model filled
# the silence with an invented one.
LAST_REQUESTED_EXPORT: Dict[str, str] = {}
# Whether this chat's exports carry the document's header band. "if i need
# header than i will tell but right now i don't need" is a standing
# instruction, not a one-message one; re-stating it on every export is exactly
# what the user said they should not have to do.
BAND_PREFERENCE: Dict[str, bool] = {}
DOCLING_DOCS: Dict[str, Any] = {}
TABULAR_TABLES: Dict[str, List[pd.DataFrame]] = {}
FILE_KIND: Dict[str, str] = {}
FILE_META: Dict[str, Dict[str, Any]] = {}
FILE_ORDER: List[str] = []
FILE_ORIGINAL_NAME: Dict[str, str] = {}
session_memories: Dict[str, InMemoryChatMessageHistory] = {}
LAST_EXPORT_SPEC: Dict[str, Dict[str, Any]] = {}
LAST_INTERACTION: Dict[str, str] = {}          # session_id -> interaction id (for rate_last)
LAST_TURN: Dict[str, Dict[str, Any]] = {}      # session_id -> full last turn (for ai_meta)
WRITTEN_EXPORTS: Dict[str, Dict[str, Any]] = {}  # output filename -> what produced it:
                                                # {"file_id", "pages", "context",
                                                #  "columns"}. Kept so a follow-up
                                                # "add the header details to that
                                                # excel" can modify the file that
                                                # exists instead of rebuilding it
                                                # from a prompt the user should not
                                                # have to repeat.
SUMMARY_CACHE: Dict[str, str] = {}             # content_hash -> summary text, avoids
                                                # re-running the summary LLM call when the
                                                # exact same file content is re-uploaded
                                                # under a new file_id


def set_active_file_id(fid: str) -> None:
    global ACTIVE_FILE_ID
    ACTIVE_FILE_ID = fid


def get_active_file_id() -> Optional[str]:
    return ACTIVE_FILE_ID


def set_active_session_id(sid: Optional[str]) -> None:
    global ACTIVE_SESSION_ID
    ACTIVE_SESSION_ID = sid


def get_active_session_id() -> Optional[str]:
    return ACTIVE_SESSION_ID


def set_current_user_prompt(text: Optional[str]) -> None:
    global CURRENT_USER_PROMPT
    CURRENT_USER_PROMPT = text


def get_current_user_prompt() -> Optional[str]:
    return CURRENT_USER_PROMPT


def set_session_focus(session_id: Optional[str], file_id: Optional[str]) -> None:
    if session_id and file_id:
        SESSION_FOCUS[session_id] = file_id


def get_session_focus(session_id: Optional[str]) -> Optional[str]:
    return SESSION_FOCUS.get(session_id) if session_id else None


def set_last_selection(session_id: Optional[str], file_id: Optional[str]) -> None:
    if session_id:
        LAST_SELECTION[session_id] = file_id


def get_last_selection(session_id: Optional[str]) -> Optional[str]:
    return LAST_SELECTION.get(session_id) if session_id else None


def set_last_requested_export(session_id: Optional[str], filename: Optional[str]) -> None:
    if session_id and filename:
        LAST_REQUESTED_EXPORT[session_id] = filename


def get_last_requested_export(session_id: Optional[str]) -> Optional[str]:
    return LAST_REQUESTED_EXPORT.get(session_id) if session_id else None


def set_band_preference(session_id: Optional[str], wanted: Optional[bool]) -> None:
    if session_id and wanted is not None:
        BAND_PREFERENCE[session_id] = wanted


def get_band_preference(session_id: Optional[str]) -> Optional[bool]:
    return BAND_PREFERENCE.get(session_id) if session_id else None


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_memories:
        session_memories[session_id] = InMemoryChatMessageHistory()
    return session_memories[session_id]
