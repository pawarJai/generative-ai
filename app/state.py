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


def set_active_file_id(fid: str) -> None:
    global ACTIVE_FILE_ID
    ACTIVE_FILE_ID = fid


def get_active_file_id() -> Optional[str]:
    return ACTIVE_FILE_ID


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_memories:
        session_memories[session_id] = InMemoryChatMessageHistory()
    return session_memories[session_id]
