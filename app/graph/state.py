"""LangGraph conversation state — replaces the fragile in-memory dicts."""
from typing import Annotated, Optional, List
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ConversationState(TypedDict):
    # Chat history — LangGraph manages this automatically with add_messages
    messages: Annotated[list, add_messages]
    # Which files are in scope for this conversation turn
    file_ids: List[str]
    # The active file_id (last explicitly referenced)
    active_file_id: Optional[str]
    # Intent resolved for this turn
    intent: Optional[str]
    # Structured result from tool execution
    tool_result: Optional[dict]
    # Session identifier
    session_id: str
