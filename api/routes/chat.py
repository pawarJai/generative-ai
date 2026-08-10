"""POST /chat — the main conversational endpoint."""
from fastapi import APIRouter
from app.models import ChatRequest, ChatResponse
from app.query.dispatch import chat as chat_fn

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    result = chat_fn(req.prompt, session_id=req.session_id, file_id=req.file_id)
    return ChatResponse(**result)
