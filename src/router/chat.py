"""聊天相关路由：/api/chat, /v1/chat/completions"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.services.chat_service import chat_completions, chat_completions_stream, chat_stream

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    thread_id: Optional[str] = None


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        chat_stream(request.messages, request.model, request.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


v1_router = APIRouter(tags=["openai-compat"])


@v1_router.post("/v1/chat/completions")
async def openai_chat_completions(request: Dict[str, Any] = Body(...)):
    if request.get("stream", False):
        return StreamingResponse(
            chat_completions_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    return await chat_completions(request)
