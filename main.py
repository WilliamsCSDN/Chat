from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from typing import List, Dict, Optional

from config import DASHSCOPE_API_KEY
from services.chat_service import chat_stream

app = FastAPI(title="百炼大模型对话", version="1.0.0")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

VALID_ROLES = {"user", "assistant", "system"}


class ChatRequest(BaseModel):
    """聊天请求体"""
    messages: List[Dict[str, str]]
    model: Optional[str] = None

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not v:
            raise ValueError("messages must not be empty")
        for idx, msg in enumerate(v):
            if "role" not in msg:
                raise ValueError(f"messages[{idx}] is missing required field 'role'")
            if "content" not in msg:
                raise ValueError(f"messages[{idx}] is missing required field 'content'")
            if msg["role"] not in VALID_ROLES:
                raise ValueError(
                    f"messages[{idx}].role must be one of {VALID_ROLES}, "
                    f"got '{msg['role']}'"
                )
            if not msg["content"].strip():
                raise ValueError(f"messages[{idx}].content must not be blank")
        return v


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回聊天页面"""
    return FileResponse("static/index.html")


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    流式对话接口。
    接收完整的对话历史 messages，返回 SSE 流。
    """
    if not DASHSCOPE_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Chat service is unavailable: DASHSCOPE_API_KEY is not configured.",
        )

    return StreamingResponse(
        chat_stream(request.messages, request.model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

