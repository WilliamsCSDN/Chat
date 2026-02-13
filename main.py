from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional

from services.chat_service import chat_stream

app = FastAPI(title="百炼大模型对话", version="1.0.0")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    """聊天请求体"""
    messages: List[Dict[str, str]]
    model: Optional[str] = None


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

