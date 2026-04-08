from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config.log import init_logging
from config_settings import LOG_FILE_PATH, LOG_LEVEL, LOG_TO_FILE, SDK_HTTP_DEBUG
from services.chat_service import chat_stream

init_logging(
    level=LOG_LEVEL,
    log_to_file=LOG_TO_FILE,
    log_file_path=LOG_FILE_PATH,
    sdk_http_debug=SDK_HTTP_DEBUG,
)

app = FastAPI(title="百炼大模型对话", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        chat_stream(request.messages, request.model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
