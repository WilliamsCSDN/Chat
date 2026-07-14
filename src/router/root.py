"""根路由：首页、健康检查、PDF-RAG"""

from typing import Any, Dict, Optional

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from src.services.pdf_rag_service import query_pdf_rag

router = APIRouter(tags=["root"])


class PdfRagRequest(BaseModel):
    query: str
    model: Optional[str] = None
    top_k: Optional[int] = None


@router.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse("static/index.html")


@router.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.post("/api/pdf-rag")
async def pdf_rag(request: PdfRagRequest) -> Dict[str, Any]:
    result = await query_pdf_rag(
        query=request.query,
        model=request.model,
        top_k=request.top_k,
    )
    return {"code": 200, "message": "success", "data": result}
