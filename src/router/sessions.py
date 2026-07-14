"""会话管理路由：/api/sessions"""

from fastapi import APIRouter

from src.services.session_service import (
    list_sessions as do_list,
    delete_session as do_delete,
    get_session_messages as get_msgs,
)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
async def list_sessions() -> dict:
    data = do_list()
    return {"code": 200, "data": data}


@router.delete("/{thread_id}")
async def delete_session(thread_id: str) -> dict:
    do_delete(thread_id)
    return {"code": 200, "message": "已删除"}


@router.get("/{thread_id}/messages")
async def get_session_messages(thread_id: str) -> dict:
    data = await get_msgs(thread_id)
    return {"code": 200, "data": data}
