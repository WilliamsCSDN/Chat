"""SQLite 会话持久化服务，集中管理所有直接 SQLite 操作。"""

import asyncio
import json
import logging
import os
import sqlite3
from typing import Optional

from openai import AsyncOpenAI
from langchain_core.messages import AIMessage, ToolMessage

from src.config.config_settings import SQLITE_DB_PATH

logger = logging.getLogger(__name__)


# ── Checkpointer 初始化 ──

async def init_checkpointer() -> None:
    """初始化 AsyncSqliteSaver 并注入到 chat_service（需要运行中的 event loop）。"""
    import os as _os
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from src.services.chat_service import set_checkpointer
    import aiosqlite as _aiosqlite

    _os.makedirs(_os.path.dirname(SQLITE_DB_PATH) or ".", exist_ok=True)
    conn = await _aiosqlite.connect(SQLITE_DB_PATH)
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    set_checkpointer(saver)


def _ensure_titles_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS session_titles ("
        " thread_id TEXT PRIMARY KEY,"
        " title TEXT NOT NULL"
        ")"
    )
    conn.commit()


# ── 标题存储（SQLite session_titles 表） ──

def save_title(thread_id: str, title: str) -> None:
    if not _db_path_exists():
        return
    conn = _get_conn()
    try:
        _ensure_titles_table(conn)
        conn.execute(
            "INSERT OR REPLACE INTO session_titles (thread_id, title) VALUES (?, ?)",
            (thread_id, title),
        )
        conn.commit()
    finally:
        conn.close()


def get_title(thread_id: str) -> Optional[str]:
    if not _db_path_exists():
        return None
    conn = _get_conn()
    try:
        _ensure_titles_table(conn)
        row = conn.execute(
            "SELECT title FROM session_titles WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


# ── LLM 标题生成（供 chat_service 在新对话完成时调用） ──

_title_client: Optional[AsyncOpenAI] = None


def _get_title_client() -> AsyncOpenAI:
    global _title_client
    if _title_client is None:
        from src.config.config_settings import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL
        _title_client = AsyncOpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
        )
    return _title_client


async def generate_and_save_title(thread_id: str, user_message: str) -> str:
    """为新会话生成标题并存入 SQLite。失败时 fallback 到 30 字截断。返回标题。"""
    prompt = (
        "将以下用户问题总结为一个非常简短的标题（不超过15个字），"
        "直接返回标题文本，不要任何标点或额外说明。\n\n"
        f"用户问题：{user_message}"
    )
    client = _get_title_client()
    try:
        resp = await client.chat.completions.create(
            model="qwen-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=30,
        )
        title = resp.choices[0].message.content.strip()
    except Exception:
        logger.warning("LLM 标题生成失败，使用截断标题 | thread_id=%s", thread_id, exc_info=True)
        title = user_message[:30]
    if title:
        save_title(thread_id, title)
    return title or user_message[:30]


def _db_path_exists() -> bool:
    import os as _os
    return _os.path.exists(SQLITE_DB_PATH)


def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(SQLITE_DB_PATH)


# ── 会话列表 ──

def list_sessions() -> list[dict]:
    """列出所有历史会话，每条返回 thread_id、标题和更新时间。"""
    if not _db_path_exists():
        return []

    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    try:
        _ensure_titles_table(conn)
        rows = conn.execute(
            "SELECT thread_id, MAX(rowid) AS max_rowid "
            "FROM checkpoints WHERE checkpoint_ns = '' "
            "GROUP BY thread_id ORDER BY max_rowid DESC"
        ).fetchall()

        sessions = []
        for row in rows:
            tid = row["thread_id"]
            # 优先从 session_titles 表读取
            tr = conn.execute(
                "SELECT title FROM session_titles WHERE thread_id = ?",
                (tid,),
            ).fetchone()
            if tr and tr["title"]:
                title = tr["title"]
            else:
                # fallback: 截断首条用户消息
                first_msg = _extract_first_user_message_full(conn, tid)
                title = (first_msg[:30] if first_msg else "新对话")

            cp = conn.execute(
                "SELECT checkpoint_id, metadata FROM checkpoints "
                "WHERE thread_id = ? AND checkpoint_ns = '' "
                "ORDER BY rowid DESC LIMIT 1",
                (tid,),
            ).fetchone()
            if cp is None:
                continue

            meta = json.loads(cp["metadata"])
            sessions.append({
                "thread_id": tid,
                "title": title[:30],
                "updated_at": meta.get("ts", ""),
            })
        return sessions
    finally:
        conn.close()


# ── 会话删除 ──

def delete_session(thread_id: str) -> None:
    """删除指定会话的所有 checkpoint、writes 和标题数据。"""
    if not _db_path_exists():
        return

    conn = _get_conn()
    try:
        _ensure_titles_table(conn)
        conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM session_titles WHERE thread_id = ?", (thread_id,))
        conn.commit()
    finally:
        conn.close()


# ── 消息历史 ──

async def get_session_messages(thread_id: str) -> list[dict]:
    """返回指定会话的完整消息历史（通过 checkpoint saver）。"""
    from src.services.chat_service import _get_checkpointer

    if not _db_path_exists():
        return []

    try:
        checkpointer = _get_checkpointer()
        config = {"configurable": {"thread_id": thread_id}}
        state = await checkpointer.aget(config)
        result = []
        if state and "channel_values" in state and "messages" in state["channel_values"]:
            for msg in state["channel_values"]["messages"]:
                role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
                content = getattr(msg, "content", "") or ""
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, dict) and part.get("type") == "tool_use":
                            pass
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = "".join(text_parts)

                entry = {
                    "role": role_map.get(msg.type, "user") if hasattr(msg, "type") else "user",
                    "content": content,
                }

                # AI messages: include tool_calls if present
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": (tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")),
                            "name": (tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")),
                            "args": (tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})),
                        }
                        for tc in msg.tool_calls
                    ]
                # Tool messages: include tool_call_id
                if isinstance(msg, ToolMessage) and hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    entry["tool_call_id"] = msg.tool_call_id

                result.append(entry)
        return result
    except Exception:
        logger.warning("获取会话消息失败 | thread_id=%s", thread_id, exc_info=True)
        return []


# ── 内部辅助 ──

def _extract_first_user_message_full(conn: sqlite3.Connection, thread_id: str) -> Optional[str]:
    """从 writes 表读取首条用户消息（完整文本，不截断）。"""
    try:
        rows = conn.execute(
            "SELECT value FROM writes "
            "WHERE thread_id = ? AND channel = 'messages' AND type = 'msgpack' "
            "ORDER BY rowid ASC LIMIT 5",
            (thread_id,),
        ).fetchall()
        for r in rows:
            result = _parse_first_human_from_msgpack(r["value"])
            if result:
                return result
        return None
    except Exception:
        return None


def _parse_first_human_from_msgpack(value_bytes: bytes) -> Optional[str]:
    """从 msgpack 序列化的 writes.value 中提取第一条用户消息（完整）。"""
    try:
        import msgpack
        data = msgpack.unpackb(value_bytes)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("type") == "human":
                    content = item.get("content", "")
                    if isinstance(content, str):
                        return content.strip()
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                return part.get("text", "").strip()
        return None
    except Exception:
        return None
