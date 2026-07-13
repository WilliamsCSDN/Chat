"""
AG-UI 协议事件构造函数与 SSE 格式化工具。

将事件类型和字段约束集中管理，避免在业务代码中散落裸字典。
"""

import json
from typing import Optional


def sse(event: dict) -> str:
    """将事件字典序列化为 SSE data 行。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ── 运行生命周期 ──

def run_started(thread_id: str, run_id: str) -> dict:
    return {"type": "RUN_STARTED", "threadId": thread_id, "runId": run_id}


def run_finished(thread_id: str, run_id: str) -> dict:
    return {"type": "RUN_FINISHED", "threadId": thread_id, "runId": run_id}


def run_error(message: str, code: str = "INTERNAL_ERROR") -> dict:
    return {"type": "RUN_ERROR", "message": message, "code": code}


# ── 文本消息 ──

def text_message_start(message_id: str, role: str = "assistant") -> dict:
    return {"type": "TEXT_MESSAGE_START", "messageId": message_id, "role": role}


def text_message_content(message_id: str, delta: str) -> dict:
    return {"type": "TEXT_MESSAGE_CONTENT", "messageId": message_id, "delta": delta}


def text_message_end(message_id: str) -> dict:
    return {"type": "TEXT_MESSAGE_END", "messageId": message_id}


# ── 工具调用 ──

def tool_call_start(
    tool_call_id: str,
    tool_call_name: str,
    parent_message_id: Optional[str] = None,
) -> dict:
    event: dict = {
        "type": "TOOL_CALL_START",
        "toolCallId": tool_call_id,
        "toolCallName": tool_call_name,
    }
    if parent_message_id is not None:
        event["parentMessageId"] = parent_message_id
    return event


def tool_call_args(tool_call_id: str, delta: str) -> dict:
    return {"type": "TOOL_CALL_ARGS", "toolCallId": tool_call_id, "delta": delta}


def tool_call_end(tool_call_id: str) -> dict:
    return {"type": "TOOL_CALL_END", "toolCallId": tool_call_id}


def tool_call_result(
    message_id: str,
    tool_call_id: str,
    content: str,
    role: str = "tool",
) -> dict:
    return {
        "type": "TOOL_CALL_RESULT",
        "messageId": message_id,
        "toolCallId": tool_call_id,
        "content": content,
        "role": role,
    }
