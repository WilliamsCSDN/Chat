"""
AG-UI 协议事件构造函数与 SSE 格式化工具。

将事件类型和字段约束集中管理，避免在业务代码中散落裸字典。
"""

import json
import uuid
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


async def stream_agui_events(agent, langchain_messages, thread_id: str):
    """流式执行 LangGraph agent 并将事件映射为 AG-UI SSE 字符串。

    内部管理消息生命周期与工具调用状态，确保 AG-UI 协议完整性：
    - on_chat_model_stream → TEXT_MESSAGE_START / CONTENT
    - on_chat_model_end   → TEXT_MESSAGE_END
    - on_tool_start       → TOOL_CALL_START / ARGS
    - on_tool_end         → TOOL_CALL_END / RESULT

    还处理中间件拦截场景（无 stream 但有最终 AIMessage 输出）。
    """
    message_started = False
    _message_count = 0
    message_id = str(uuid.uuid4())
    tool_started: set = set()
    last_event = None

    async for event in agent.astream_events(
        {"messages": langchain_messages},
        version="v2",
        config={"configurable": {"thread_id": thread_id}},
    ):
        last_event = event
        evt_type = event.get("event", "")
        data = event.get("data", {})
        run_evt_id = event.get("run_id", "")

        if evt_type == "on_chat_model_stream":
            chunk = data.get("chunk")
            if chunk is None:
                continue
            if hasattr(chunk, "content") and chunk.content:
                content_text = chunk.content
                if isinstance(content_text, list):
                    text_parts = []
                    for part in content_text:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content_text = "".join(text_parts)
                if isinstance(content_text, str) and content_text:
                    if _message_count >= 1 and not message_started:
                        continue
                    if not message_started:
                        _message_count += 1
                        message_started = True
                        message_id = str(uuid.uuid4())
                        yield sse(text_message_start(message_id, "assistant"))
                    yield sse(text_message_content(message_id, content_text))

        elif evt_type == "on_chat_model_end":
            if message_started:
                yield sse(text_message_end(message_id))
                message_started = False

        elif evt_type == "on_tool_start":
            tool_name = event.get("name", "") or data.get("name", "unknown")
            tool_id = run_evt_id
            tool_started.add(tool_id)
            yield sse(tool_call_start(tool_id, tool_name))
            tool_input = data.get("input", {})
            if tool_input:
                yield sse(tool_call_args(tool_id, json.dumps(tool_input, ensure_ascii=False)))

        elif evt_type == "on_tool_end":
            tool_id = run_evt_id
            if tool_id in tool_started:
                yield sse(tool_call_end(tool_id))
                tool_started.discard(tool_id)
            output = data.get("output", "")
            if output:
                if hasattr(output, "content"):
                    tool_content = str(output.content)
                elif isinstance(output, str):
                    tool_content = output
                else:
                    tool_content = str(output)
                result_msg_id = str(uuid.uuid4())
                yield sse(tool_call_result(result_msg_id, tool_id, tool_content, "tool"))

    # 被中间件拦截的情况：从未 streaming 但有最终 AIMessage 输出
    if _message_count == 0 and not message_started and last_event is not None:
        final_output = last_event.get("data", {}).get("output", {})
        final_messages = final_output.get("messages", []) if isinstance(final_output, dict) else []
        if final_messages:
            last_msg = final_messages[-1]
            if hasattr(last_msg, "content") and isinstance(last_msg.content, str) and last_msg.content:
                message_id = str(uuid.uuid4())
                yield sse(text_message_start(message_id, "assistant"))
                yield sse(text_message_content(message_id, last_msg.content))
                yield sse(text_message_end(message_id))

    if message_started:
        yield sse(text_message_end(message_id))
