"""
AG-UI 协议事件构造函数与 SSE 格式化工具。

将事件类型和字段约束集中管理，避免在业务代码中散落裸字典。
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from src.config.config_settings import AGENT_RECURSION_LIMIT, MODEL_NAME
from src.services.openai_format import (
    build_openai_chat_completion,
    build_openai_chat_completion_chunk,
    build_openai_request_body,
    chunk_to_openai_delta,
    extract_model_request_messages,
    extract_usage,
    finish_reason_from_message,
    lc_message_to_openai,
)


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


# ── OpenAI 协议调试事件 ──

def openai_model_request(call_id: str, request: dict) -> dict:
    return {"type": "OPENAI_MODEL_REQUEST", "callId": call_id, "request": request}


def openai_model_chunk(call_id: str, chunk: dict) -> dict:
    return {
        "type": "OPENAI_MODEL_CHUNK",
        "callId": call_id,
        "chunk": chunk,
    }


def openai_model_response(call_id: str, response: dict) -> dict:
    return {
        "type": "OPENAI_MODEL_RESPONSE",
        "callId": call_id,
        "response": response,
    }


def openai_messages_upsert(message: dict) -> dict:
    return {"type": "OPENAI_MESSAGES_UPSERT", "message": message}


def _metadata_model(event: dict, fallback: str) -> str:
    meta = event.get("metadata") or {}
    if isinstance(meta, dict):
        for key in ("ls_model_name", "model_name", "model"):
            if meta.get(key):
                return str(meta[key])
    return fallback


async def stream_agui_events(
    agent,
    langchain_messages,
    thread_id: str,
    *,
    model: Optional[str] = None,
    tools: Optional[Sequence[Dict[str, Any]]] = None,
):
    """流式执行 LangGraph agent 并将事件映射为 AG-UI SSE 字符串。

    内部管理消息生命周期与工具调用状态，确保 AG-UI 协议完整性：
    - on_chat_model_stream → TEXT_MESSAGE_START / CONTENT
    - on_chat_model_end   → TEXT_MESSAGE_END
    - on_tool_start       → TOOL_CALL_START / ARGS
    - on_tool_end         → TOOL_CALL_END / RESULT

    同时推送 OpenAI 协议调试事件：
    - on_chat_model_start → OPENAI_MODEL_REQUEST
    - on_chat_model_stream → OPENAI_MODEL_CHUNK
    - on_chat_model_end → OPENAI_MODEL_RESPONSE + OPENAI_MESSAGES_UPSERT
    - on_tool_end → OPENAI_MESSAGES_UPSERT (tool)
    - 输入的用户消息也会先 UPSERT

    还处理中间件拦截场景（无 stream 但有最终 AIMessage 输出）。
    """
    use_model = model or MODEL_NAME
    tools_list: List[Dict[str, Any]] = list(tools) if tools else []

    message_started = False
    _message_count = 0
    message_id = str(uuid.uuid4())
    tool_started: set = set()
    last_event = None
    active_model_call_id: Optional[str] = None
    active_model_name: str = use_model
    active_completion_id: Optional[str] = None
    active_completion_created: Optional[int] = None

    # 先推送本轮输入消息（通常为最后一条 human）
    for msg in langchain_messages:
        converted = lc_message_to_openai(msg)
        if converted:
            yield sse(openai_messages_upsert(converted))

    async for event in agent.astream_events(
        {"messages": langchain_messages},
        version="v2",
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": AGENT_RECURSION_LIMIT,
        },
    ):
        last_event = event
        evt_type = event.get("event", "")
        data = event.get("data", {})
        run_evt_id = event.get("run_id", "") or str(uuid.uuid4())

        if evt_type == "on_chat_model_start":
            active_model_call_id = run_evt_id
            active_model_name = _metadata_model(event, use_model)
            active_completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            active_completion_created = int(time.time())
            req_messages = extract_model_request_messages(data.get("input"))
            request_body = build_openai_request_body(
                model=active_model_name,
                messages=req_messages,
                tools=tools_list or None,
            )
            yield sse(openai_model_request(active_model_call_id, request_body))

        elif evt_type == "on_chat_model_stream":
            chunk = data.get("chunk")
            if chunk is None:
                continue

            call_id = active_model_call_id or run_evt_id
            delta = chunk_to_openai_delta(chunk)
            if delta:
                completion_id = active_completion_id or f"chatcmpl-{call_id[:24]}"
                openai_chunk = build_openai_chat_completion_chunk(
                    delta=delta,
                    model=active_model_name,
                    completion_id=completion_id,
                    created=active_completion_created,
                )
                yield sse(openai_model_chunk(call_id, openai_chunk))

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

            output = data.get("output")
            call_id = active_model_call_id or run_evt_id
            openai_msg = lc_message_to_openai(output)
            if openai_msg is None:
                openai_msg = {"role": "assistant", "content": ""}
            reason = finish_reason_from_message(output)
            completion = build_openai_chat_completion(
                message=openai_msg,
                finish_reason=reason,
                model=active_model_name,
                completion_id=active_completion_id,
                created=active_completion_created,
                usage=extract_usage(output),
            )
            yield sse(openai_model_response(call_id, completion))
            yield sse(openai_messages_upsert(openai_msg))
            active_model_call_id = None
            active_completion_id = None
            active_completion_created = None

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
            tool_content = ""
            tool_openai_msg = None
            if output:
                if hasattr(output, "content") or getattr(output, "type", None) == "tool":
                    tool_openai_msg = lc_message_to_openai(output)
                    if hasattr(output, "content"):
                        tool_content = str(output.content)
                    elif isinstance(output, str):
                        tool_content = output
                    else:
                        tool_content = str(output)
                elif isinstance(output, str):
                    tool_content = output
                else:
                    tool_content = str(output)
                result_msg_id = str(uuid.uuid4())
                yield sse(tool_call_result(result_msg_id, tool_id, tool_content, "tool"))
                if tool_openai_msg is None:
                    tool_openai_msg = {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": tool_content,
                    }
                elif not tool_openai_msg.get("tool_call_id"):
                    tool_openai_msg["tool_call_id"] = tool_id
                yield sse(openai_messages_upsert(tool_openai_msg))

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
                openai_msg = lc_message_to_openai(last_msg)
                if openai_msg:
                    yield sse(openai_messages_upsert(openai_msg))

    if message_started:
        yield sse(text_message_end(message_id))


# ── 建议问题 ──

def suggested_questions(questions: list) -> dict:
    """推荐后续问题事件，questions 为字符串列表。"""
    return {"type": "SUGGESTED_QUESTIONS", "questions": questions}


# ── 会话标题 ──

def session_title(thread_id: str, title: str) -> dict:
    """新会话生成标题后推送到前端。"""
    return {"type": "SESSION_TITLE", "threadId": thread_id, "title": title}
