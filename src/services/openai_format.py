"""LangChain 消息 / chunk → OpenAI Chat Completions 规范格式。"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", "") or "")
        return "".join(parts)
    return str(content)


def _tool_call_to_openai(tc: Any) -> Dict[str, Any]:
    if isinstance(tc, dict):
        tc_id = tc.get("id", "") or ""
        name = tc.get("name", "") or ""
        args = tc.get("args", {})
        # 兼容已是 OpenAI 形状
        if "function" in tc:
            return {
                "id": tc_id or tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": (tc.get("function") or {}).get("name", name),
                    "arguments": (tc.get("function") or {}).get("arguments", "{}"),
                },
            }
    else:
        tc_id = getattr(tc, "id", "") or ""
        name = getattr(tc, "name", "") or ""
        args = getattr(tc, "args", {}) or {}

    if isinstance(args, str):
        arguments = args
    else:
        arguments = json.dumps(args if args is not None else {}, ensure_ascii=False)

    return {
        "id": tc_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def lc_message_to_openai(msg: Any) -> Optional[Dict[str, Any]]:
    """将单条 LangChain 消息转为 OpenAI chat message dict。"""
    if msg is None:
        return None

    msg_type = getattr(msg, "type", None)
    role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    if msg_type in role_map:
        role = role_map[msg_type]
    elif isinstance(msg, dict) and msg.get("role"):
        role = msg["role"]
    else:
        role = "user"

    content = _content_to_text(getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content"))

    entry: Dict[str, Any] = {"role": role, "content": content}

    tool_calls = None
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_calls = msg.tool_calls
    elif isinstance(msg, dict) and msg.get("tool_calls"):
        tool_calls = msg["tool_calls"]

    if role == "assistant" and tool_calls:
        entry["tool_calls"] = [_tool_call_to_openai(tc) for tc in tool_calls]
        # OpenAI：仅有 tool_calls 时 content 可为 null
        if not content:
            entry["content"] = None

    tool_call_id = None
    if hasattr(msg, "tool_call_id") and msg.tool_call_id:
        tool_call_id = msg.tool_call_id
    elif isinstance(msg, dict) and msg.get("tool_call_id"):
        tool_call_id = msg["tool_call_id"]

    if role == "tool" and tool_call_id:
        entry["tool_call_id"] = tool_call_id

    name = getattr(msg, "name", None) if not isinstance(msg, dict) else msg.get("name")
    if role == "tool" and name:
        entry["name"] = name

    return entry


def lc_messages_to_openai(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for msg in messages:
        converted = lc_message_to_openai(msg)
        if converted is not None:
            result.append(converted)
    return result


def chunk_to_openai_delta(chunk: Any) -> Dict[str, Any]:
    """将 AIMessageChunk 转为 OpenAI stream choices[0].delta。"""
    delta: Dict[str, Any] = {}
    if chunk is None:
        return delta

    content = getattr(chunk, "content", None)
    text = _content_to_text(content)
    if text:
        delta["content"] = text

    tool_call_chunks = getattr(chunk, "tool_call_chunks", None) or []
    if tool_call_chunks:
        deltas = []
        for i, tcc in enumerate(tool_call_chunks):
            if isinstance(tcc, dict):
                index = tcc.get("index", i)
                tc_id = tcc.get("id")
                name = tcc.get("name")
                args = tcc.get("args")
            else:
                index = getattr(tcc, "index", i)
                tc_id = getattr(tcc, "id", None)
                name = getattr(tcc, "name", None)
                args = getattr(tcc, "args", None)

            item: Dict[str, Any] = {"index": index if index is not None else i}
            if tc_id:
                item["id"] = tc_id
            item["type"] = "function"
            fn: Dict[str, Any] = {}
            if name:
                fn["name"] = name
            if args is not None:
                fn["arguments"] = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
            if fn:
                item["function"] = fn
            deltas.append(item)
        if deltas:
            delta["tool_calls"] = deltas

    # 部分 chunk 用 tool_calls 而非 tool_call_chunks
    if "tool_calls" not in delta:
        tool_calls = getattr(chunk, "tool_calls", None) or []
        if tool_calls:
            delta["tool_calls"] = [
                {
                    "index": i,
                    **_tool_call_to_openai(tc),
                }
                for i, tc in enumerate(tool_calls)
            ]

    return delta


def finish_reason_from_message(msg: Any) -> str:
    if msg is None:
        return "stop"
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        return "tool_calls"
    # LangChain / 百炼可能已写入 response_metadata
    meta = getattr(msg, "response_metadata", None) or {}
    if isinstance(meta, dict) and meta.get("finish_reason"):
        return str(meta["finish_reason"])
    return "stop"


def extract_usage(msg: Any) -> Optional[Dict[str, int]]:
    """从 AIMessage 提取 OpenAI usage 字段（若有）。"""
    if msg is None:
        return None

    usage_meta = getattr(msg, "usage_metadata", None)
    if isinstance(usage_meta, dict) and usage_meta:
        prompt = usage_meta.get("input_tokens")
        completion = usage_meta.get("output_tokens")
        total = usage_meta.get("total_tokens")
        if prompt is None and completion is None and total is None:
            return None
        prompt_i = int(prompt or 0)
        completion_i = int(completion or 0)
        total_i = int(total if total is not None else prompt_i + completion_i)
        return {
            "prompt_tokens": prompt_i,
            "completion_tokens": completion_i,
            "total_tokens": total_i,
        }

    meta = getattr(msg, "response_metadata", None) or {}
    if not isinstance(meta, dict):
        return None
    token_usage = meta.get("token_usage") or meta.get("usage") or {}
    if not isinstance(token_usage, dict) or not token_usage:
        return None
    prompt = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
    completion = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
    total = token_usage.get("total_tokens")
    if prompt is None and completion is None and total is None:
        return None
    prompt_i = int(prompt or 0)
    completion_i = int(completion or 0)
    total_i = int(total if total is not None else prompt_i + completion_i)
    return {
        "prompt_tokens": prompt_i,
        "completion_tokens": completion_i,
        "total_tokens": total_i,
    }


def build_openai_chat_completion(
    *,
    message: Dict[str, Any],
    finish_reason: str,
    model: str,
    completion_id: Optional[str] = None,
    created: Optional[int] = None,
    usage: Optional[Dict[str, int]] = None,
    system_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    """构造 OpenAI 兼容的 chat.completion 响应对象。"""
    response: Dict[str, Any] = {
        "id": completion_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(created if created is not None else time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage:
        response["usage"] = usage
    if system_fingerprint:
        response["system_fingerprint"] = system_fingerprint
    return response


def build_openai_chat_completion_chunk(
    *,
    delta: Dict[str, Any],
    model: str,
    completion_id: str,
    created: Optional[int] = None,
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """构造 OpenAI 兼容的 chat.completion.chunk 流式对象。"""
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(created if created is not None else time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


def extract_model_request_messages(input_data: Any) -> List[Any]:
    """从 on_chat_model_start 的 data.input 提取 messages 列表。"""
    if input_data is None:
        return []
    if isinstance(input_data, dict):
        msgs = input_data.get("messages")
        if msgs is not None:
            # LangChain 有时是 [[SystemMessage, HumanMessage, ...]]
            if msgs and isinstance(msgs, list) and msgs and isinstance(msgs[0], list):
                return list(msgs[0])
            if isinstance(msgs, list):
                return list(msgs)
        return []
    if isinstance(input_data, (list, tuple)):
        if input_data and isinstance(input_data[0], list):
            return list(input_data[0])
        return list(input_data)
    return []


def build_openai_request_body(
    *,
    model: str,
    messages: Sequence[Any],
    tools: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model,
        "messages": lc_messages_to_openai(messages),
    }
    if tools:
        body["tools"] = list(tools)
    return body
