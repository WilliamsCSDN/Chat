import json
import asyncio
import logging
import time
import uuid
from typing import AsyncGenerator, List, Dict

from openai import AsyncOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

from src.config.config_settings import (
    CHAT_LOG_FULL_MESSAGES,
    CHAT_LOG_STREAM_CHUNKS,
    CHAT_LOG_TRUNCATE_CHARS,
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    MODEL_NAME,
)
from src.tools.rag_tool import retrieve_knowledge
from src.tools.skills import skills_load, load_skills_for_context
from src.tools.weather_tool import get_wether

logger = logging.getLogger(__name__)

# 创建异步 OpenAI 客户端（兼容阿里云百炼）
client = AsyncOpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
)

# 注册可用的工具（name -> 实际函数的映射）
AVAILABLE_TOOLS = {
    "get_wether": get_wether,
    "retrieve_knowledge": retrieve_knowledge,
    "skills_load": skills_load,
}

# 转换为 OpenAI tools 格式（列表）
TOOLS_SCHEMA = [
    convert_to_openai_tool(get_wether),
    convert_to_openai_tool(retrieve_knowledge),
    convert_to_openai_tool(skills_load),
]

def _truncate(text: str, max_len: int = 500) -> str:
    if CHAT_LOG_TRUNCATE_CHARS <= 0:
        return text
    max_len = CHAT_LOG_TRUNCATE_CHARS
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}...(truncated)"


def _log_messages(prefix: str, messages: List[Dict[str, str]]) -> None:
    if not CHAT_LOG_FULL_MESSAGES:
        return
    logger.info("%s | count=%s", prefix, len(messages))
    for idx, msg in enumerate(messages, start=1):
        role = msg.get("role")
        content = _truncate(str(msg.get("content", "")).replace("\n", "\\n"))
        logger.info("消息[%s] | role=%s | content=%s", idx, role, content)




async def chat_stream(messages: List[Dict[str, str]], model: str = None) -> AsyncGenerator[str, None]:
    """
    流式调用阿里云百炼大模型，支持 tool calling，逐步返回 SSE 格式数据。
    """
    request_id = str(uuid.uuid4())[:8]
    started_at = time.perf_counter()
    use_model = model or MODEL_NAME
    logger.info(
        "Chat 开始 | req_id=%s | model=%s | input_messages=%s",
        request_id,
        use_model,
        len(messages),
    )
    _log_messages(f"Chat 输入消息 | req_id={request_id}", list(messages))

    # 复制一份 messages，避免修改原始数据
    conversation = list(messages)

    # 注入可用技能列表到系统提示词（仅一次）
    skills_ctx = load_skills_for_context()
    if skills_ctx:
        skill_lines = "\n".join(f"- **{name}**: {desc}" for name, desc in skills_ctx)
        system_msg = {"role": "system", "content": f"你是百炼AI助手。当用户问题涉及以下领域时，下面是你拥有的技能清单：\n{skill_lines}"}
        if conversation and conversation[0]["role"] == "system":
            conversation[0] = system_msg
        else:
            conversation.insert(0, system_msg)

    try:
        # 最多循环几轮工具调用，防止无限循环
        max_tool_rounds = 5
        accumulated_tool_cost_ms = 0.0

        for round_idx in range(1, max_tool_rounds + 1):
            round_started_at = time.perf_counter()
            logger.info(
                "LLM 调用开始 | req_id=%s | round=%s | model=%s | messages=%s | tools=%s",
                request_id,
                round_idx,
                use_model,
                len(conversation),
                [tool["function"]["name"] for tool in TOOLS_SCHEMA],
            )
            api_started_at = time.perf_counter()
            response = await client.chat.completions.create(
                model=use_model,
                tools=TOOLS_SCHEMA,
                messages=conversation,
                stream=True,
            )
            api_call_cost_ms = (time.perf_counter() - api_started_at) * 1000
            logger.info(
                "LLM 请求已建立流 | req_id=%s | round=%s | api_elapsed=%.2fms",
                request_id,
                round_idx,
                api_call_cost_ms,
            )

            # 用于累积本轮流式返回的内容和 tool_calls
            full_content = ""
            tool_calls_map = {}  # index -> {id, function_name, arguments_str}
            finish_reason = None
            chunk_count = 0
            first_chunk_at = None

            async for chunk in response:
                if not chunk.choices or len(chunk.choices) == 0:
                    continue
                now = time.perf_counter()
                chunk_count += 1
                if first_chunk_at is None:
                    first_chunk_at = now

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                # 1. 处理普通文本内容 —— 流式推送给前端
                if delta.content:
                    full_content += delta.content
                    if CHAT_LOG_STREAM_CHUNKS:
                        logger.info(
                            "LLM 流式片段 | req_id=%s | round=%s | content=%s",
                            request_id,
                            round_idx,
                            _truncate(delta.content.replace("\n", "\\n"), 200),
                        )
                    data = json.dumps({"content": delta.content}, ensure_ascii=False)
                    yield f"data: {data}\n\n"

                # 2. 处理 tool_calls（流式分片到达，需要累积）
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "id": tc.id or "",
                                "function_name": "",
                                "arguments": "",
                            }
                        if tc.id:
                            tool_calls_map[idx]["id"] = tc.id
                        if tc.function and tc.function.name:
                            tool_calls_map[idx]["function_name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_calls_map[idx]["arguments"] += tc.function.arguments
                        if CHAT_LOG_STREAM_CHUNKS:
                            logger.info(
                                "LLM 工具流片段 | req_id=%s | round=%s | index=%s | id=%s | name=%s | args_chunk=%s",
                                request_id,
                                round_idx,
                                idx,
                                tc.id or "",
                                tc.function.name if tc.function else "",
                                _truncate(tc.function.arguments if tc.function and tc.function.arguments else "", 200),
                            )

            # ---- 本轮流式结束 ----
            round_cost_ms = (time.perf_counter() - round_started_at) * 1000
            ttfb_ms = ((first_chunk_at - round_started_at) * 1000) if first_chunk_at is not None else -1.0
            stream_cost_ms = ((time.perf_counter() - first_chunk_at) * 1000) if first_chunk_at is not None else 0.0
            logger.info(
                "LLM 调用结束 | req_id=%s | round=%s | finish_reason=%s | output_len=%s | tool_calls=%s | chunks=%s | ttfb=%.2fms | stream_elapsed=%.2fms | total_elapsed=%.2fms",
                request_id,
                round_idx,
                finish_reason,
                len(full_content),
                len(tool_calls_map),
                chunk_count,
                ttfb_ms,
                stream_cost_ms,
                round_cost_ms,
            )

            # 如果模型要求调用工具
            if finish_reason == "tool_calls" and tool_calls_map:
                logger.info("模型请求调用工具 | req_id=%s | round=%s | detail=%s", request_id, round_idx, tool_calls_map)

                # 构造 assistant 消息（包含 tool_calls）
                assistant_tool_calls = []
                for idx in sorted(tool_calls_map.keys()):
                    tc_info = tool_calls_map[idx]
                    assistant_tool_calls.append({
                        "id": tc_info["id"],
                        "type": "function",
                        "function": {
                            "name": tc_info["function_name"],
                            "arguments": tc_info["arguments"],
                        }
                    })

                # 把 assistant 的 tool_calls 消息加入对话
                conversation.append({
                    "role": "assistant",
                    "content": full_content or None,
                    "tool_calls": assistant_tool_calls,
                })

                # 给前端一个提示：正在调用工具
                for tc_call in assistant_tool_calls:
                    func_name = tc_call["function"]["name"]
                    func_args = tc_call["function"]["arguments"]
                    tool_started_at = time.perf_counter()
                    logger.info(
                        "工具执行开始 | req_id=%s | round=%s | tool=%s | args=%s",
                        request_id,
                        round_idx,
                        func_name,
                        _truncate(func_args, 500),
                    )
                    hint = json.dumps({"content": f"\n\n🔧 正在调用工具 `{func_name}`...\n\n"}, ensure_ascii=False)
                    yield f"data: {hint}\n\n"

                    # 执行工具函数
                    tool_func = AVAILABLE_TOOLS.get(func_name)
                    if tool_func:
                        try:
                            args = json.loads(func_args)
                            # LangChain @tool 装饰的函数，用 .invoke() 调用
                            tool_result = await asyncio.to_thread(tool_func.invoke, args)
                            tool_cost_ms = (time.perf_counter() - tool_started_at) * 1000
                            accumulated_tool_cost_ms += tool_cost_ms
                            logger.info(
                                "工具执行成功 | req_id=%s | round=%s | tool=%s | elapsed=%.2fms | result=%s",
                                request_id,
                                round_idx,
                                func_name,
                                tool_cost_ms,
                                _truncate(str(tool_result), 500),
                            )
                        except Exception as e:
                            tool_result = f"工具调用出错: {str(e)}"
                            tool_cost_ms = (time.perf_counter() - tool_started_at) * 1000
                            accumulated_tool_cost_ms += tool_cost_ms
                            logger.error(
                                "工具执行失败 | req_id=%s | round=%s | tool=%s | elapsed=%.2fms | err=%s",
                                request_id,
                                round_idx,
                                func_name,
                                tool_cost_ms,
                                e,
                            )
                    else:
                        tool_result = f"未找到工具: {func_name}"
                        tool_cost_ms = (time.perf_counter() - tool_started_at) * 1000
                        accumulated_tool_cost_ms += tool_cost_ms
                        logger.error(
                            "工具未找到 | req_id=%s | round=%s | tool=%s | elapsed=%.2fms",
                            request_id,
                            round_idx,
                            func_name,
                            tool_cost_ms,
                        )

                    # 把工具结果以 tool 角色消息加入对话
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tc_call["id"],
                        "content": str(tool_result),
                    })

                # 继续循环，让模型根据工具结果生成最终回复
                continue

            else:
                # 没有工具调用，正常结束
                if finish_reason is not None:
                    total_cost_ms = (time.perf_counter() - started_at) * 1000
                    logger.info(
                        "Chat 结束 | req_id=%s | finish_reason=%s | tool_elapsed=%.2fms | total_elapsed=%.2fms",
                        request_id,
                        finish_reason,
                        accumulated_tool_cost_ms,
                        total_cost_ms,
                    )
                    yield f"data: {json.dumps({'done': True})}\n\n"
                break

    except Exception as e:
        logger.error("chat_stream 异常 | req_id=%s | err=%s", request_id, e, exc_info=True)
        error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
        yield f"data: {error_data}\n\n"
