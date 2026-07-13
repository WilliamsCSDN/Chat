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
from src.services.agui import (
    stream_agui_events,
    sse,
    run_started,
    run_finished,
    run_error,
)
from src.services.input_guard import *
from src.config.config_settings import INPUT_GUARD_ENABLED

logger = logging.getLogger(__name__)

# Agent 缓存：按 model name 缓存 CompiledStateGraph，避免每次请求重新编译 graph
_agent_cache: dict = {}


def _get_agent(model: str):
    """返回指定 model 对应的 agent 实例，首次使用时创建并缓存。"""
    if model not in _agent_cache:
        from langchain.agents import create_agent
        from langchain_openai import ChatOpenAI
        from src.middleware.input_guard import InputGuardMiddleware
        from src.middleware.security_prompt import inject_security_prompt

        chat_model = ChatOpenAI(
            model=model,
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            temperature=0.7,
        )
        _agent_cache[model] = create_agent(
            model=chat_model,
            tools=[get_wether, retrieve_knowledge, skills_load],
            middleware=[
                InputGuardMiddleware(),
                inject_security_prompt,
            ],
        )
    return _agent_cache[model]


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
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    thread_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    started_at = time.perf_counter()
    use_model = model or MODEL_NAME
    log_prefix = run_id[:8]
    logger.info(
        "Chat 开始 | run_id=%s | model=%s | input_messages=%s",
        log_prefix,
        use_model,
        len(messages),
    )
    _log_messages(f"Chat 输入消息 | run_id={log_prefix}", list(messages))

    # AG-UI: RUN_STARTED
    yield sse(run_started(thread_id, run_id))

    try:
        # 将 dict 消息转换为 LangChain 消息对象
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                langchain_messages.append(HumanMessage(content=content))

        agent = _get_agent(use_model)

        # 流式执行 agent，通过 agui 模块映射事件
        async for sse_event in stream_agui_events(agent, langchain_messages, thread_id):
            yield sse_event

        total_cost_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "Chat 结束 | run_id=%s | total_elapsed=%.2fms",
            log_prefix,
            total_cost_ms,
        )
        # AG-UI: RUN_FINISHED
        yield sse(run_finished(thread_id, run_id))

    except Exception as e:
        logger.error("chat_stream 异常 | run_id=%s | err=%s", log_prefix, e, exc_info=True)
        yield sse(run_error(str(e), "INTERNAL_ERROR"))
async def chat_completions(request_body: dict) -> dict:
    """纯转发：非流式调用百炼 OpenAI 兼容接口，返回完整 JSON 响应。"""
    request_id = str(uuid.uuid4())[:8]
    request_body['model'] = MODEL_NAME
    use_model = request_body.get("model", MODEL_NAME)

    logger.info("Chat 转发(非流式) 开始 | req_id=%s | model=%s", request_id, use_model)

    body = dict(request_body)
    body.pop("stream", None)

    try:
        response = await client.chat.completions.create(**body)
        finish = response.choices[0].finish_reason if response.choices else "unknown"
        tokens = response.usage.total_tokens if response.usage else 0
        logger.info("Chat 转发(非流式) 结束 | req_id=%s | finish_reason=%s | tokens=%s", request_id, finish, tokens)
        return response.model_dump(exclude_unset=True)
    except Exception as e:
        logger.error("chat_completions 异常 | req_id=%s | err=%s", request_id, e, exc_info=True)
        return {"error": {"message": str(e), "type": "server_error"}}


async def chat_completions_stream(request_body: dict) -> AsyncGenerator[str, None]:
    """纯转发：流式调用百炼 OpenAI 兼容接口，透传 SSE 事件流。"""
    request_id = str(uuid.uuid4())[:8]
    request_body['model'] = MODEL_NAME
    use_model = request_body.get("model", MODEL_NAME)

    logger.info("Chat 转发(流式) 开始 | req_id=%s | model=%s", request_id, use_model)

    body = dict(request_body)
    body["stream"] = True

    try:
        response = await client.chat.completions.create(**body)
        async for chunk in response:
            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
        yield "data: [DONE]\n\n"
        logger.info("Chat 转发(流式) 结束 | req_id=%s", request_id)
    except Exception as e:
        logger.error("chat_completions_stream 异常 | req_id=%s | err=%s", request_id, e, exc_info=True)
        yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
