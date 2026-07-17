import json
import asyncio
import logging
import time
import uuid
from typing import AsyncGenerator, List, Dict

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
)
from openai import AsyncOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

from src.config.config_settings import (
    AGENT_MODEL_CALL_LIMIT,
    CHAT_LOG_FULL_MESSAGES,
    CHAT_LOG_STREAM_CHUNKS,
    CHAT_LOG_TRUNCATE_CHARS,
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    MODEL_NAME,
    SUMMARY_KEEP_MESSAGES,
    SUMMARY_TRIGGER_MESSAGES,
    SUMMARY_TRIGGER_TOKENS,
)
from src.middleware.security_prompt import inject_security_prompt
from src.middleware.skill_load_guard import prevent_duplicate_skill_load
from src.tools.rag_tool import retrieve_knowledge
from src.tools.skills import skills_load, load_skills_for_context
from src.tools.weather_tool import get_wether
from src.tools.mcp_meta import mcp_list_tools, mcp_get_schema, mcp_call_tool
from src.services.agui import (
    stream_agui_events,
    session_title,
    suggested_questions,
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
_agent_tools_cache: dict[str, list] = {}

# SQLite checkpoint saver 单例 - 由 main.py 在 startup 时初始化
_checkpointer = None


def set_checkpointer(cp) -> None:
    global _checkpointer
    _checkpointer = cp


def _get_checkpointer():
    return _checkpointer


def _build_agent_middleware(chat_model):
    """构建有界且不会污染会话状态的 Agent 中间件。"""
    return [
        ModelCallLimitMiddleware(
            run_limit=AGENT_MODEL_CALL_LIMIT,
            exit_behavior="end",
        ),
        inject_security_prompt,
        prevent_duplicate_skill_load,
        PIIMiddleware(
            "phone_number",
            detector=r"\+?\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{4}",
            strategy="mask",
            apply_to_output=True,
        ),
        # SummarizationMiddleware(
        #     model=chat_model,
        #     trigger=[
        #         ("tokens", SUMMARY_TRIGGER_TOKENS),
        #         ("messages", SUMMARY_TRIGGER_MESSAGES),
        #     ],
        #     keep=("messages", SUMMARY_KEEP_MESSAGES),
        # ),
    ]


def _get_agent(model: str):
    """返回指定 model 对应的 agent 实例，首次使用时创建并缓存。"""
    if model not in _agent_cache:
        from langchain.agents import create_agent
        from langchain_openai import ChatOpenAI
        from src.middleware.input_guard import InputGuardMiddleware

        # MCP 工具通过 meta-tool 惰性披露（避免全量 schema 注入上下文）
        tools = [
            get_wether,
            retrieve_knowledge,
            skills_load,
            mcp_list_tools,
            mcp_get_schema,
            mcp_call_tool,
        ]

        chat_model = ChatOpenAI(
            model=model,
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            temperature=0.7,
        )
        _agent_cache[model] = create_agent(
            model=chat_model,
            tools=tools,
            middleware=_build_agent_middleware(chat_model),
            checkpointer=_get_checkpointer(),
        )
        _agent_tools_cache[model] = tools
    return _agent_cache[model]


# 创建异步 OpenAI 客户端（兼容阿里云百炼）
client = AsyncOpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
)

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



async def chat_stream(messages: List[Dict[str, str]], model: str = None, thread_id: str = None) -> AsyncGenerator[str, None]:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    def _build_tools_schema():
        agent_tools = _agent_tools_cache.get(
            use_model,
            [
                get_wether,
                retrieve_knowledge,
                skills_load,
                mcp_list_tools,
                mcp_get_schema,
                mcp_call_tool,
            ],
        )
        return [convert_to_openai_tool(t) for t in agent_tools]

    is_new_thread = thread_id is None
    thread_id = thread_id or str(uuid.uuid4())
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

    # 新对话：立即启动标题生成任务，与 agent 响应并行
    title_task = None
    if is_new_thread and messages:
        first_msg = messages[0].get("content", "") if messages else ""
        if first_msg:
            from src.services.session_service import generate_and_save_title
            title_task = asyncio.create_task(generate_and_save_title(thread_id, first_msg))

    try:
        # 将 dict 消息转换为 LangChain 消息对象
        langchain_messages = []
        # 复用 thread 时 LangGraph 从 checkpoint 恢复历史，
        # add_messages reducer 会追加新消息，所以只传最后一条避免重复。
        source_messages = [messages[-1]] if not is_new_thread else messages
        for msg in source_messages:
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
        async for sse_event in stream_agui_events(
            agent,
            langchain_messages,
            thread_id,
            model=use_model,
            tools=_build_tools_schema(),
        ):
            yield sse_event

        total_cost_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "Chat 结束 | run_id=%s | total_elapsed=%.2fms",
            log_prefix,
            total_cost_ms,
        )
        # AG-UI: RUN_FINISHED
        yield sse(run_finished(thread_id, run_id))
        # 等待标题任务完成并推送到前端
        if title_task:
            try:
                title = await title_task
                yield sse(session_title(thread_id, title))
            except Exception:
                pass
        # 推荐后续问题
        async for event in recommend_question(agent, thread_id, use_model, log_prefix):
            yield event

    except Exception as e:
        logger.error("chat_stream 异常 | run_id=%s | err=%s", log_prefix, e, exc_info=True)
        yield sse(run_error(str(e), "INTERNAL_ERROR"))

async def recommend_question(agent, thread_id: str, use_model: str, log_prefix: str):
    try:
        # 从 LangGraph checkpoint 获取完整对话历史
        config = {"configurable": {"thread_id": thread_id}}
        state = await agent.aget_state(config)
        checkpoint_messages = []
        if state and state.values and "messages" in state.values:
            checkpoint_messages = state.values["messages"]
        elif state and hasattr(state, "values") and state.values and "messages" in state.values:
            checkpoint_messages = state.values["messages"]

        suggestion_prompt = (
            "根据以上对话内容，请推荐 3 个用户可能感兴趣的后续问题。\n"
            '输出一个 JSON 对象，格式为 {"questions": ["问题1", "问题2", "问题3"]}'
        )
        suggestion_messages = [
            {"role": "system", "content": suggestion_prompt},
        ]
        # 取最近 6 条 checkpoint 消息作为上下文
        for msg in checkpoint_messages[-6:]:
            content_text = msg.content if hasattr(msg, "content") else str(msg)
            _role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
            role = _role_map.get(msg.type, "user")
            suggestion_messages.append({"role": role, "content": content_text})
        suggestion_response = await client.chat.completions.create(
            model=use_model,
            messages=suggestion_messages,
            temperature=0.7,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        content = suggestion_response.choices[0].message.content.strip()
        import json
        data = json.loads(content)
        questions = data.get("questions", [])
        if isinstance(questions, list) and len(questions) > 0:
            yield sse(suggested_questions(questions[:3]))
            logger.info(
                "Chat 建议问题 | run_id=%s | questions=%s",
                log_prefix, questions,
            )
    except Exception as e:
        logger.warning("Chat 建议问题生成失败 | run_id=%s | err=%s", log_prefix, e)

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
