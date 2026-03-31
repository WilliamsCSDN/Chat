import json
import logging
from typing import AsyncGenerator, List, Dict

from openai import AsyncOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

from config_settings import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, MODEL_NAME
from tools.weather_tool import get_wether

# 创建异步 OpenAI 客户端（兼容阿里云百炼）
client = AsyncOpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
)

# 注册可用的工具（name -> 实际函数的映射）
AVAILABLE_TOOLS = {
    "get_wether": get_wether,
}

# 转换为 OpenAI tools 格式（列表）
TOOLS_SCHEMA = [convert_to_openai_tool(get_wether)]


async def chat_stream(messages: List[Dict[str, str]], model: str = None) -> AsyncGenerator[str, None]:
    """
    流式调用阿里云百炼大模型，支持 tool calling，逐步返回 SSE 格式数据。
    """
    use_model = model or MODEL_NAME
    # 复制一份 messages，避免修改原始数据
    conversation = list(messages)

    try:
        # 最多循环几轮工具调用，防止无限循环
        max_tool_rounds = 5

        for _ in range(max_tool_rounds):
            response = await client.chat.completions.create(
                model=use_model,
                tools=TOOLS_SCHEMA,
                messages=conversation,
                stream=True,
            )

            # 用于累积本轮流式返回的内容和 tool_calls
            full_content = ""
            tool_calls_map = {}  # index -> {id, function_name, arguments_str}
            finish_reason = None

            async for chunk in response:
                if not chunk.choices or len(chunk.choices) == 0:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                # 1. 处理普通文本内容 —— 流式推送给前端
                if delta.content:
                    full_content += delta.content
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

            # ---- 本轮流式结束 ----

            # 如果模型要求调用工具
            if finish_reason == "tool_calls" and tool_calls_map:
                logging.info(f"模型请求调用工具: {tool_calls_map}")

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
                    hint = json.dumps({"content": f"\n\n🔧 正在调用工具 `{func_name}`...\n\n"}, ensure_ascii=False)
                    yield f"data: {hint}\n\n"

                    # 执行工具函数
                    tool_func = AVAILABLE_TOOLS.get(func_name)
                    if tool_func:
                        try:
                            args = json.loads(func_args)
                            # LangChain @tool 装饰的函数，用 .invoke() 调用
                            tool_result = tool_func.invoke(args)
                            logging.info(f"工具 {func_name} 返回: {tool_result}")
                        except Exception as e:
                            tool_result = f"工具调用出错: {str(e)}"
                            logging.error(f"工具 {func_name} 执行失败: {e}")
                    else:
                        tool_result = f"未找到工具: {func_name}"

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
                    yield f"data: {json.dumps({'done': True})}\n\n"
                break

    except Exception as e:
        logging.error(f"chat_stream 异常: {e}", exc_info=True)
        error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
        yield f"data: {error_data}\n\n"
