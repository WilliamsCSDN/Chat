import json
from typing import AsyncGenerator, List, Dict

from openai import AsyncOpenAI

from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, MODEL_NAME


# 创建异步 OpenAI 客户端（兼容阿里云百炼）
client = AsyncOpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
)


async def chat_stream(messages: List[Dict[str, str]], model: str = None) -> AsyncGenerator[str, None]:
    """
    流式调用阿里云百炼大模型，逐步返回 SSE 格式数据。

    Args:
        messages: 对话历史消息列表，格式为 [{"role": "user/assistant/system", "content": "..."}]
        model: 模型名称，默认使用配置文件中的 MODEL_NAME

    Yields:
        SSE 格式的字符串，如 "data: {json}\n\n"
    """
    use_model = model or MODEL_NAME

    try:
        response = await client.chat.completions.create(
            model=use_model,
            messages=messages,
            stream=True,
        )

        async for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                # 获取增量内容
                if delta.content:
                    data = json.dumps({"content": delta.content}, ensure_ascii=False)
                    yield f"data: {data}\n\n"

                # 检查是否结束
                if chunk.choices[0].finish_reason is not None:
                    yield f"data: {json.dumps({'done': True})}\n\n"

    except Exception as e:
        error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
        yield f"data: {error_data}\n\n"

