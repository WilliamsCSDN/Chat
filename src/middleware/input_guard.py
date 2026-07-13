"""输入守卫中间件——在 LLM 调用前检查用户消息，拦截可疑请求。"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import AIMessage

from src.services.input_guard import get_guard
from src.config.config_settings import INPUT_GUARD_ENABLED

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = "抱歉，我无法回答这个问题。请问有什么其他可以帮助你的吗？"


def _get_last_user_message(messages: list) -> str:
    """提取最后一条用户消息内容。"""
    for msg in reversed(messages):
        role = getattr(msg, "type", None) or getattr(msg, "role", "")
        if role in ("user", "human"):
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return str(content[0] if content else "")
    return ""


class InputGuardMiddleware(AgentMiddleware):
    """在每次模型调用前检查用户输入，拦截高风险请求。"""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> Any:
        if not INPUT_GUARD_ENABLED:
            return await handler(request)

        messages = getattr(request, "messages", []) or []
        last_user = _get_last_user_message(messages)

        guard = get_guard()
        result = guard.check_message(last_user)

        logger.info(
            "输入守卫检查完成 | risk=%s | reason=%s",
            result.risk_level,
            result.reason or "none",
        )

        if result.blocked and result.risk_level == "high":
            logger.info(
                "输入守卫拦截 | risk=%s | reason=%s",
                result.risk_level,
                result.reason,
            )
            return AIMessage(content=REFUSAL_MESSAGE)

        if result.blocked and result.risk_level == "medium":
            logger.info(
                "输入守卫标记 | risk=%s | reason=%s",
                result.risk_level,
                result.reason,
            )

        return await handler(request)
