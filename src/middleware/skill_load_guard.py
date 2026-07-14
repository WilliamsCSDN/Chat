"""防止同一用户轮次重复加载同名 Skill。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.types import Command


def _skill_name_from_call(tool_call: dict[str, Any]) -> str:
    args = tool_call.get("args", {})
    if not isinstance(args, dict):
        return ""
    return str(args.get("name", "")).strip()


def skill_loaded_in_current_turn(
    messages: Sequence[AnyMessage],
    skill_name: str,
    current_tool_call_id: str,
) -> bool:
    """判断最近一条用户消息之后是否已加载同名 Skill。"""
    latest_human_index = max(
        (
            index
            for index, message in enumerate(messages)
            if isinstance(message, HumanMessage)
        ),
        default=-1,
    )

    for message in messages[latest_human_index + 1 :]:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.tool_calls:
            if tool_call.get("id") == current_tool_call_id:
                continue
            if (
                tool_call.get("name") == "skills_load"
                and _skill_name_from_call(tool_call) == skill_name
            ):
                return True
    return False


def _duplicate_skill_message(request: ToolCallRequest) -> ToolMessage | None:
    tool_call = request.tool_call
    if tool_call.get("name") != "skills_load":
        return None

    skill_name = _skill_name_from_call(tool_call)
    if not skill_name or not skill_loaded_in_current_turn(
        request.state.get("messages", []),
        skill_name,
        str(tool_call.get("id", "")),
    ):
        return None

    return ToolMessage(
        content=(
            f"Skill '{skill_name}' 已在当前用户轮次加载。"
            "禁止再次加载；请使用已有指令继续，并立即完成当前回复。"
        ),
        tool_call_id=str(tool_call["id"]),
    )


class SkillLoadGuardMiddleware(AgentMiddleware):
    """为同步和异步 Agent 提供同轮 Skill 去重。"""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        blocked = _duplicate_skill_message(request)
        if blocked is not None:
            return blocked
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest],
            Awaitable[ToolMessage | Command[Any]],
        ],
    ) -> ToolMessage | Command[Any]:
        blocked = _duplicate_skill_message(request)
        if blocked is not None:
            return blocked
        return await handler(request)


prevent_duplicate_skill_load = SkillLoadGuardMiddleware()
