"""MCP meta-tools — 惰性披露 MCP 工具目录，避免全量 schema 注入。

工作流：LLM 依次调用
  1. mcp_list_tools(server?, keyword?)   → 查看可用工具（仅 name/description）
  2. mcp_get_schema(server, name)        → 查看某个工具的完整参数格式
  3. mcp_call_tool(server, name, args)   → 实际调用

这样 LLM 的默认上下文里只有 3 个通用 meta-tool，而不是所有 MCP 工具
的全量 schema，可大幅降低 token 占用。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from src.mcp_client import get_mcp_manager


def _mcp_not_ready() -> str:
    return json.dumps(
        {"error": "MCP manager 未初始化，请检查服务启动日志。"},
        ensure_ascii=False,
    )


@tool
async def mcp_list_tools(server: str = "", keyword: str = "") -> str:
    """列出可用的外部 MCP 工具（酒店、机票等业务能力）。

    仅在需要调用外部业务能力时使用（订机票、查酒店等）。
    - server: 可选，仅列出指定服务器的工具，如 'RollingGo-Hotel'
    - keyword: 可选，按工具名或描述做模糊过滤，如 '酒店'、'预订'

    返回 JSON 数组，每项包含 server / name / description（不含参数 schema）。
    确定要用哪个工具后，先调用 mcp_get_schema 查看它的参数格式，
    再调用 mcp_call_tool 执行。
    """
    mcp = get_mcp_manager()
    if not mcp:
        return _mcp_not_ready()
    metas = await mcp.list_tools_meta(server or None, keyword or None)
    if not metas:
        return json.dumps(
            {"tools": [], "note": "no matching tools"},
            ensure_ascii=False,
        )
    return json.dumps({"tools": metas}, ensure_ascii=False)


@tool
async def mcp_get_schema(server: str, name: str) -> str:
    """查询某个 MCP 工具的完整参数 schema。

    - server: 来自 mcp_list_tools 返回项的 server 字段
    - name: 来自 mcp_list_tools 返回项的 name 字段

    返回 JSON，包含 description 和 inputSchema。根据 schema 生成参数后，
    调用 mcp_call_tool 实际执行。
    """
    mcp = get_mcp_manager()
    if not mcp:
        return _mcp_not_ready()
    try:
        schema = await mcp.get_tool_schema(server, name)
    except ValueError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    return json.dumps(schema, ensure_ascii=False)


@tool
async def mcp_call_tool(
    server: str,
    name: str,
    arguments: dict[str, Any],
) -> str:
    """调用一个具体的 MCP 工具。

    - server: 来自 mcp_list_tools 返回项的 server 字段
    - name: 来自 mcp_list_tools 返回项的 name 字段
    - arguments: 参数对象（字典），需符合 mcp_get_schema 返回的 inputSchema

    若参数格式不确定，请先调用 mcp_get_schema 查询。
    """
    mcp = get_mcp_manager()
    if not mcp:
        return _mcp_not_ready()
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return json.dumps(
            {"error": "arguments 必须是 JSON 对象（key-value 字典）。"},
            ensure_ascii=False,
        )
    return await mcp.invoke(server, name, arguments)
