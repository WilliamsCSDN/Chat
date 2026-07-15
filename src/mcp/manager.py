"""MCP 客户端管理器 — 连接 MCP 服务器，发现工具并转换为 LangChain 工具。

每次工具调用独立建立连接（无长连接），避免会话生命周期管理的复杂性。
支持的传输类型：streamablehttp。
配置文件位置：项目根目录 mcp_config.json。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, create_model
from langchain_core.tools import StructuredTool

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)

# ── JSON Schema → Pydantic 类型映射 ──
_PYDANTIC_TYPE_MAP: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
}


def _json_schema_to_pydantic(schema: dict, model_name: str) -> type[BaseModel] | None:
    """将 JSON Schema 转换为 Pydantic 模型。若无参数返回 None。"""
    properties = schema.get("properties", {})
    if not properties:
        return None

    required: set[str] = set(schema.get("required", []))
    fields: dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get("type", "string")
        field_type = _PYDANTIC_TYPE_MAP.get(prop_type, str)
        description = prop_schema.get("description", "")
        default = ... if prop_name in required else None
        fields[prop_name] = (field_type, Field(default=default, description=description))

    return create_model(model_name, **fields)


class _ServerConfig:
    """单个 MCP 服务器的连接配置。"""

    def __init__(self, name: str, config: dict) -> None:
        self.name = name
        self.url: str = config.get("url", "")
        self.timeout: float = config.get("timeout", 30)

        transport = config.get("type", "streamablehttp")
        if transport not in ("streamablehttp",):
            raise ValueError(f"Unsupported MCP transport type: {transport}")

        # 验证 headers 值均为 ASCII（HTTP header 要求）
        raw_headers = config.get("headers", {})
        sanitized: dict[str, str] = {}
        for k, v in raw_headers.items():
            try:
                v.encode("ascii")
                sanitized[k] = v
            except UnicodeEncodeError:
                logger.warning(
                    "MCP server '%s': header '%s' contains non-ASCII characters, skipping. "
                    "Did you forget to replace the placeholder token?", name, k,
                )
        self.headers = sanitized


class MCPManager:
    """MCP 客户端管理器。

    使用方式：
        manager = MCPManager()
        manager.load_config("mcp_config.json")
        await manager.discover_tools()     # 发现并注册工具
        tools = manager.get_tools()        # 获取 LangChain 工具
    """

    def __init__(self) -> None:
        self._servers: dict[str, _ServerConfig] = {}
        self._tools: list[StructuredTool] = []
        self._tool_server_map: dict[str, str] = {}  # tool_name → server_name

    # ── 公共 API ──

    def load_config(self, config_path: str | Path) -> None:
        """从 JSON 配置文件加载 MCP 服务器列表。"""
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning("MCP config file not found: %s", config_path)
            return

        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)

        servers = data.get("mcpServers", {})
        for name, cfg in servers.items():
            try:
                self._servers[name] = _ServerConfig(name, cfg)
            except ValueError as e:
                logger.warning("Skipping MCP server '%s': %s", name, e)

        logger.info("Loaded %d MCP server config(s) from %s", len(self._servers), config_path)

    async def discover_tools(self) -> list[StructuredTool]:
        """连接到所有 MCP 服务器，发现工具并生成 LangChain 工具。

        单个服务器连接失败不影响其他服务器。
        """
        if not self._servers:
            logger.info("No MCP servers configured, skipping discovery")
            return []

        for name, server in self._servers.items():
            await self._discover_from_server(name, server)

        logger.info(
            "MCP discovery complete: %d tool(s) from %d server(s)",
            len(self._tools), len(self._servers),
        )
        return list(self._tools)

    def get_tools(self) -> list[StructuredTool]:
        """返回所有已注册的 LangChain 工具。"""
        return list(self._tools)

    # ── 内部方法 ──

    async def _discover_from_server(self, name: str, server: _ServerConfig) -> None:
        """从单个 MCP 服务器发现工具。"""
        if not server.headers:
            logger.warning(
                "MCP server '%s' has no valid headers, skipping. "
                "Check your mcp_config.json — tokens must be ASCII.", name,
            )
            return

        logger.info("MCP discovering tools from %s (%s)", name, server.url)
        try:
            async with streamablehttp_client(
                server.url,
                headers=server.headers,
                timeout=server.timeout,
            ) as (read, write, _get_session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()

                    tool_names: list[str] = []
                    for tool in result.tools:
                        lc_tool = self._build_langchain_tool(name, server, tool)
                        self._tools.append(lc_tool)
                        self._tool_server_map[tool.name] = name
                        tool_names.append(tool.name)

                    logger.info(
                        "MCP server %s: %d tools — %s",
                        name, len(tool_names), tool_names,
                    )
        except Exception:
            logger.error("MCP discovery failed for %s", name, exc_info=True)

    def _build_langchain_tool(
        self,
        server_name: str,
        server: _ServerConfig,
        mcp_tool: Any,
    ) -> StructuredTool:
        """将 MCP 工具转换为 LangChain StructuredTool。

        每次调用该工具时会独立创建 MCP 连接，用完即释放。
        """
        tool_name: str = mcp_tool.name
        description: str = mcp_tool.description or f"MCP tool: {tool_name}"
        input_schema: dict = getattr(mcp_tool, "inputSchema", {}) or {}
        args_model = _json_schema_to_pydantic(
            input_schema,
            model_name=f"mcp__{server_name}__{tool_name}__args",
        )

        if args_model is None:
            args_model = create_model(
                f"mcp__{server_name}__{tool_name}__noargs",
            )

            async def _call_no_args() -> str:
                return await _invoke_mcp_tool(server, tool_name, {})

            return StructuredTool(
                name=tool_name,
                description=description,
                args_schema=args_model,
                coroutine=_call_no_args,
            )

        async def _call_mcp_tool(**kwargs: Any) -> str:
            return await _invoke_mcp_tool(server, tool_name, kwargs)

        return StructuredTool(
            name=tool_name,
            description=description,
            args_schema=args_model,
            coroutine=_call_mcp_tool,
        )


async def _invoke_mcp_tool(server: _ServerConfig, tool_name: str, arguments: dict) -> str:
    """独立建立 MCP 连接，调用工具后断开。"""
    try:
        async with streamablehttp_client(
            server.url,
            headers=server.headers,
            timeout=server.timeout,
        ) as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                if result.isError:
                    error_text = _extract_text(result.content)
                    logger.warning(
                        "MCP tool error | server=%s tool=%s | %s",
                        server.name, tool_name, error_text[:200],
                    )
                    return f"MCP tool error: {error_text}"
                return _extract_text(result.content)
    except Exception as e:
        logger.error(
            "MCP tool call failed | server=%s tool=%s | %s",
            server.name, tool_name, e,
        )
        return f"Error calling tool '{tool_name}': {e}"


def _extract_text(content: list[Any]) -> str:
    """从 MCP ContentBlock 列表中提取纯文本。"""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
    return "\n".join(parts) if parts else str(content)
