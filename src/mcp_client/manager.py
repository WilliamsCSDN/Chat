"""MCP 客户端管理器 — 惰性发现 + 按需调用 MCP 服务器工具。

设计要点：
- 启动只加载配置，不连接 server（避免上下文膨胀 / 冷启动阻塞）。
- 首次访问某个 server 时才 list_tools，metadata 缓存到进程内。
- 通过 meta-tool 方式暴露给 LLM（见 src/tools/mcp_meta.py）：
    list_tools_meta / get_tool_schema / invoke 三个接口。
- 每次工具调用独立建立连接（无长连接），避免会话生命周期管理。
- 支持的传输类型：streamablehttp。
- 配置文件位置：项目根目录 mcp_config.json。
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


class _ServerConfig:
    """单个 MCP 服务器的连接配置。"""

    def __init__(self, name: str, config: dict) -> None:
        self.name = name
        self.url: str = config.get("url", "")
        self.timeout: float = config.get("timeout", 30)
        # 可选：config 里手写的本地 description，作为 instructions 缺失时的兜底
        self.description: str = str(config.get("description", "")).strip()

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
    """MCP 客户端管理器（惰性发现）。

    使用方式：
        manager = MCPManager()
        manager.load_config("mcp_config.json")
        set_mcp_manager(manager)
        # 之后由 meta-tool 按需触发 list_tools_meta / get_tool_schema / invoke
    """

    def __init__(self) -> None:
        self._servers: dict[str, _ServerConfig] = {}
        # server_name -> {tool_name -> {"description": str, "inputSchema": dict}}
        self._tools_meta: dict[str, dict[str, dict[str, Any]]] = {}
        # server_name -> {"title", "version", "instructions"}
        # 来自 MCP initialize 返回的 serverInfo/instructions
        self._server_meta: dict[str, dict[str, str]] = {}
        # 标记哪些 server 已经尝试过 list_tools（避免重复连接）
        self._discovered: set[str] = set()

    # ---- 公共 API ----

    def load_config(self, config_path: str | Path) -> None:
        """从 JSON 配置文件加载 MCP 服务器列表（不连接服务器）。"""
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

        logger.info(
            "Loaded %d MCP server config(s) from %s (lazy discovery)",
            len(self._servers), config_path,
        )

    def list_servers(self) -> list[str]:
        """返回已注册的 MCP 服务器名称列表（不触发连接）。"""
        return list(self._servers.keys())

    def list_servers_meta(self) -> list[dict[str, str]]:
        """返回 MCP 服务器元信息列表（不触发连接，仅使用已缓存数据）。

        每项字段：
            name        — server 唯一名（config 里的 key）
            title       — 显示名（来自 initialize.serverInfo.title，可能为空）
            version     — 版本号（来自 initialize.serverInfo.version，可能为空）
            description — 用途描述，三层优先级：
                          instructions（MCP 协议）
                          → config.description（本地覆盖）
                          → "" （无）
        """
        results: list[dict[str, str]] = []
        for name, cfg in self._servers.items():
            meta = self._server_meta.get(name, {})
            description = (
                (meta.get("instructions") or "").strip()
                or cfg.description
                or ""
            )
            results.append({
                "name": name,
                "title": (meta.get("title") or "").strip(),
                "version": (meta.get("version") or "").strip(),
                "description": description,
            })
        return results

    async def list_tools_meta(
        self,
        server: str | None = None,
        keyword: str | None = None,
    ) -> list[dict[str, str]]:
        """列出可用的 MCP 工具元信息（惰性发现）。

        返回项仅含 server/name/description，不含 inputSchema，用于节省上下文。
        - server：可选，仅列出指定 server 的工具
        - keyword：可选，按 name 或 description 模糊过滤（忽略大小写）
        """
        targets = [server] if server else list(self._servers.keys())
        results: list[dict[str, str]] = []
        for name in targets:
            if name not in self._servers:
                logger.warning("MCP list_tools_meta: unknown server '%s'", name)
                continue
            await self._ensure_discovered(name)
            for tool_name, meta in self._tools_meta.get(name, {}).items():
                if keyword:
                    haystack = f"{tool_name} {meta.get('description', '')}".lower()
                    if keyword.lower() not in haystack:
                        continue
                results.append({
                    "server": name,
                    "name": tool_name,
                    "description": meta.get("description", ""),
                })
        return results

    async def get_tool_schema(self, server: str, name: str) -> dict[str, Any]:
        """获取某个 MCP 工具的完整 inputSchema（惰性发现）。

        找不到时抛出 ValueError，由调用方决定如何回给 LLM。
        """
        if server not in self._servers:
            raise ValueError(f"unknown mcp server: {server}")
        await self._ensure_discovered(server)
        meta = self._tools_meta.get(server, {}).get(name)
        if meta is None:
            raise ValueError(f"tool not found: {server}/{name}")
        return {
            "server": server,
            "name": name,
            "description": meta.get("description", ""),
            "inputSchema": meta.get("inputSchema", {}),
        }

    async def invoke(
        self,
        server: str,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        """调用具体的 MCP 工具，返回文本结果或错误提示。"""
        if server not in self._servers:
            return f"unknown mcp server: {server}"
        await self._ensure_discovered(server)
        if name not in self._tools_meta.get(server, {}):
            return f"tool not found: {server}/{name}"
        return await _invoke_mcp_tool(self._servers[server], name, arguments)

    async def discover_all(self) -> None:
        """预热所有 server（并行）：拉 initialize + list_tools。

        - 并行执行，多个 server 不互相阻塞
        - 单个 server 失败不影响其他 server
        - 结果缓存在 _server_meta / _tools_meta，供 list_servers_meta /
          list_tools_meta 使用（system prompt 与 meta-tool 都能直接命中缓存）
        """
        if not self._servers:
            return
        tasks = [self._ensure_discovered(name) for name in self._servers.keys()]
        await asyncio.gather(*tasks, return_exceptions=True)

    # ---- 内部方法 ----

    async def _ensure_discovered(self, name: str) -> None:
        """首次访问 server 时懒发现工具目录，之后使用缓存。

        无论成功失败都会写入 _discovered，避免每次调用都重连。
        如需重试，重启服务或手动清理 _discovered。
        """
        if name in self._discovered:
            return
        server = self._servers.get(name)
        if not server:
            return
        if not server.headers:
            logger.warning(
                "MCP server '%s' has no valid headers, skipping. "
                "Check your mcp_config.json — tokens must be ASCII.", name,
            )
            self._discovered.add(name)
            return

        logger.info("MCP discovering tools from %s (%s)", name, server.url)
        try:
            async with streamablehttp_client(
                server.url,
                headers=server.headers,
                timeout=server.timeout,
            ) as (read, write, _get_session_id):
                async with ClientSession(read, write) as session:
                    init_result = await session.initialize()

                    # 缓存 server 元信息（供 system prompt 使用）
                    info = getattr(init_result, "serverInfo", None)
                    self._server_meta[name] = {
                        "title": getattr(info, "title", None) or "",
                        "version": getattr(info, "version", None) or "",
                        "instructions": getattr(init_result, "instructions", None) or "",
                    }

                    result = await session.list_tools()
                    tools_map: dict[str, dict[str, Any]] = {}
                    for tool in result.tools:
                        tools_map[tool.name] = {
                            "description": tool.description or "",
                            "inputSchema": getattr(tool, "inputSchema", {}) or {},
                        }
                    self._tools_meta[name] = tools_map
                    logger.info(
                        "MCP server %s: %d tools cached — %s (instructions=%s)",
                        name,
                        len(tools_map),
                        list(tools_map.keys()),
                        "yes" if self._server_meta[name]["instructions"] else "no",
                    )
        except Exception:
            logger.error("MCP discovery failed for %s", name, exc_info=True)
        finally:
            self._discovered.add(name)


async def _invoke_mcp_tool(
    server: _ServerConfig,
    tool_name: str,
    arguments: dict,
) -> str:
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
