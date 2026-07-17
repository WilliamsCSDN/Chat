"""MCP 客户端模块 — 将 MCP 服务器工具接入 LangChain Agent。

目录命名为 `mcp_client` 而非 `mcp`，避免与 PyPI 官方 `mcp` SDK 命名冲突
（否则某些启动路径下 `from mcp.client.session import ...` 会被误解析到本地包，
触发循环 import）。
"""

from src.mcp_client.manager import MCPManager

_mcp_manager: MCPManager | None = None


def get_mcp_manager() -> MCPManager | None:
    """获取全局 MCP 管理器实例。"""
    return _mcp_manager


def set_mcp_manager(manager: MCPManager) -> None:
    """设置全局 MCP 管理器实例。"""
    global _mcp_manager
    _mcp_manager = manager


__all__ = ["MCPManager", "get_mcp_manager", "set_mcp_manager"]
