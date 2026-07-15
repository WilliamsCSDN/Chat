"""MCP 客户端模块 — 将 MCP 服务器工具接入 LangChain Agent。"""

from src.mcp.manager import MCPManager

_mcp_manager: MCPManager | None = None


def get_mcp_manager() -> MCPManager | None:
    """获取全局 MCP 管理器实例。"""
    return _mcp_manager


def set_mcp_manager(manager: MCPManager) -> None:
    """设置全局 MCP 管理器实例。"""
    global _mcp_manager
    _mcp_manager = manager


__all__ = ["MCPManager", "get_mcp_manager", "set_mcp_manager"]
