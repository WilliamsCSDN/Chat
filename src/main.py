"""FastAPI 应用入口 — 负责初始化日志、挂载静态文件、注册路由、启动事件及 MCP 工具。"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.config.log import init_logging
from src.config.config_settings import LOG_FILE_PATH, LOG_LEVEL, LOG_TO_FILE, SDK_HTTP_DEBUG

init_logging(
    level=LOG_LEVEL,
    log_to_file=LOG_TO_FILE,
    log_file_path=LOG_FILE_PATH,
    sdk_http_debug=SDK_HTTP_DEBUG,
)

app = FastAPI(title="百炼大模型对话", version="1.0.0")


@app.on_event("startup")
async def startup_checkpointer():
    from src.services.session_service import init_checkpointer
    from src.mcp_client import MCPManager, set_mcp_manager
    await init_checkpointer()
    mcp = MCPManager()
    mcp.load_config("mcp_config.json")
    set_mcp_manager(mcp)
    # 并行 warmup：拉 instructions + 预热 tools_meta；
    # 单个 server 失败不阻塞启动
    await mcp.discover_all()


# ── 静态文件 ──
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── 路由注册 ──
from src.router.root import router as root_router
from src.router.chat import router as chat_router, v1_router
from src.router.sessions import router as sessions_router

app.include_router(root_router)
app.include_router(chat_router)
app.include_router(v1_router)
app.include_router(sessions_router)
