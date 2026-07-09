#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# ── 颜色 ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { printf "${GREEN}[INFO]${NC}  %s\n" "$*"; }
warn() { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()  { printf "${RED}[ERROR]${NC} %s\n" "$*"; }

# ── 检查 uv ──
if ! command -v uv &>/dev/null; then
    err "未找到 uv，请先安装: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# ── 检查 .env ──
if [ ! -f ".env" ]; then
    err ".env 文件不存在，请先配置环境变量"
    exit 1
fi

# ── 端口占用检查 ──
if lsof -i ":$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    err "端口 $PORT 已被占用，请先释放或设置 PORT 环境变量后重试"
    exit 1
fi

# ── 安装依赖 ──
log "同步依赖中..."
uv sync --quiet

# ── 启动 ──
log "启动服务 → http://${HOST}:${PORT}"
echo ""
exec uv run uvicorn src.main:app --host "$HOST" --port "$PORT" --reload
