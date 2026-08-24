#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
ACTION="${1:-start}"

# ── 颜色 ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { printf "${GREEN}[INFO]${NC}  %s\n" "$*"; }
warn() { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()  { printf "${RED}[ERROR]${NC} %s\n" "$*"; }

usage() {
    printf "用法: %s [start|restart]\n" "$(basename "$0")"
    printf "  start    启动服务（默认）；端口占用则失败\n"
    printf "  restart  先终止占用端口的进程，再启动\n"
}

port_in_use() {
    lsof -i ":$PORT" -sTCP:LISTEN -t >/dev/null 2>&1
}

port_pids() {
    lsof -i ":$PORT" -sTCP:LISTEN -t 2>/dev/null || true
}

kill_port_listeners() {
    if ! command -v lsof &>/dev/null; then
        err "未找到 lsof，无法检测/释放端口"
        exit 1
    fi

    local pids
    pids="$(port_pids)"
    if [[ -z "${pids}" ]]; then
        log "端口 $PORT 空闲，无需终止进程"
        return 0
    fi

    log "终止占用端口 $PORT 的进程: $(echo "$pids" | tr '\n' ' ')"
    while IFS= read -r pid; do
        [[ -z "$pid" ]] && continue
        kill "$pid" 2>/dev/null || true
    done <<< "$pids"

    local i
    for i in 1 2 3 4 5 6; do
        if ! port_in_use; then
            log "端口 $PORT 已释放"
            return 0
        fi
        sleep 0.5
    done

    pids="$(port_pids)"
    if [[ -n "${pids}" ]]; then
        warn "进程未退出，发送 SIGKILL: $(echo "$pids" | tr '\n' ' ')"
        while IFS= read -r pid; do
            [[ -z "$pid" ]] && continue
            kill -9 "$pid" 2>/dev/null || true
        done <<< "$pids"
    fi

    for i in 1 2 3 4; do
        if ! port_in_use; then
            log "端口 $PORT 已释放"
            return 0
        fi
        sleep 0.5
    done

    err "端口 $PORT 在终止进程后仍被占用"
    exit 1
}

ensure_prereqs() {
    if ! command -v uv &>/dev/null; then
        err "未找到 uv，请先安装: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi

    if [ ! -f ".env" ]; then
        err ".env 文件不存在，请先配置环境变量"
        exit 1
    fi
}

assert_port_free() {
    if port_in_use; then
        err "端口 $PORT 已被占用，请先释放或设置 PORT 环境变量后重试（或使用: ./start.sh restart）"
        exit 1
    fi
}

run_server() {
    log "同步依赖中..."
    uv sync --quiet

    log "启动服务 → http://${HOST}:${PORT}"
    echo ""
    exec uv run uvicorn src.main:app --host "$HOST" --port "$PORT" --reload
}

case "$ACTION" in
    start)
        ensure_prereqs
        assert_port_free
        run_server
        ;;
    restart)
        ensure_prereqs
        kill_port_listeners
        run_server
        ;;
    *)
        err "未知命令: $ACTION"
        usage
        exit 1
        ;;
esac
