#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8001}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { printf "${GREEN}[INFO]${NC}  %s\n" "$*"; }
warn() { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()  { printf "${RED}[ERROR]${NC} %s\n" "$*"; }

usage() {
    printf "用法: %s [stop]\n" "$(basename "$0")"
    printf "  stop  停止占用 PORT 的本项目 uvicorn 服务（默认）\n"
}

port_in_use() {
    lsof -i ":$PORT" -sTCP:LISTEN -t >/dev/null 2>&1
}

port_pids() {
    lsof -i ":$PORT" -sTCP:LISTEN -t 2>/dev/null || true
}

uvicorn_pids() {
    local pid command

    while IFS= read -r pid; do
        [[ -z "$pid" ]] && continue
        command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
        case " $command " in
            *" --port $PORT "*) printf '%s\n' "$pid" ;;
            *" --port $PORT")  printf '%s\n' "$pid" ;;
        esac
    done < <(pgrep -f "[u]vicorn src.main:app" 2>/dev/null || true)
}

wait_for_port_release() {
    local attempts="$1"
    local i

    for ((i = 0; i < attempts; i++)); do
        if ! port_in_use; then
            return 0
        fi
        sleep 0.5
    done

    return 1
}

stop_server() {
    if ! command -v lsof &>/dev/null; then
        err "未找到 lsof，无法检测/释放端口"
        exit 1
    fi

    if ! command -v pgrep &>/dev/null || ! command -v ps &>/dev/null; then
        err "未找到 pgrep/ps，无法安全识别本项目进程"
        exit 1
    fi

    if ! port_in_use; then
        log "端口 $PORT 空闲，服务已停止"
        return 0
    fi

    local app_pids
    app_pids="$(uvicorn_pids)"
    if [[ -z "$app_pids" ]]; then
        err "端口 $PORT 有进程监听，但未找到本项目 uvicorn 进程，已停止以避免误杀"
        lsof -nP -i ":$PORT" -sTCP:LISTEN || true
        exit 1
    fi

    local listener_pids
    listener_pids="$(port_pids)"
    log "停止服务进程: $(echo "$listener_pids" | tr '\n' ' ')"

    while IFS= read -r pid; do
        [[ -z "$pid" ]] && continue
        kill "$pid" 2>/dev/null || true
    done <<< "$listener_pids"

    if ! wait_for_port_release 6; then
        listener_pids="$(port_pids)"
        warn "进程未退出，发送 SIGKILL: $(echo "$listener_pids" | tr '\n' ' ')"
        while IFS= read -r pid; do
            [[ -z "$pid" ]] && continue
            kill -9 "$pid" 2>/dev/null || true
        done <<< "$listener_pids"
    fi

    if ! wait_for_port_release 4; then
        err "端口 $PORT 在终止进程后仍被占用"
        exit 1
    fi

    log "端口 $PORT 已释放"
}

ACTION="${1:-stop}"

case "$ACTION" in
    stop)
        stop_server
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        err "未知命令: $ACTION"
        usage
        exit 1
        ;;
esac
