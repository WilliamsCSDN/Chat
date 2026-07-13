## Context

`start.sh` 目前是单一启动脚本：检查 `uv`、`.env`、端口空闲后 `exec` 启动 uvicorn。端口被占用时直接失败。开发场景需要 `restart`：先释放端口上的旧进程，再走同一套启动流程。

约束：保持 bash、`set -euo pipefail`、通过 `PORT`/`HOST` 环境变量配置；尽量不引入新依赖。

## Goals / Non-Goals

**Goals:**

- 支持 `./start.sh restart`（以及显式 `./start.sh start`）
- `restart` 先 kill 占用 `PORT` 的 LISTEN 进程，再启动
- 无参数时行为与今天一致（启动；端口占用则报错）

**Non-Goals:**

- 不做进程 PID 文件管理或 systemd/launchd 集成
- 不改 uvicorn 参数、不引入 Docker 编排
- 不跨机器远程杀进程；仅本机 `lsof` 可见的监听进程

## Decisions

1. **子命令分发，默认 `start`**
   - `ACTION="${1:-start}"`，支持 `start` / `restart`；未知命令打印用法并退出非零。
   - 替代方案：单独 `restart.sh` —— 否决，避免两套检查逻辑漂移。

2. **按端口杀进程，而不是按进程名**
   - 使用 `lsof -i ":$PORT" -sTCP:LISTEN -t` 取 PID，再 `kill`（必要时短暂等待后 `kill -9`）。
   - 替代方案：`pkill -f uvicorn` —— 否决，可能误杀其他项目的 uvicorn。

3. **仅 `restart` 杀进程；`start` 仍严格失败**
   - 避免「以为在空端口启动」却悄悄干掉别人的服务。
   - `restart` 在杀完后复用与 `start` 相同的依赖检查与启动逻辑。

4. **kill 后短轮询确认端口释放**
   - 最多等待约 2–3 秒；仍占用则报错退出，避免半死不活状态直接 `exec`。

## Risks / Trade-offs

- [误杀同端口其他服务] → 仅杀 LISTEN PID，日志打印将终止的 PID；`start` 不杀
- [进程忽略 SIGTERM] → 等待后 `kill -9` 兜底
- [无 `lsof`] → 启动前检查，缺失则明确报错（与现网依赖一致，当前脚本已用 `lsof`）

## Migration Plan

- 直接更新 `start.sh`；开发者改用 `./start.sh restart` 即可
- 回滚：还原脚本即可，无数据迁移

## Open Questions

- 无。若后续需要 `stop` 子命令，可在同一分发结构上追加。
