## Why

当前 `start.sh` 在端口已被占用时会直接报错退出，没有 `restart` 能力。开发时经常需要「先停掉旧进程再拉起」，手工 `lsof`/`kill` 容易遗漏，也和日常 `./start.sh restart` 的习惯不一致。

## What Changes

- 为 `start.sh` 增加子命令：`start`（默认）与 `restart`
- `restart` 时：先查找并终止占用目标端口（默认 `8000`）的监听进程，确认端口释放后再执行启动流程
- `start` 行为保持现有校验：端口仍被占用则报错退出，不静默杀进程
- 无 **BREAKING** 变更：不传参数时仍等价于启动；仅新增 `restart` 路径

## Capabilities

### New Capabilities

- `dev-start-script`: 本地开发启动脚本的子命令与重启语义（启动前依赖检查、端口占用策略、restart 先杀后启）

### Modified Capabilities

- （无）当前 `openspec/specs/` 下尚无既有能力规格

## Impact

- 影响文件：`start.sh`
- 依赖：沿用现有 `lsof` / `kill`（macOS/Linux 开发环境常见工具）
- 不影响 FastAPI 应用代码、API 或前端
