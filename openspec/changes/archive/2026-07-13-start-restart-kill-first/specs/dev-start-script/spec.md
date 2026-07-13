## ADDED Requirements

### Requirement: Start script supports start and restart commands

开发启动脚本 MUST 接受可选子命令 `start` 与 `restart`。省略子命令时 MUST 等价于 `start`。未知子命令 MUST 以非零状态退出并提示用法。

#### Scenario: Default invocation starts the service

- **WHEN** 用户执行 `./start.sh`（无参数）且目标端口空闲
- **THEN** 脚本 MUST 同步依赖并以配置的 host/port 启动 uvicorn

#### Scenario: Explicit start with occupied port fails

- **WHEN** 用户执行 `./start.sh start`（或无参数）且目标端口已被占用
- **THEN** 脚本 MUST 报错退出且 MUST NOT 终止占用该端口的进程

#### Scenario: Unknown command is rejected

- **WHEN** 用户执行 `./start.sh` 并传入非 `start`/`restart` 的子命令
- **THEN** 脚本 MUST 打印用法并以非零状态退出

### Requirement: Restart kills listeners on the target port before starting

当用户执行 `restart` 时，脚本 MUST 先终止占用目标端口（`PORT`，默认 `8000`）的 TCP LISTEN 进程，在确认端口可用后再执行与 `start` 相同的启动流程。

#### Scenario: Restart with occupied port succeeds

- **WHEN** 用户执行 `./start.sh restart` 且目标端口上有本机监听进程
- **THEN** 脚本 MUST 终止这些监听进程，MUST 在端口释放后启动服务

#### Scenario: Restart with free port starts normally

- **WHEN** 用户执行 `./start.sh restart` 且目标端口空闲
- **THEN** 脚本 MUST 跳过杀进程步骤并正常启动服务

#### Scenario: Restart fails if port remains occupied after kill

- **WHEN** 用户执行 `./start.sh restart` 且杀进程后目标端口在短等待内仍被占用
- **THEN** 脚本 MUST 报错并以非零状态退出，MUST NOT 继续启动
