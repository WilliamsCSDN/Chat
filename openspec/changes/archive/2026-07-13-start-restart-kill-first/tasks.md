## 1. Script structure

- [x] 1.1 为 `start.sh` 增加子命令解析（默认 `start`，支持 `restart`，未知命令打印用法并非零退出）
- [x] 1.2 抽取公共启动流程函数（检查 `uv`、`.env`、`uv sync`、启动 uvicorn），供 `start`/`restart` 复用

## 2. Restart kill-then-start

- [x] 2.1 实现按 `PORT` 查找 TCP LISTEN PID 并终止（SIGTERM，必要时 SIGKILL），并短轮询确认端口释放
- [x] 2.2 `restart`：先执行杀进程逻辑，成功后再走公共启动流程；`start`：端口占用仍直接报错不杀进程

## 3. Verify

- [x] 3.1 本地验证：`./start.sh` / `./start.sh start` 在端口占用时报错；`./start.sh restart` 先杀后启可用
