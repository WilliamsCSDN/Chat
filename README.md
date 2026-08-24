# Chat

Ali LLM chat demo.

## 环境要求

- Python 3.10+
- `uv`（建议使用最新版）

## 依赖管理（仅 uv）

本项目只使用 `uv` 管理依赖，依赖定义在 `pyproject.toml`，锁文件为 `uv.lock`。

### 安装依赖

```bash
uv sync
```

### 新增依赖

```bash
uv add <package>
```

### 更新锁文件

```bash
uv lock
```

## 配置环境变量

在项目根目录创建 `.env` 文件：

```bash
cat > .env <<'EOF'
DASHSCOPE_API_KEY=你的阿里云百炼Key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen3.5-plus
MILVUS_EMBEDDING_MODEL=text-embedding-v4
MILVUS_ENABLED=true
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=app_faq
MILVUS_TOP_K=3
LOG_LEVEL=INFO
SDK_HTTP_DEBUG=true
CHAT_LOG_FULL_MESSAGES=true
CHAT_LOG_STREAM_CHUNKS=true
CHAT_LOG_TRUNCATE_CHARS=0
RAG_LOG_PREVIEW_CHARS=0
EOF
```

> `MODEL_NAME` 可选，不填时默认使用 `qwen3.5-plus`。
>
> 向量检索相关可选项：
>
> - `MILVUS_VECTOR_FIELD`（默认 `embedding`）
> - `MILVUS_TEXT_FIELD`（默认 `answer`）
> - `MILVUS_SOURCE_FIELDS`（默认 `category_l1,category_l2,category_l3,question`，用于日志和上下文来源）
> - `MILVUS_NPROBE`（默认 `32`）
> - `MILVUS_SEARCH_EXPR`（默认空，用于 Milvus 表达式过滤）
> - `MILVUS_HYBRID_ENABLED`（默认 `true`，开启向量+BM25双路召回）
> - `MILVUS_HYBRID_SPARSE_RECALL_K`（默认 `40`，BM25召回候选数）
> - `MILVUS_HYBRID_RRF_K`（默认 `60`，RRF 融合参数）
> - `MILVUS_MIN_TOP1_SCORE`（默认 `0.62`，低于该分值则放弃 RAG 注入）
> - `MILVUS_MIN_TOP1_MARGIN`（默认 `0.05`，Top1 与 Top2 差距不足则放弃注入）
> - `MILVUS_MIN_TOP1_LEXICAL`（默认 `0.18`，词面相关性过低则放弃注入）
> - `RAG_DIRECT_ANSWER_ENABLED`（默认 `true`，高置信命中时直接输出 FAQ 答案）
> - `RAG_DIRECT_ANSWER_SCORE`（默认 `0.90`，直出所需最低融合分）
> - `RAG_DIRECT_ANSWER_LEXICAL`（默认 `0.95`，直出所需最低词面分）
> - `RAG_DIRECT_ANSWER_MARGIN`（默认 `0.10`，Top1 相对 Top2 的最小领先幅度）
> - `RAG_DIRECT_ANSWER_INCLUDE_SOURCE`（默认 `true`，直出时附带“命中问题/来源/置信度”）
> - `MILVUS_EMBEDDING_MODEL`（默认 `text-embedding-v4`，通过阿里云 DashScope 生成向量）
> - `MILVUS_EMBEDDING_MODEL_PATH`（保留兼容字段，当前不再用于加载本地模型）
> - `LOG_LEVEL`（默认 `INFO`，调试可用 `DEBUG`）
> - `SDK_HTTP_DEBUG`（默认 `true`，打印 OpenAI/httpx 底层请求日志）
> - `CHAT_LOG_FULL_MESSAGES`（默认 `true`，打印调用模型前后消息）
> - `CHAT_LOG_STREAM_CHUNKS`（默认 `true`，打印流式 token/tool 分片）
> - `CHAT_LOG_TRUNCATE_CHARS`（默认 `0`，表示不截断消息日志）
> - `RAG_LOG_PREVIEW_CHARS`（默认 `0`，表示不截断命中内容日志）

## 启动项目

### 开发模式（推荐）

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

### 生产模式

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8080
```

启动后访问：

- 首页: `http://127.0.0.1:8080/`
- API 文档: `http://127.0.0.1:8080/docs`
- 健康检查: `http://127.0.0.1:8080/health`

## Milvus 向量检索说明

- `chat` 服务会在每次请求时，自动用最后一条用户消息做向量检索（Milvus）。
- 检索命中的片段会作为系统上下文注入，再交给大模型生成回复。
- 若 Milvus 不可用或检索失败，会自动降级为普通对话，不影响服务可用性。

## 快速验证

```bash
curl "http://127.0.0.1:8080/health"
```

期望返回：

```text
{"status":"ok"}
```
