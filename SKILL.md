---
name: chat-patterns
description: Coding patterns extracted from the Chat (RAG LLM service) repository
version: 1.0.0
source: local-git-analysis
analyzed_commits: 8
---

# Chat Patterns

This project is a Python-based RAG (Retrieval-Augmented Generation) chat service built with FastAPI, integrating Alibaba Cloud's DashScope LLM (Qwen) and Milvus vector database for knowledge retrieval.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Framework | FastAPI + Uvicorn |
| LLM | Qwen (via DashScope, OpenAI-compatible API) |
| Vector DB | Milvus (pymilvus) |
| Embeddings | DashScope text-embedding-v4 (OpenAI-compatible API) |
| Hybrid Search | Dense (COSINE) + Sparse (BM25 in-memory) |
| Fusion | RRF (Reciprocal Rank Fusion) |
| Frontend | Static HTML/CSS/JS |
| Deps | `uv` + `pyproject.toml` |

## Code Architecture

```
.
├── config/                  # Config submodules (logging, init)
│   ├── _init_.py
│   └── log.py
├── config_settings.py       # Centralized settings (env vars, helpers)
├── entity/
│   └── ApiResponse.py       # Response DTO
├── main.py                  # FastAPI app entry point, routes
├── services/
│   ├── chat_service.py      # LLM streaming + tool calling + RAG injection
│   ├── milvus_retriever.py  # Milvus retriever (dense + BM25 hybrid, fusion, confidence gating)
│   └── pdf_rag_service.py   # PDF RAG service (separate collection, reranking)
├── static/                  # Frontend files
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tools/
│   ├── __init__.py
│   └── weather_tool.py      # Example tool (LangChain @tool)
├── test/
│   ├── milvus/              # Milvus scripts (import, query, benchmark, embedding)
│   ├── services/            # Service tests (chat, milvus_rag, pdf_rag)
│   ├── fastapi/             # FastAPI test scripts
│   ├── langchain/           # LangChain test scripts
│   ├── openai/              # OpenAI context test
│   └── start/               # Python basics tests
├── docs/
│   └── rag-optimization-notes.md
├── pyproject.toml
└── requirements.txt
```

## Commit Conventions

This project uses **conventional commits** in Chinese:
- `feat:` - New features (e.g., `feat: 新增rag - pdf格式`)
- `test:` - Test additions (e.g., `test: 新增qwen模型调用测试流式输出`)
- `init` - Initial commit
- Plain English messages for minor updates (e.g., `test: add py test`)

## Configuration Pattern

All configuration is centralized in `config_settings.py`:
- Environment variables loaded via `python-dotenv`
- Type helpers: `_get_bool()`, `_get_int()`, `_get_csv()`, `_get_str()`
- Settings are imported by services rather than reading `os.getenv` inline
- New features add their own section of settings variables (e.g., `PDF_RAG_*`)

## RAG Architecture

### Multi-Path Recall (多路召回)
1. **Dense recall**: Milvus vector search (COSINE similarity)
2. **Sparse recall**: In-memory BM25 over collection data
3. Both run in parallel via `ThreadPoolExecutor`

### Fusion Strategy
- RRF (Reciprocal Rank Fusion) with configurable `rrf_k` (default 60)
- Final score: `0.8 * rrf_score + 0.2 * dense_hint`

### Confidence Gating (置信门控)
- `min_top1_score` (0.62): Minimum fusion score threshold
- `min_top1_margin` (0.05): Minimum gap between Top1 and Top2
- `min_top1_lexical` (0.18): Minimum lexical overlap score
- If any threshold fails, retrieval context is not injected

### Direct Answer (高置信直出)
- When confidence is very high, skip LLM and output answer directly
- Configurable via `RAG_DIRECT_ANSWER_*` settings
- Extracts answer after "答案：" pattern from FAQ entries

## Service Patterns

### Retriever Pattern
Each RAG service has:
1. A `MilvusRetriever` class instance (lazy-initialized singleton)
2. A `search()` method that runs hybrid recall
3. A `rerank_passages()` function for re-scoring
4. Confidence gating before injection

### Tool Calling Pattern
- Tools defined in `tools/` with LangChain `@tool` decorator
- Registered in `chat_service.py` via `AVAILABLE_TOOLS` dict
- Schema generated via `convert_to_openai_tool()`
- Tool calls handled in streaming loop (max 5 rounds)

### Streaming Pattern
- All LLM calls use `stream=True`
- SSE format: `data: {json}\n\n`
- Timing metrics logged at each phase (RAG, API, tool, total)

## Logging Pattern

Structured logging with Chinese labels:
- Request tracking via `request_id` (short UUID)
- Timing metrics: `rag_elapsed`, `api_elapsed`, `tool_elapsed`, `total_elapsed`
- Stream chunks optionally logged with truncation
- Performance stats: `ttfb`, `stream_elapsed`, `round_cost_ms`

## Dependency Management

- Uses `uv` exclusively (no `pip`, `poetry`, `pipenv`)
- `pyproject.toml` defines dependencies
- `uv.lock` for reproducible installs
- Install: `uv sync`
- Add: `uv add <package>`
- Run: `uv run uvicorn main:app --reload`

## Test Organization

- `test/services/` - Service-level tests (prefixed `test_`)
- `test/milvus/` - Milvus utility scripts and integration tests
- `test/fastapi/`, `test/langchain/`, `test/openai/` - Integration test scripts
- `test/start/` - Python language basics tests
- PDF test files stored alongside scripts (e.g., `test/milvus/kaoqin.pdf`)
- Embedded models stored under `test/milvus/models/`

## Adding a New RAG Feature

1. Add settings to `config_settings.py` with `_get_*` helpers
2. Create service in `services/<name>_rag_service.py`
3. Implement retriever or reuse `MilvusRetriever` with new collection
4. Add route in `main.py` with Pydantic request model
5. Write tests in `test/services/test_<name>_rag_service.py`
6. Update `README.md` with new env vars and usage instructions
7. Update `.gitignore` if needed (e.g., `test/*.pdf`, models)

## Adding a New Tool

1. Define function in `tools/<name>_tool.py` with `@tool` decorator
2. Import in `services/chat_service.py`
3. Add to `AVAILABLE_TOOLS` dict
4. Add to `TOOLS_SCHEMA` list via `convert_to_openai_tool()`

## Key Design Decisions

- **Graceful degradation**: If Milvus is unavailable, service falls back to plain conversation
- **Singleton retrievers**: Lazy-initialized with module-level globals (`_DEFAULT_RETRIEVER`, `_PDF_RETRIEVER`)
- **Async-to-thread bridging**: Blocking Milvus calls wrapped in `asyncio.to_thread()`
- **Dataclass for data transfer**: `RetrievedPassage`, `SparseDoc`, `ConfidencePolicy`
- **Protocol for abstraction**: `RetrieverProtocol` enables testability
