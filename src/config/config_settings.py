import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 阿里云百炼 DashScope API 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.5-plus")

# 百炼 OpenAI 兼容模式的 base_url
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_csv(name: str, default: str) -> list[str]:
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() or default


# Milvus 检索配置
MILVUS_ENABLED = _get_bool("MILVUS_ENABLED", True)
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = _get_int("MILVUS_PORT", 19530)
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "app_faq")
MILVUS_VECTOR_FIELD = os.getenv("MILVUS_VECTOR_FIELD", "embedding")
MILVUS_TEXT_FIELD = os.getenv("MILVUS_TEXT_FIELD", "answer")
MILVUS_SOURCE_FIELDS = _get_csv(
    "MILVUS_SOURCE_FIELDS",
    "category_l1,category_l2,category_l3,question",
)
MILVUS_TOP_K = _get_int("MILVUS_TOP_K", 3)
MILVUS_NPROBE = _get_int("MILVUS_NPROBE", 32)
MILVUS_SEARCH_EXPR = os.getenv("MILVUS_SEARCH_EXPR", "").strip()
MILVUS_HYBRID_ENABLED = _get_bool("MILVUS_HYBRID_ENABLED", True)
MILVUS_HYBRID_SPARSE_RECALL_K = _get_int("MILVUS_HYBRID_SPARSE_RECALL_K", 40)
MILVUS_HYBRID_RRF_K = _get_int("MILVUS_HYBRID_RRF_K", 60)
MILVUS_MIN_TOP1_SCORE = float(os.getenv("MILVUS_MIN_TOP1_SCORE", "0.62"))
MILVUS_MIN_TOP1_MARGIN = float(os.getenv("MILVUS_MIN_TOP1_MARGIN", "0.05"))
MILVUS_MIN_TOP1_LEXICAL = float(os.getenv("MILVUS_MIN_TOP1_LEXICAL", "0.18"))
RAG_DIRECT_ANSWER_ENABLED = _get_bool("RAG_DIRECT_ANSWER_ENABLED", True)
RAG_DIRECT_ANSWER_SCORE = float(os.getenv("RAG_DIRECT_ANSWER_SCORE", "0.90"))
RAG_DIRECT_ANSWER_LEXICAL = float(os.getenv("RAG_DIRECT_ANSWER_LEXICAL", "0.95"))
RAG_DIRECT_ANSWER_MARGIN = float(os.getenv("RAG_DIRECT_ANSWER_MARGIN", "0.10"))
RAG_DIRECT_ANSWER_INCLUDE_SOURCE = _get_bool("RAG_DIRECT_ANSWER_INCLUDE_SOURCE", True)
MILVUS_EMBEDDING_MODEL = os.getenv(
    "MILVUS_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
MILVUS_EMBEDDING_MODEL_PATH = os.getenv(
    "MILVUS_EMBEDDING_MODEL_PATH",
    "../../test/milvus/models/paraphrase-multilingual-MiniLM-L12-v2",
)

# 日志配置
LOG_LEVEL = _get_str("LOG_LEVEL", "INFO").upper()
LOG_TO_FILE = _get_bool("LOG_TO_FILE", True)
LOG_FILE_PATH = _get_str("LOG_FILE_PATH", ".log/app.log")
SDK_HTTP_DEBUG = _get_bool("SDK_HTTP_DEBUG", True)
CHAT_LOG_FULL_MESSAGES = _get_bool("CHAT_LOG_FULL_MESSAGES", True)

# 输入安全守卫配置
INPUT_GUARD_ENABLED = _get_bool("INPUT_GUARD_ENABLED", True)

# Agent 循环与上下文压缩保护
AGENT_MODEL_CALL_LIMIT = _get_int("AGENT_MODEL_CALL_LIMIT", 12)
AGENT_RECURSION_LIMIT = _get_int("AGENT_RECURSION_LIMIT", 80)
SUMMARY_TRIGGER_TOKENS = _get_int("SUMMARY_TRIGGER_TOKENS", 4000)
SUMMARY_TRIGGER_MESSAGES = _get_int("SUMMARY_TRIGGER_MESSAGES", 30)
SUMMARY_KEEP_MESSAGES = _get_int("SUMMARY_KEEP_MESSAGES", 12)

CHAT_LOG_STREAM_CHUNKS = _get_bool("CHAT_LOG_STREAM_CHUNKS", True)
CHAT_LOG_TRUNCATE_CHARS = _get_int("CHAT_LOG_TRUNCATE_CHARS", 0)
RAG_LOG_PREVIEW_CHARS = _get_int("RAG_LOG_PREVIEW_CHARS", 0)

# PDF-RAG 检索配置
PDF_RAG_ENABLED = _get_bool("PDF_RAG_ENABLED", True)
PDF_RAG_COLLECTION = os.getenv("PDF_RAG_COLLECTION", "kaoqin_pdf")
PDF_RAG_VECTOR_FIELD = os.getenv("PDF_RAG_VECTOR_FIELD", "embedding")
PDF_RAG_TEXT_FIELD = os.getenv("PDF_RAG_TEXT_FIELD", "text")
PDF_RAG_SOURCE_FIELDS = _get_csv(
    "PDF_RAG_SOURCE_FIELDS",
    "doc_name,section_title,page_no,chunk_no",
)
PDF_RAG_TOP_K = _get_int("PDF_RAG_TOP_K", 3)
PDF_RAG_NPROBE = _get_int("PDF_RAG_NPROBE", 32)
PDF_RAG_HYBRID_ENABLED = _get_bool("PDF_RAG_HYBRID_ENABLED", True)
PDF_RAG_HYBRID_SPARSE_RECALL_K = _get_int("PDF_RAG_HYBRID_SPARSE_RECALL_K", 40)
PDF_RAG_HYBRID_RRF_K = _get_int("PDF_RAG_HYBRID_RRF_K", 60)
PDF_RAG_SEARCH_EXPR = os.getenv("PDF_RAG_SEARCH_EXPR", "").strip()
PDF_RAG_MIN_RECALL_K = _get_int("PDF_RAG_MIN_RECALL_K", 12)
PDF_RAG_RECALL_MULTIPLIER = _get_int("PDF_RAG_RECALL_MULTIPLIER", 6)


# SQLite 会话存储配置
SQLITE_DB_PATH = _get_str("SQLITE_DB_PATH", "data/chat_sessions.db")
