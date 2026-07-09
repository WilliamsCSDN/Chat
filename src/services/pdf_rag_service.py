from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from src.config.config_settings import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    MILVUS_EMBEDDING_MODEL,
    MILVUS_EMBEDDING_MODEL_PATH,
    MILVUS_HOST,
    MILVUS_PORT,
    MODEL_NAME,
    PDF_RAG_COLLECTION,
    PDF_RAG_ENABLED,
    PDF_RAG_HYBRID_ENABLED,
    PDF_RAG_HYBRID_RRF_K,
    PDF_RAG_HYBRID_SPARSE_RECALL_K,
    PDF_RAG_MIN_RECALL_K,
    PDF_RAG_NPROBE,
    PDF_RAG_RECALL_MULTIPLIER,
    PDF_RAG_SEARCH_EXPR,
    PDF_RAG_SOURCE_FIELDS,
    PDF_RAG_TEXT_FIELD,
    PDF_RAG_TOP_K,
    PDF_RAG_VECTOR_FIELD,
)
from src.services.milvus_retriever import (
    MilvusRetriever,
    RetrievedPassage,
    lexical_overlap_score,
    normalize_match_text,
)

logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
)

_PDF_RETRIEVER: Optional[MilvusRetriever] = None
_PAGE_MARKER_PATTERN = re.compile(r"^-{2}\s*\d+\s+of\s+\d+\s*-{2}$")


def get_pdf_retriever() -> Optional[MilvusRetriever]:
    global _PDF_RETRIEVER
    if not PDF_RAG_ENABLED:
        return None
    if _PDF_RETRIEVER is None:
        _PDF_RETRIEVER = MilvusRetriever(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            collection_name=PDF_RAG_COLLECTION,
            vector_field=PDF_RAG_VECTOR_FIELD,
            text_field=PDF_RAG_TEXT_FIELD,
            top_k=PDF_RAG_TOP_K,
            nprobe=PDF_RAG_NPROBE,
            embedding_model_name=MILVUS_EMBEDDING_MODEL,
            embedding_model_path=MILVUS_EMBEDDING_MODEL_PATH,
            source_fields=PDF_RAG_SOURCE_FIELDS,
            hybrid_enabled=PDF_RAG_HYBRID_ENABLED,
            hybrid_sparse_recall_k=PDF_RAG_HYBRID_SPARSE_RECALL_K,
            hybrid_rrf_k=PDF_RAG_HYBRID_RRF_K,
            search_expr=PDF_RAG_SEARCH_EXPR,
            alias="pdf_rag_milvus",
        )
    return _PDF_RETRIEVER


def _resolve_requested_top_k(top_k: Optional[int]) -> int:
    if top_k is None:
        return PDF_RAG_TOP_K
    return max(1, min(top_k, 10))


def _resolve_recall_top_k(top_k: int) -> int:
    return max(PDF_RAG_MIN_RECALL_K, top_k * max(PDF_RAG_RECALL_MULTIPLIER, 1))


def _is_low_signal_passage_text(text: str) -> bool:
    normalized = " ".join((text or "").split())
    if not normalized:
        return True
    if normalized in {"保密级别：内部资料", "保密级别:内部资料", "内部资料"}:
        return True
    if _PAGE_MARKER_PATTERN.match(normalized.lower()):
        return True
    if len(normalized) <= 12 and "保密级别" in normalized:
        return True
    return False


def rerank_pdf_passages(
    query: str,
    passages: List[RetrievedPassage],
    top_k: int,
) -> List[RetrievedPassage]:
    if not passages:
        return []

    deduped: Dict[str, RetrievedPassage] = {}
    for passage in passages:
        clean_text = (passage.text or "").strip()
        if not clean_text:
            continue
        source = (passage.source or "").strip()
        vector_score = float(passage.vector_score or passage.score)
        lexical_score = lexical_overlap_score(query, f"{source}\n{clean_text}")
        low_signal = _is_low_signal_passage_text(clean_text)
        fused_score = 0.7 * vector_score + 0.3 * lexical_score
        if low_signal:
            fused_score *= 0.55

        reranked = RetrievedPassage(
            text=clean_text,
            score=fused_score,
            source=source,
            question=passage.question,
            vector_score=vector_score,
            lexical_score=lexical_score,
        )
        dedup_key = normalize_match_text(f"{source}\n{clean_text}")
        existing = deduped.get(dedup_key)
        if existing is None or reranked.score > existing.score:
            deduped[dedup_key] = reranked

    ranked = sorted(
        deduped.values(),
        key=lambda item: (item.score, item.lexical_score, item.vector_score),
        reverse=True,
    )
    preferred = [item for item in ranked if not _is_low_signal_passage_text(item.text)]
    if len(preferred) < top_k:
        leftovers = [item for item in ranked if item not in preferred]
        preferred.extend(leftovers)
    return preferred[:top_k]


def _build_pdf_context(passages: List[RetrievedPassage]) -> str:
    refs: List[str] = []
    for idx, passage in enumerate(passages, start=1):
        refs.append(
            f"{idx}. 来源: {passage.source} | 相似度: {passage.score:.4f}\n"
            f"   内容: {passage.text}"
        )
    return "【PDF 检索片段】\n" + "\n".join(refs)


async def query_pdf_rag(query: str, model: Optional[str] = None, top_k: Optional[int] = None) -> Dict[str, Any]:
    clean_query = (query or "").strip()
    if not clean_query:
        raise ValueError("query 不能为空")

    retriever = get_pdf_retriever()
    if retriever is None:
        raise RuntimeError("PDF_RAG_ENABLED=false，未启用 PDF-RAG")

    requested_top_k = _resolve_requested_top_k(top_k)
    recall_top_k = _resolve_recall_top_k(requested_top_k)
    raw_passages = await asyncio.to_thread(retriever.search, clean_query, recall_top_k)
    passages = rerank_pdf_passages(clean_query, raw_passages, requested_top_k)
    if not passages:
        return {
            "query": clean_query,
            "answer": "未在 PDF 知识库中检索到相关内容，请尝试更具体的关键词。",
            "passages": [],
        }

    context = _build_pdf_context(passages)
    answer_model = model or MODEL_NAME
    completion = await client.chat.completions.create(
        model=answer_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是公司 PDF 制度问答助手。请严格依据给定检索片段回答，"
                    "不要编造未出现的信息。回答末尾附上你参考的来源。"
                ),
            },
            {"role": "system", "content": context},
            {"role": "user", "content": clean_query},
        ],
        stream=False,
    )

    answer = (completion.choices[0].message.content or "").strip()
    if not answer:
        answer = "已检索到相关片段，但模型未返回文本，请稍后重试。"

    payload_passages: List[Dict[str, Any]] = []
    for item in passages:
        payload_passages.append(
            {
                "text": item.text,
                "score": item.score,
                "lexical_score": item.lexical_score,
                "source": item.source,
            }
        )

    logger.info(
        "PDF-RAG 查询完成 | query=%s | passages=%s | recall_top_k=%s | final_top_k=%s | model=%s",
        clean_query,
        len(payload_passages),
        recall_top_k,
        requested_top_k,
        answer_model,
    )
    return {
        "query": clean_query,
        "answer": answer,
        "passages": payload_passages,
    }
