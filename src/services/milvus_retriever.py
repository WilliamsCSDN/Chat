from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from src.config.config_settings import (
    MILVUS_COLLECTION,
    MILVUS_EMBEDDING_MODEL,
    MILVUS_EMBEDDING_MODEL_PATH,
    MILVUS_ENABLED,
    MILVUS_HOST,
    MILVUS_HYBRID_ENABLED,
    MILVUS_HYBRID_RRF_K,
    MILVUS_HYBRID_SPARSE_RECALL_K,
    MILVUS_MIN_TOP1_LEXICAL,
    MILVUS_MIN_TOP1_MARGIN,
    MILVUS_MIN_TOP1_SCORE,
    MILVUS_NPROBE,
    MILVUS_PORT,
    MILVUS_SEARCH_EXPR,
    MILVUS_SOURCE_FIELDS,
    MILVUS_TEXT_FIELD,
    MILVUS_TOP_K,
    MILVUS_VECTOR_FIELD,
    RAG_LOG_PREVIEW_CHARS,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievedPassage:
    text: str
    score: float
    source: str = ""
    question: str = ""
    vector_score: float = 0.0
    lexical_score: float = 0.0


@dataclass
class SparseDoc:
    text: str
    source: str
    question: str
    tf: Counter[str]
    dl: int


@dataclass(frozen=True)
class ConfidencePolicy:
    min_top1_score: float = 0.62
    min_top1_margin: float = 0.05
    min_top1_lexical: float = 0.18


PUNCTUATION_PATTERN = re.compile(r"[\s\u3000，。！？!?:：；;、'\"“”‘’\-\_/()（）\[\]【】{}<>《》]+")


def normalize_match_text(text: str) -> str:
    cleaned = PUNCTUATION_PATTERN.sub("", (text or "").strip().lower())
    return cleaned


def _build_query_terms(text: str) -> List[str]:
    normalized = normalize_match_text(text)
    if not normalized:
        return []
    terms = {normalized}
    for n in (2, 3, 4):
        if len(normalized) >= n:
            for i in range(len(normalized) - n + 1):
                terms.add(normalized[i : i + n])
    return sorted(terms, key=len, reverse=True)


def lexical_overlap_score(query: str, candidate: str) -> float:
    normalized_query = normalize_match_text(query)
    normalized_candidate = normalize_match_text(candidate)
    if not normalized_query or not normalized_candidate:
        return 0.0
    if normalized_query == normalized_candidate:
        return 1.0

    terms = _build_query_terms(normalized_query)
    if not terms:
        return 0.0

    hit_terms = [term for term in terms if term in normalized_candidate]
    if not hit_terms:
        return 0.0

    coverage = len(hit_terms) / len(terms)
    weighted = sum(len(term) for term in hit_terms) / sum(len(term) for term in terms)
    return 0.4 * coverage + 0.6 * weighted


def rerank_passages(query: str, passages: Sequence[RetrievedPassage], top_k: int) -> List[RetrievedPassage]:
    if not passages:
        return []

    normalized_query = normalize_match_text(query)
    dedup: Dict[str, RetrievedPassage] = {}
    for passage in passages:
        question = (passage.question or "").strip()
        vector_score = float(passage.vector_score or passage.score)
        lexical_score = lexical_overlap_score(query, question or passage.text)
        is_exact_match = 1.0 if question and normalize_match_text(question) == normalized_query else 0.0
        final_score = 0.65 * vector_score + 0.25 * lexical_score + 0.10 * is_exact_match
        reranked = RetrievedPassage(
            text=passage.text,
            score=final_score,
            source=passage.source,
            question=question,
            vector_score=vector_score,
            lexical_score=lexical_score,
        )
        dedup_key = normalize_match_text(question) or normalize_match_text(passage.text) or reranked.source
        existing = dedup.get(dedup_key)
        if existing is None or reranked.score > existing.score:
            dedup[dedup_key] = reranked

    ranked = sorted(
        dedup.values(),
        key=lambda item: (item.score, item.vector_score, item.lexical_score),
        reverse=True,
    )
    return ranked[:top_k]


def _fusion_key(passage: RetrievedPassage) -> str:
    question_key = normalize_match_text(passage.question)
    text_key = normalize_match_text(passage.text)
    return question_key or text_key or passage.source


def fuse_ranked_passages(
    dense_passages: Sequence[RetrievedPassage],
    sparse_passages: Sequence[RetrievedPassage],
    top_k: int,
    rrf_k: int = 60,
) -> List[RetrievedPassage]:
    if not dense_passages and not sparse_passages:
        return []

    accum: Dict[str, float] = {}
    merged: Dict[str, RetrievedPassage] = {}

    def add_passages(passages: Sequence[RetrievedPassage], channel: str) -> None:
        for rank, passage in enumerate(passages, start=1):
            key = _fusion_key(passage)
            if not key:
                continue
            accum[key] = accum.get(key, 0.0) + 1.0 / (rrf_k + rank)
            existing = merged.get(key)
            if existing is None:
                merged[key] = RetrievedPassage(
                    text=passage.text,
                    score=passage.score,
                    source=passage.source,
                    question=passage.question,
                    vector_score=passage.vector_score,
                    lexical_score=passage.lexical_score,
                )
                continue

            # 统一保留信息更完整的文本和来源；向量分取更高值。
            if channel == "dense" and passage.vector_score > existing.vector_score:
                existing.vector_score = passage.vector_score
                existing.text = passage.text or existing.text
                existing.source = passage.source or existing.source
                existing.question = passage.question or existing.question
            elif not existing.text and passage.text:
                existing.text = passage.text
                existing.source = passage.source or existing.source
                existing.question = passage.question or existing.question

    add_passages(dense_passages, "dense")
    add_passages(sparse_passages, "sparse")

    if not merged:
        return []

    max_rrf = max(accum.values()) if accum else 1.0
    fused: List[RetrievedPassage] = []
    for key, passage in merged.items():
        rrf_score = accum.get(key, 0.0) / max_rrf if max_rrf > 0 else 0.0
        dense_hint = passage.vector_score if passage.vector_score > 0 else 0.0
        fusion_score = 0.8 * rrf_score + 0.2 * dense_hint
        fused.append(
            RetrievedPassage(
                text=passage.text,
                score=fusion_score,
                source=passage.source,
                question=passage.question,
                vector_score=fusion_score,
                lexical_score=passage.lexical_score,
            )
        )

    fused.sort(key=lambda item: item.score, reverse=True)
    return fused[:top_k]


def filter_passages_by_confidence(
    passages: Sequence[RetrievedPassage],
    policy: ConfidencePolicy,
) -> List[RetrievedPassage]:
    ranked = list(passages)
    if not ranked:
        return []

    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else None
    margin = top1.score - top2.score if top2 else 1.0

    if top1.score < policy.min_top1_score:
        return []
    has_lexical_signal = top1.lexical_score > 0 or bool((top1.question or "").strip())
    if has_lexical_signal and top1.lexical_score < policy.min_top1_lexical:
        return []
    if margin < policy.min_top1_margin:
        return []

    filtered = [p for p in ranked if p.score >= policy.min_top1_score]
    return filtered or [top1]


class RetrieverProtocol(Protocol):
    def search(self, query: str, top_k: Optional[int] = None, expr: str = "") -> List[RetrievedPassage]:
        ...


class MilvusRetriever:
    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        vector_field: str,
        text_field: str,
        top_k: int,
        nprobe: int,
        embedding_model_name: str,
        embedding_model_path: str,
        source_fields: Sequence[str],
        hybrid_enabled: bool,
        hybrid_sparse_recall_k: int,
        hybrid_rrf_k: int,
        search_expr: str = "",
        alias: str = "chat_milvus",
    ) -> None:
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_field = vector_field
        self.text_field = text_field
        self.top_k = top_k
        self.nprobe = nprobe
        self.embedding_model_name = embedding_model_name
        self.embedding_model_path = embedding_model_path
        self.source_fields = list(source_fields)
        self.hybrid_enabled = hybrid_enabled
        self.hybrid_sparse_recall_k = hybrid_sparse_recall_k
        self.hybrid_rrf_k = hybrid_rrf_k
        self.search_expr = search_expr
        self.alias = alias

        self._model = None
        self._collection = None
        self._connected = False
        self._disabled = False
        self._sparse_docs: List[SparseDoc] = []
        self._sparse_df: Counter[str] = Counter()
        self._sparse_avgdl: float = 0.0
        self._sparse_ready = False

    def _resolve_model_path(self) -> Path:
        configured = Path(self.embedding_model_path)
        if configured.is_absolute():
            return configured
        project_root = Path(__file__).resolve().parent.parent
        return project_root / configured

    def _get_model(self):
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        model_path = self._resolve_model_path()
        if model_path.exists():
            logger.info("使用本地 embedding 模型: %s", model_path)
            self._model = SentenceTransformer(str(model_path))
        else:
            logger.info("使用在线 embedding 模型: %s", self.embedding_model_name)
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _get_collection(self):
        if self._collection is not None:
            return self._collection

        from pymilvus import Collection, connections

        if not self._connected:
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=str(self.port),
            )
            self._connected = True
        self._collection = Collection(name=self.collection_name, using=self.alias)
        return self._collection

    @staticmethod
    def _entity_value(entity: Any, field: str) -> Any:
        value = getattr(entity, field, None)
        if value is not None:
            return value
        if hasattr(entity, "get"):
            try:
                return entity.get(field)
            except Exception:
                return None
        return None

    def _get_output_fields(self, collection) -> List[str]:
        available_fields = {field.name for field in collection.schema.fields}
        candidates = [self.text_field, "question"] + self.source_fields
        output_fields: List[str] = []
        for field_name in candidates:
            if field_name in available_fields and field_name not in output_fields:
                output_fields.append(field_name)
        return output_fields

    def _build_source(self, entity: Any) -> str:
        parts: List[str] = []
        for field in self.source_fields:
            value = self._entity_value(entity, field)
            if value is None:
                continue
            value_text = str(value).strip()
            if value_text:
                parts.append(value_text)
        return " / ".join(parts) if parts else "Milvus"

    @staticmethod
    def _tokenize_sparse_text(text: str) -> List[str]:
        normalized = normalize_match_text(text)
        if not normalized:
            return []
        tokens: List[str] = [normalized]
        for n in (2, 3):
            if len(normalized) >= n:
                tokens.extend(normalized[i : i + n] for i in range(len(normalized) - n + 1))
        return tokens

    def _build_sparse_text(self, question: str, text: str, source: str) -> str:
        parts = [question, text, source.replace("/", " ")]
        return " ".join([part for part in parts if part])

    def _ensure_sparse_index(self, collection, output_fields: Sequence[str]) -> None:
        if self._sparse_ready:
            return
        if not self.hybrid_enabled:
            self._sparse_ready = True
            return

        try:
            query_fields = list(dict.fromkeys(output_fields))
            rows = collection.query(
                expr="id >= 0",
                output_fields=query_fields,
                limit=16384,
            )
        except Exception as exc:
            logger.warning("稀疏召回索引构建失败，回退仅向量召回: %s", exc)
            self._sparse_ready = True
            self.hybrid_enabled = False
            return

        docs: List[SparseDoc] = []
        df_counter: Counter[str] = Counter()
        total_len = 0
        for row in rows:
            question = str(row.get("question") or "").strip()
            text = str(row.get(self.text_field) or "").strip()
            if not text:
                continue
            source_parts = [str(row.get(field) or "").strip() for field in self.source_fields]
            source = " / ".join([part for part in source_parts if part]) or "Milvus"
            sparse_text = self._build_sparse_text(question, text, source)
            tokens = self._tokenize_sparse_text(sparse_text)
            if not tokens:
                continue
            tf = Counter(tokens)
            total_len += len(tokens)
            for token in tf.keys():
                df_counter[token] += 1
            docs.append(
                SparseDoc(
                    text=f"问题：{question}\n答案：{text}" if question and self.text_field == "answer" else text,
                    source=source,
                    question=question,
                    tf=tf,
                    dl=len(tokens),
                )
            )

        self._sparse_docs = docs
        self._sparse_df = df_counter
        self._sparse_avgdl = (total_len / len(docs)) if docs else 0.0
        self._sparse_ready = True
        logger.info("稀疏召回索引就绪 | docs=%s | avgdl=%.2f", len(docs), self._sparse_avgdl)

    def _sparse_recall(self, query: str, limit: int) -> List[RetrievedPassage]:
        if not self.hybrid_enabled or not self._sparse_docs:
            return []

        query_tokens = self._tokenize_sparse_text(query)
        if not query_tokens:
            return []

        unique_tokens = list(dict.fromkeys(query_tokens))
        n_docs = len(self._sparse_docs)
        if n_docs == 0:
            return []

        k1 = 1.2
        b = 0.75
        scored: List[Tuple[float, SparseDoc]] = []
        for doc in self._sparse_docs:
            score = 0.0
            for token in unique_tokens:
                tf = doc.tf.get(token, 0)
                if tf <= 0:
                    continue
                df = self._sparse_df.get(token, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1.0 - b + b * doc.dl / (self._sparse_avgdl or 1.0))
                score += idf * (tf * (k1 + 1.0)) / denom
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        passages: List[RetrievedPassage] = []
        for score, doc in scored[:limit]:
            passages.append(
                RetrievedPassage(
                    text=doc.text,
                    score=score,
                    source=doc.source,
                    question=doc.question,
                    vector_score=0.0,
                )
            )
        return passages

    def _run_hybrid_recall_parallel(
        self,
        collection,
        search_kwargs: Dict[str, Any],
        query: str,
        sparse_limit: int,
    ) -> Tuple[Any, List[RetrievedPassage]]:
        """
        并行执行 dense/sparse 两路召回：
        - dense: Milvus vector search
        - sparse: 本地 BM25 recall
        """
        if not self.hybrid_enabled:
            return collection.search(**search_kwargs), []

        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(collection.search, **search_kwargs)
            sparse_future = executor.submit(self._sparse_recall, query, sparse_limit)
            hits = dense_future.result()
            try:
                sparse_passages = sparse_future.result()
            except Exception as exc:
                logger.warning("稀疏召回执行失败，回退仅 dense: %s", exc)
                sparse_passages = []
        return hits, sparse_passages

    def search(self, query: str, top_k: Optional[int] = None, expr: str = "") -> List[RetrievedPassage]:
        query = (query or "").strip()
        if not query or self._disabled:
            return []

        limit = top_k or self.top_k
        start = time.perf_counter()
        logger.info(
            "RAG 检索开始 | collection=%s | query=%s | top_k=%s | expr=%s",
            self.collection_name,
            query,
            limit,
            expr or self.search_expr or "<empty>",
        )
        try:
            model = self._get_model()
            vector = model.encode([query], normalize_embeddings=True)[0]
            if hasattr(vector, "tolist"):
                vector = vector.tolist()

            collection = self._get_collection()
            collection.load()
            output_fields = self._get_output_fields(collection)
            if self.text_field not in output_fields:
                logger.warning(
                    "Milvus 检索字段缺失: collection=%s text_field=%s",
                    self.collection_name,
                    self.text_field,
                )
                return []

            recall_k = max(limit * 8, 30)
            search_kwargs: Dict[str, Any] = {
                "data": [vector],
                "anns_field": self.vector_field,
                "param": {
                    "metric_type": "COSINE",
                    "params": {"nprobe": self.nprobe},
                },
                "limit": recall_k,
                "output_fields": output_fields,
            }
            effective_expr = expr or self.search_expr
            if effective_expr:
                search_kwargs["expr"] = effective_expr

            sparse_limit = max(self.hybrid_sparse_recall_k, limit * 5)
            self._ensure_sparse_index(collection, output_fields)
            logger.info(
                "RAG 检索参数 | vector_field=%s | output_fields=%s | nprobe=%s | dense_recall_k=%s | sparse_recall_k=%s | parallel=%s",
                self.vector_field,
                output_fields,
                self.nprobe,
                recall_k,
                sparse_limit,
                self.hybrid_enabled,
            )
            hits, sparse_passages = self._run_hybrid_recall_parallel(
                collection=collection,
                search_kwargs=search_kwargs,
                query=query,
                sparse_limit=sparse_limit,
            )
            if not hits:
                cost_ms = (time.perf_counter() - start) * 1000
                logger.info("RAG 检索完成 | 命中=0 | 耗时=%.2fms", cost_ms)
                return []

            dense_passages: List[RetrievedPassage] = []
            for hit in hits[0]:
                entity = hit.entity
                text = self._entity_value(entity, self.text_field)
                if not text:
                    continue
                question = str(self._entity_value(entity, "question") or "").strip()
                source = self._build_source(entity)
                passage_text = str(text)
                if question and self.text_field == "answer":
                    passage_text = f"问题：{question}\n答案：{passage_text}"

                dense_passages.append(
                    RetrievedPassage(
                        text=passage_text,
                        score=float(hit.score),
                        source=source,
                        question=question,
                        vector_score=float(hit.score),
                    )
                )

            logger.info(
                "RAG 召回统计 | dense=%s | sparse=%s | hybrid=%s",
                len(dense_passages),
                len(sparse_passages),
                self.hybrid_enabled,
            )
            passages = fuse_ranked_passages(
                dense_passages=dense_passages,
                sparse_passages=sparse_passages,
                top_k=max(limit * 6, 20),
                rrf_k=self.hybrid_rrf_k,
            )
            passages = rerank_passages(query, passages, top_k=limit)
            for idx, passage in enumerate(passages, start=1):
                preview = passage.text.replace("\n", "\\n")
                if RAG_LOG_PREVIEW_CHARS > 0:
                    preview = preview[:RAG_LOG_PREVIEW_CHARS]
                logger.info(
                    "RAG 命中[%s] | score=%.4f | vector=%.4f | lexical=%.4f | source=%s | text=%s",
                    idx,
                    passage.score,
                    passage.vector_score,
                    passage.lexical_score,
                    passage.source,
                    preview,
                )
            cost_ms = (time.perf_counter() - start) * 1000
            logger.info("RAG 检索完成 | 命中=%s | 耗时=%.2fms", len(passages), cost_ms)
            return passages
        except Exception as exc:
            self._disabled = True
            cost_ms = (time.perf_counter() - start) * 1000
            logger.warning("Milvus 检索不可用，已降级为无检索模式 | 耗时=%.2fms | err=%s", cost_ms, exc)
            return []


_DEFAULT_RETRIEVER: Optional[MilvusRetriever] = None
_DEFAULT_CONFIDENCE_POLICY = ConfidencePolicy(
    min_top1_score=MILVUS_MIN_TOP1_SCORE,
    min_top1_margin=MILVUS_MIN_TOP1_MARGIN,
    min_top1_lexical=MILVUS_MIN_TOP1_LEXICAL,
)


def get_default_retriever() -> Optional[MilvusRetriever]:
    global _DEFAULT_RETRIEVER
    if not MILVUS_ENABLED:
        return None
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = MilvusRetriever(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            collection_name=MILVUS_COLLECTION,
            vector_field=MILVUS_VECTOR_FIELD,
            text_field=MILVUS_TEXT_FIELD,
            top_k=MILVUS_TOP_K,
            nprobe=MILVUS_NPROBE,
            embedding_model_name=MILVUS_EMBEDDING_MODEL,
            embedding_model_path=MILVUS_EMBEDDING_MODEL_PATH,
            source_fields=MILVUS_SOURCE_FIELDS,
            hybrid_enabled=MILVUS_HYBRID_ENABLED,
            hybrid_sparse_recall_k=MILVUS_HYBRID_SPARSE_RECALL_K,
            hybrid_rrf_k=MILVUS_HYBRID_RRF_K,
            search_expr=MILVUS_SEARCH_EXPR,
        )
    return _DEFAULT_RETRIEVER


def extract_latest_user_query(messages: Sequence[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def build_retrieval_system_message(passages: Sequence[RetrievedPassage]) -> Dict[str, str]:
    refs: List[str] = []
    for idx, passage in enumerate(passages, start=1):
        refs.append(
            f"{idx}. 来源: {passage.source} | 相似度: {passage.score:.4f}\n"
            f"   内容: {passage.text}"
        )

    content = (
        "以下是与当前问题相关的检索参考资料，请优先参考这些内容回答。"
        "若资料不足，请明确说明不确定，再结合通用知识补充。\n\n"
        "【检索参考资料】\n"
        f"{chr(10).join(refs)}"
    )
    return {"role": "system", "content": content}


async def inject_retrieval_context(
    messages: Sequence[Dict[str, Any]],
    retriever: Optional[RetrieverProtocol] = None,
) -> Tuple[List[Dict[str, Any]], List[RetrievedPassage]]:
    copied_messages: List[Dict[str, Any]] = [dict(msg) for msg in messages]
    query = extract_latest_user_query(copied_messages)
    if not query:
        logger.info("RAG 跳过：未找到用户问题")
        return copied_messages, []

    use_retriever = retriever or get_default_retriever()
    if use_retriever is None:
        logger.info("RAG 跳过：MILVUS_ENABLED=false 或 retriever 未配置")
        return copied_messages, []

    passages = await asyncio.to_thread(use_retriever.search, query)
    if not passages:
        logger.info("RAG 注入跳过：检索结果为空")
        return copied_messages, []

    filtered_passages = filter_passages_by_confidence(passages, _DEFAULT_CONFIDENCE_POLICY)
    if not filtered_passages:
        top1 = passages[0]
        top2 = passages[1] if len(passages) > 1 else None
        margin = top1.score - top2.score if top2 else 1.0
        logger.info(
            "RAG 注入跳过：置信度不足 | top1=%.4f | lexical=%.4f | margin=%.4f | gate(score>=%.2f,lexical>=%.2f,margin>=%.2f)",
            top1.score,
            top1.lexical_score,
            margin,
            _DEFAULT_CONFIDENCE_POLICY.min_top1_score,
            _DEFAULT_CONFIDENCE_POLICY.min_top1_lexical,
            _DEFAULT_CONFIDENCE_POLICY.min_top1_margin,
        )
        return copied_messages, []

    system_message = build_retrieval_system_message(filtered_passages)
    insert_at = 0
    while insert_at < len(copied_messages) and copied_messages[insert_at].get("role") == "system":
        insert_at += 1
    copied_messages.insert(insert_at, system_message)
    logger.info("RAG 注入完成 | 插入位置=%s | 注入条数=%s", insert_at, len(filtered_passages))
    return copied_messages, filtered_passages
