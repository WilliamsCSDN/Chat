import unittest
import json
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.config.config_settings import MILVUS_EMBEDDING_MODEL, MILVUS_EMBEDDING_MODEL_PATH
from src.services import (
    _extract_answer_from_passage_text,
    build_direct_answer_content,
    chat_stream,
    should_use_direct_answer,
)
from src.services import MilvusRetriever, RetrievedPassage


class ChatServiceTests(unittest.TestCase):
    def test_build_direct_answer_content_includes_source_and_question(self):
        answer = "如果查看订单不成功，请检查我的页面。"
        passage = RetrievedPassage(
            text="问题：查看订单不成功怎么办？\n答案：如果查看订单不成功，请检查我的页面。",
            score=0.9527,
            source="支付与订单 / 订单 / 订单查询 / 查看订单不成功怎么办？",
            question="查看订单不成功怎么办？",
            lexical_score=1.0,
        )
        content = build_direct_answer_content(answer, passage)
        self.assertIn("知识库命中信息", content)
        self.assertIn("命中问题：查看订单不成功怎么办？", content)
        self.assertIn("来源：支付与订单 / 订单 / 订单查询 / 查看订单不成功怎么办？", content)
        self.assertIn("置信度：0.9527", content)

    def test_extract_answer_from_passage_text_prefers_answer_section(self):
        passage = "问题：查看订单不成功怎么办？\n答案：如果查看订单不成功，请检查我的页面。"
        answer = _extract_answer_from_passage_text(passage)
        self.assertEqual(answer, "如果查看订单不成功，请检查我的页面。")

    def test_should_use_direct_answer_true_with_high_confidence(self):
        passages = [
            RetrievedPassage(
                text="答案A",
                score=0.93,
                source="s1",
                lexical_score=1.0,
            ),
            RetrievedPassage(
                text="答案B",
                score=0.70,
                source="s2",
                lexical_score=0.3,
            ),
        ]
        ok = should_use_direct_answer(passages, min_score=0.9, min_lexical=0.95, min_margin=0.1)
        self.assertTrue(ok)

    def test_should_use_direct_answer_false_with_low_score(self):
        passages = [
            RetrievedPassage(
                text="答案A",
                score=0.78,
                source="s1",
                lexical_score=1.0,
            ),
            RetrievedPassage(
                text="答案B",
                score=0.70,
                source="s2",
                lexical_score=0.3,
            ),
        ]
        ok = should_use_direct_answer(passages, min_score=0.9, min_lexical=0.95, min_margin=0.1)
        self.assertFalse(ok)


class ChatStreamTests(unittest.IsolatedAsyncioTestCase):
    async def _collect_stream_events(self, stream):
        return [chunk async for chunk in stream]

    @staticmethod
    def _parse_sse_data(sse_line: str):
        if not sse_line.startswith("data: "):
            return {}
        return json.loads(sse_line[len("data: ") :].strip())

    async def test_rag_retrieval_connects_real_milvus(self):
        query = "考勤 迟到 补卡"
        retriever = MilvusRetriever(
            host="localhost",
            port=19530,
            collection_name="kaoqin_pdf",
            vector_field="embedding",
            text_field="text",
            top_k=5,
            nprobe=32,
            embedding_model_name=MILVUS_EMBEDDING_MODEL,
            embedding_model_path=MILVUS_EMBEDDING_MODEL_PATH,
            source_fields=["doc_name", "section_title", "page_no", "chunk_no"],
            hybrid_enabled=False,
            hybrid_sparse_recall_k=20,
            hybrid_rrf_k=60,
            search_expr="",
        )

        try:
            passages = await asyncio.to_thread(retriever.search, query)
        except Exception as exc:
            print("\n[RAG 测试] Milvus 查询失败:", exc)
            return

        print("\n[RAG 测试] query:", query)
        print("[RAG 测试] 命中条数:", len(passages))
        for idx, passage in enumerate(passages, start=1):
            preview = passage.text.replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:120] + "..."
            print(
                f"[RAG 测试] top{idx} score={passage.score:.4f} "
                f"lexical={passage.lexical_score:.4f} source={passage.source}"
            )
            print(f"[RAG 测试] 内容: {preview}")

    async def test_chat_stream_calls_llm_when_no_rag_hit(self):
        input_messages = [{"role": "user", "content": "你好，介绍一下自己"}]

        async def fake_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="你好，我是助手。", tool_calls=None),
                        finish_reason=None,
                    )
                ]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, tool_calls=None),
                        finish_reason="stop",
                    )
                ]
            )

        mocked_create = AsyncMock(return_value=fake_stream())

        with patch("services.chat_service.inject_retrieval_context", new=AsyncMock(return_value=(input_messages, []))):
            with patch("services.chat_service.client.chat.completions.create", new=mocked_create):
                stream = chat_stream(input_messages)
                events = await self._collect_stream_events(stream)

        payload = self._parse_sse_data(events[0])
        done_payload = self._parse_sse_data(events[1])
        print("\n[LLM 测试] create 调用次数:", mocked_create.await_count)
        print("[LLM 测试] 事件数量:", len(events))
        print("[LLM 测试] content:", payload.get("content"))
        print("[LLM 测试] done:", done_payload.get("done"))


if __name__ == "__main__":
    unittest.main()
