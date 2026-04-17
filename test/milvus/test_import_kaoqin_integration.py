from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MILVUS_TEST_DIR = PROJECT_ROOT / "test" / "milvus"
if str(MILVUS_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(MILVUS_TEST_DIR))

import import_kaoqin_pdf as kaoqin_import
from config_settings import MILVUS_EMBEDDING_MODEL, MILVUS_EMBEDDING_MODEL_PATH
from services.milvus_retriever import MilvusRetriever


class KaoqinPdfImportIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from pymilvus import Collection, connections, utility
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"pymilvus 不可用，跳过集成测试: {exc}")

        cls._Collection = Collection
        cls._utility = utility

        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        try:
            connections.connect(alias="default", host=host, port=port)
            utility.list_collections()
        except Exception as exc:
            raise unittest.SkipTest(f"Milvus 不可连接，跳过集成测试: {exc}")

        cls.collection_name = f"kaoqin_pdf_it_{int(time.time())}"
        cls.pdf_path = MILVUS_TEST_DIR / "kaoqin.pdf"
        if not cls.pdf_path.exists():
            raise unittest.SkipTest(f"测试文件不存在: {cls.pdf_path}")

        page_texts = kaoqin_import.collect_pdf_pages(cls.pdf_path)
        cls.records = kaoqin_import.build_chunk_records(
            doc_name=cls.pdf_path.name,
            page_texts=page_texts,
            max_sentences=kaoqin_import.DEFAULT_MAX_SENTENCES,
        )
        if not cls.records:
            raise unittest.SkipTest("PDF 切分结果为空，无法执行导入验证")

        cls.collection = kaoqin_import.init_collection(
            collection_name=cls.collection_name,
            dim=kaoqin_import.DEFAULT_DIM,
            drop_existing=True,
        )
        kaoqin_import.insert_records(cls.collection, cls.records, batch_size=kaoqin_import.DEFAULT_BATCH_SIZE)
        kaoqin_import.ensure_index(cls.collection)

        cls.retriever = MilvusRetriever(
            host=host,
            port=int(port),
            collection_name=cls.collection_name,
            vector_field="embedding",
            text_field="text",
            top_k=5,
            nprobe=32,
            embedding_model_name=MILVUS_EMBEDDING_MODEL,
            embedding_model_path=MILVUS_EMBEDDING_MODEL_PATH,
            source_fields=["doc_name", "section_title", "page_no", "chunk_no"],
            hybrid_enabled=True,
            hybrid_sparse_recall_k=40,
            hybrid_rrf_k=60,
            search_expr="",
            alias=f"{cls.collection_name}_alias",
        )

    @classmethod
    def tearDownClass(cls):
        if not hasattr(cls, "_utility") or not hasattr(cls, "_Collection"):
            return
        try:
            if cls._utility.has_collection(cls.collection_name):
                cls._Collection(cls.collection_name).drop()
        except Exception:
            pass

    def _assert_query_contains_keywords(self, query: str, keywords: list[str]) -> None:
        passages = self.retriever.search(query, top_k=5)
        self.assertGreaterEqual(len(passages), 1, msg=f"query={query} 未检索到任何结果")
        merged_text = "\n".join((p.text or "") for p in passages[:5])
        self.assertTrue(
            any(keyword in merged_text for keyword in keywords),
            msg=f"query={query} 的 Top5 未命中关键词: {keywords}",
        )

    def test_import_entity_count_matches_chunk_count(self):
        self.assertGreaterEqual(len(self.records), 5)
        self.assertEqual(self.collection.num_entities, len(self.records))

    def test_retrieval_probe_for_travel_policy(self):
        self._assert_query_contains_keywords(
            query="加班认定条件是什么",
            keywords=["加班", "正常工作时间外", "突发事件"],
        )

    def test_retrieval_probe_for_approval_policy(self):
        self._assert_query_contains_keywords(
            query="旷工如何认定",
            keywords=["旷工", "未办理请假手续", "虚假请假"],
        )


if __name__ == "__main__":
    unittest.main()
