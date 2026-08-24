import unittest
from unittest.mock import patch

from src.services.milvus_retriever import MilvusRetriever


class MilvusRetrieverEmbeddingTests(unittest.TestCase):
    def test_embedding_failure_skips_query_without_disabling_retriever(self):
        retriever = MilvusRetriever(
            host="localhost",
            port=19530,
            collection_name="test",
            vector_field="embedding",
            text_field="text",
            top_k=3,
            nprobe=32,
            embedding_model_name="text-embedding-v4",
            embedding_model_path="unused",
            source_fields=[],
            hybrid_enabled=False,
            hybrid_sparse_recall_k=10,
            hybrid_rrf_k=60,
        )

        with patch.object(
            retriever,
            "_embed_query",
            side_effect=RuntimeError("dashscope unavailable"),
        ):
            passages = retriever.search("测试查询")

        self.assertEqual(passages, [])
        self.assertFalse(retriever._disabled)


if __name__ == "__main__":
    unittest.main()
