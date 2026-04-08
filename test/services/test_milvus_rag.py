import asyncio
import unittest

from services.milvus_retriever import (
    ConfidencePolicy,
    RetrievedPassage,
    build_retrieval_system_message,
    extract_latest_user_query,
    filter_passages_by_confidence,
    fuse_ranked_passages,
    inject_retrieval_context,
    lexical_overlap_score,
    rerank_passages,
)


class FakeRetriever:
    def __init__(self, passages):
        self.passages = passages
        self.called_with = None

    def search(self, query: str, top_k: int = None):
        self.called_with = query
        return self.passages


class MilvusRagTests(unittest.TestCase):
    def test_fuse_ranked_passages_merges_dense_and_sparse(self):
        dense = [
            RetrievedPassage(text="A", score=0.8, source="s1", question="q1", vector_score=0.8),
            RetrievedPassage(text="B", score=0.7, source="s2", question="q2", vector_score=0.7),
        ]
        sparse = [
            RetrievedPassage(text="B", score=8.0, source="s2", question="q2"),
            RetrievedPassage(text="C", score=7.0, source="s3", question="q3"),
        ]
        fused = fuse_ranked_passages(dense, sparse, top_k=5, rrf_k=60)
        self.assertEqual(len(fused), 3)
        questions = [item.question for item in fused]
        self.assertIn("q1", questions)
        self.assertIn("q2", questions)
        self.assertIn("q3", questions)
        for item in fused:
            self.assertGreater(item.vector_score, 0.0)

    def test_filter_passages_by_confidence_rejects_low_confidence(self):
        passages = [
            RetrievedPassage(text="A", score=0.46, source="s1", question="q1", lexical_score=0.2),
            RetrievedPassage(text="B", score=0.45, source="s2", question="q2", lexical_score=0.2),
        ]
        policy = ConfidencePolicy(min_top1_score=0.5, min_top1_margin=0.03, min_top1_lexical=0.15)
        filtered = filter_passages_by_confidence(passages, policy)
        self.assertEqual(filtered, [])

    def test_filter_passages_by_confidence_keeps_only_above_score_floor(self):
        passages = [
            RetrievedPassage(text="A", score=0.78, source="s1", question="q1", lexical_score=1.0),
            RetrievedPassage(text="B", score=0.52, source="s2", question="q2", lexical_score=0.2),
            RetrievedPassage(text="C", score=0.31, source="s3", question="q3", lexical_score=0.1),
        ]
        policy = ConfidencePolicy(min_top1_score=0.6, min_top1_margin=0.03, min_top1_lexical=0.15)
        filtered = filter_passages_by_confidence(passages, policy)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].question, "q1")

    def test_lexical_overlap_score_exact_match_highest(self):
        score_exact = lexical_overlap_score("查看订单不成功怎么办", "查看订单不成功怎么办")
        score_partial = lexical_overlap_score("查看订单不成功怎么办", "为什么不能查看订单")
        self.assertGreater(score_exact, score_partial)
        self.assertGreaterEqual(score_exact, 0.99)

    def test_rerank_passages_prioritizes_exact_and_deduplicates_question(self):
        query = "查看订单不成功怎么办？"
        passages = [
            RetrievedPassage(
                text="答案A",
                score=0.66,
                source="支付与订单 / 充值 / 充值记录 / 查看订单不成功怎么办？",
                question="查看订单不成功怎么办？",
                vector_score=0.66,
            ),
            RetrievedPassage(
                text="答案B",
                score=0.68,
                source="故障与反馈 / 性能问题 / 卡顿延迟 / 查看订单不成功怎么办？",
                question="查看订单不成功怎么办？",
                vector_score=0.68,
            ),
            RetrievedPassage(
                text="答案C",
                score=0.69,
                source="支付与订单 / 订单 / 订单查询 / 为什么不能查看订单？",
                question="为什么不能查看订单？",
                vector_score=0.69,
            ),
        ]

        reranked = rerank_passages(query, passages, top_k=3)
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0].question, "查看订单不成功怎么办？")
        self.assertNotEqual(reranked[0].source, reranked[1].source)

    def test_extract_latest_user_query(self):
        messages = [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答"},
            {"role": "user", "content": "第二问"},
        ]
        self.assertEqual(extract_latest_user_query(messages), "第二问")

    def test_build_retrieval_system_message_contains_context(self):
        passages = [
            RetrievedPassage(text="孙悟空来自花果山。", score=0.92, source="第一回"),
            RetrievedPassage(text="他拜师菩提老祖。", score=0.89, source="第二回"),
        ]
        message = build_retrieval_system_message(passages)
        self.assertEqual(message["role"], "system")
        self.assertIn("检索参考资料", message["content"])
        self.assertIn("第一回", message["content"])
        self.assertIn("孙悟空来自花果山", message["content"])

    def test_inject_retrieval_context_adds_system_message(self):
        messages = [{"role": "user", "content": "孙悟空在哪里出生"}]
        retriever = FakeRetriever(
            [RetrievedPassage(text="石猴出世于花果山。", score=0.88, source="第一回")]
        )

        updated_messages, passages = asyncio.run(
            inject_retrieval_context(messages, retriever=retriever)
        )

        self.assertEqual(retriever.called_with, "孙悟空在哪里出生")
        self.assertEqual(len(passages), 1)
        self.assertEqual(updated_messages[0]["role"], "system")
        self.assertIn("石猴出世于花果山", updated_messages[0]["content"])

    def test_inject_retrieval_context_skips_when_no_user_message(self):
        messages = [{"role": "system", "content": "你是助手"}]
        retriever = FakeRetriever([RetrievedPassage(text="x", score=0.9, source="src")])

        updated_messages, passages = asyncio.run(
            inject_retrieval_context(messages, retriever=retriever)
        )

        self.assertIsNone(retriever.called_with)
        self.assertEqual(passages, [])
        self.assertEqual(updated_messages, messages)


if __name__ == "__main__":
    unittest.main()
