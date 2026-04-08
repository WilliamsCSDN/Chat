import unittest

from services.chat_service import (
    _extract_answer_from_passage_text,
    build_direct_answer_content,
    should_use_direct_answer,
)
from services.milvus_retriever import RetrievedPassage


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


if __name__ == "__main__":
    unittest.main()
