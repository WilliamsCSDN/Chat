import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from src.services import RetrievedPassage
from src.services import query_pdf_rag, rerank_pdf_passages


class PdfRagServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_rerank_pdf_passages_filters_low_signal_text(self):
        passages = [
            RetrievedPassage(
                text="保密级别：内部资料",
                score=0.95,
                source="kaoqin.pdf / 1 / 1",
                vector_score=0.95,
            ),
            RetrievedPassage(
                text="员工探亲往返前，需按照公司出差/请假管理规定履行审批流程。",
                score=0.62,
                source="kaoqin.pdf / 第十四条 探亲审批 / 5 / 1",
                vector_score=0.62,
            ),
        ]

        ranked = rerank_pdf_passages("探亲审批流程", passages, top_k=1)
        self.assertEqual(len(ranked), 1)
        self.assertIn("审批流程", ranked[0].text)

    async def test_query_pdf_rag_expands_recall_before_final_top_k(self):
        fake_retriever = Mock()
        fake_retriever.search.return_value = [
            RetrievedPassage(
                text="员工探亲往返前，需按照公司出差/请假管理规定履行审批流程。",
                score=0.75,
                source="kaoqin.pdf / 第十四条 探亲审批 / 5 / 1",
                vector_score=0.75,
            )
        ]
        fake_completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="需要先审批后执行。"))]
        )

        with patch("services.pdf_rag_service.get_pdf_retriever", return_value=fake_retriever):
            with patch(
                "services.pdf_rag_service.client.chat.completions.create",
                new=AsyncMock(return_value=fake_completion),
            ):
                result = await query_pdf_rag("探亲审批流程是什么", top_k=2)

        fake_retriever.search.assert_called_once()
        called_query, called_top_k = fake_retriever.search.call_args.args
        self.assertEqual(called_query, "探亲审批流程是什么")
        self.assertGreaterEqual(called_top_k, 8)
        self.assertEqual(len(result["passages"]), 1)


if __name__ == "__main__":
    unittest.main()
