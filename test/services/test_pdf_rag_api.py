import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.main import app


class PdfRagApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_pdf_rag_endpoint_success(self):
        fake_result = {
            "query": "迟到补卡怎么处理",
            "answer": "迟到后需要在系统提交补卡并附原因。",
            "passages": [
                {
                    "text": "迟到员工应在当日完成补卡申请。",
                    "score": 0.91,
                    "source": "kaoqin.pdf / 第一条 / 1 / 1",
                }
            ],
        }
        with patch("main.query_pdf_rag", new=AsyncMock(return_value=fake_result)) as mocked_query:
            response = self.client.post(
                "/api/pdf-rag",
                json={"query": "迟到补卡怎么处理", "model": "qwen-plus", "top_k": 3},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["code"], 200)
        self.assertEqual(payload["data"]["answer"], fake_result["answer"])
        mocked_query.assert_awaited_once()

    def test_pdf_rag_endpoint_requires_query(self):
        response = self.client.post("/api/pdf-rag", json={"model": "qwen-plus"})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
