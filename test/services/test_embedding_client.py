import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openai import APIConnectionError

from src.services import embedding_client
from src.services.embedding_client import embed_documents, embed_query


def _response_with_embeddings(embeddings):
    return SimpleNamespace(
        data=[
            SimpleNamespace(index=index, embedding=embedding)
            for index, embedding in enumerate(embeddings)
        ]
    )


class EmbeddingClientTests(unittest.TestCase):
    def test_embed_documents_preserves_order_and_normalizes(self):
        fake_client = Mock()
        fake_client.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[3.0, 4.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            ]
        )

        with patch.object(embedding_client, "_client", fake_client):
            vectors = embed_documents(["first", "second"], batch_size=2)

        self.assertEqual(len(vectors), 2)
        self.assertEqual(vectors[0], [1.0, 0.0])
        self.assertAlmostEqual(vectors[1][0], 0.6)
        self.assertAlmostEqual(vectors[1][1], 0.8)

    def test_embed_documents_batches_calls(self):
        fake_client = Mock()
        fake_client.embeddings.create.side_effect = [
            _response_with_embeddings([[1.0, 0.0], [0.0, 1.0]]),
            _response_with_embeddings([[1.0, 0.0]]),
        ]

        with patch.object(embedding_client, "_client", fake_client):
            vectors = embed_documents(["a", "b", "c"], batch_size=2)

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(fake_client.embeddings.create.call_count, 2)
        self.assertEqual(fake_client.embeddings.create.call_args_list[0].kwargs["input"], ["a", "b"])
        self.assertEqual(fake_client.embeddings.create.call_args_list[1].kwargs["input"], ["c"])

    def test_embed_documents_retries_transient_connection_error(self):
        fake_client = Mock()
        fake_client.embeddings.create.side_effect = [
            APIConnectionError(request=Mock()),
            _response_with_embeddings([[1.0]]),
        ]

        with patch.object(embedding_client, "_client", fake_client):
            with patch("src.services.embedding_client.time.sleep") as sleep:
                vectors = embed_documents(["a"])

        self.assertEqual(vectors, [[1.0]])
        self.assertEqual(fake_client.embeddings.create.call_count, 2)
        sleep.assert_called_once()

    def test_embed_query_rejects_blank_text(self):
        with self.assertRaises(ValueError):
            embed_query("   ")


if __name__ == "__main__":
    unittest.main()
