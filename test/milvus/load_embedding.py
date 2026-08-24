from functools import lru_cache
from pathlib import Path
import sys
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.embedding_client import embed_documents


MODEL_ID = "text-embedding-v4"
FALLBACK_DIM = 1024


class EncodedVector(list):
    def tolist(self):
        return list(self)


class EncodedMatrix(list):
    def __getitem__(self, index):
        value = super().__getitem__(index)
        if isinstance(index, int):
            return EncodedVector(value)
        return EncodedMatrix(value)

    def tolist(self):
        return [list(row) for row in self]


class DashScopeEmbeddingModel:
    def encode(
        self,
        texts: str | Iterable[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> EncodedMatrix:
        _ = normalize_embeddings, show_progress_bar
        items = [texts] if isinstance(texts, str) else list(texts)
        return EncodedMatrix(embed_documents(items))


@lru_cache(maxsize=1)
def load_embedding_model() -> DashScopeEmbeddingModel:
    print(f"使用阿里云 DashScope embedding 模型: {MODEL_ID}")
    return DashScopeEmbeddingModel()
