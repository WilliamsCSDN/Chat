from functools import lru_cache
import hashlib
from pathlib import Path
import re
from typing import Iterable, List

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
FALLBACK_DIM = 384


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


class HashingEmbeddingModel:
    """
    轻量回退向量器（仅在 sentence-transformers 缺失时启用）：
    - 字符 2/3-gram hashing
    - 输出维度与主模型保持一致（384）
    """

    def __init__(self, dim: int = FALLBACK_DIM):
        self.dim = dim

    @staticmethod
    def _ngrams(text: str) -> Iterable[str]:
        normalized = re.sub(r"\s+", "", text)
        if not normalized:
            return []

        grams: List[str] = [normalized]
        for n in (2, 3):
            if len(normalized) >= n:
                grams.extend(normalized[i : i + n] for i in range(len(normalized) - n + 1))
        return grams

    def _encode_one(self, text: str, normalize_embeddings: bool) -> EncodedVector:
        vec = [0.0] * self.dim
        grams = self._ngrams(text)

        for gram in grams:
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vec[idx] += sign

        if normalize_embeddings:
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]

        return EncodedVector(vec)

    def encode(self, texts, normalize_embeddings: bool = True, show_progress_bar: bool = False):
        _ = show_progress_bar
        return EncodedMatrix([self._encode_one(t, normalize_embeddings) for t in texts])


@lru_cache(maxsize=1)
def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print(f"⚠️  sentence-transformers 不可用（{e}），回退到轻量哈希向量器")
        return HashingEmbeddingModel(dim=FALLBACK_DIM)

    # 优先加载项目内模型目录，便于离线运行。
    if LOCAL_MODEL_DIR.exists():
        try:
            print(f"📁 使用本地模型: {LOCAL_MODEL_DIR}")
            return SentenceTransformer(str(LOCAL_MODEL_DIR))
        except Exception as e:
            print(f"⚠️  本地模型加载失败（{e}），回退在线模型")

    print(f"🌐 使用在线模型: {MODEL_ID}")
    return SentenceTransformer(MODEL_ID)