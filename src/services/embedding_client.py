from __future__ import annotations

import logging
import math
import time
from typing import List, Sequence

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from src.config.config_settings import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    MILVUS_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

_client: OpenAI | None = None
_MAX_ATTEMPTS = 3


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
        )
    return _client


def _normalize(vector: Sequence[float]) -> List[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]


def _embed_batch(texts: Sequence[str]) -> List[List[float]]:
    client = _get_client()
    last_error: Exception | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = client.embeddings.create(
                model=MILVUS_EMBEDDING_MODEL,
                input=list(texts),
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            return [_normalize(item.embedding) for item in ordered]
        except (APIConnectionError, APITimeoutError) as exc:
            last_error = exc
        except APIStatusError as exc:
            last_error = exc
            if exc.status_code not in {429, 500, 502, 503, 504}:
                raise

        if attempt < _MAX_ATTEMPTS:
            time.sleep(0.5 * attempt)

    if last_error is not None:
        raise last_error
    raise RuntimeError("embedding request failed without a captured exception")


def embed_query(text: str) -> List[float]:
    if not text.strip():
        raise ValueError("text must not be empty")
    return embed_documents([text])[0]


def embed_documents(texts: Sequence[str], batch_size: int = 10) -> List[List[float]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    items = list(texts)
    if not items:
        return []

    vectors: List[List[float]] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        vectors.extend(_embed_batch(batch))
    return vectors
