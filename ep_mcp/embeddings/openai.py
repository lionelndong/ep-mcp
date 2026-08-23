"""Direct OpenAI embedding provider.

The provider intentionally reads credentials from ``OPENAI_API_KEY`` (or an
explicit configuration value) and never logs source text.  It mirrors the
Azure provider's batching, retry, and query/document separation while using
the public OpenAI API directly.
"""

from __future__ import annotations

import asyncio
import logging
import os

from openai import AsyncOpenAI

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)

_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
_MAX_BATCH_SIZE = 512
_MAX_PARALLEL_BATCHES = 4


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings with bounded concurrency and exponential retry."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        max_parallel_batches: int = _MAX_PARALLEL_BATCHES,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._max_retries = max(1, max_retries)
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key required — set OPENAI_API_KEY or embedding.api_key"
            )
        kwargs: dict[str, str] = {"api_key": key}
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if resolved_base_url:
            kwargs["base_url"] = resolved_base_url
        self._client = AsyncOpenAI(**kwargs)
        self._semaphore = asyncio.Semaphore(max(1, max_parallel_batches))

    @property
    def model_name(self) -> str:
        return f"openai/{self._model}"

    @property
    def dimension(self) -> int:
        return self._dimensions or _MODEL_DIMENSIONS.get(self._model, 1536)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        batches = [
            texts[i : i + _MAX_BATCH_SIZE]
            for i in range(0, len(texts), _MAX_BATCH_SIZE)
        ]
        results = await asyncio.gather(
            *(self._embed_batch_with_semaphore(batch) for batch in batches)
        )
        return [vector for batch in results for vector in batch]

    async def embed_query(self, query: str) -> list[float]:
        result = await self._embed_batch_with_semaphore([query])
        return result[0]

    async def _embed_batch_with_semaphore(self, texts: list[str]) -> list[list[float]]:
        async with self._semaphore:
            return await self._embed_batch(texts)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(self._max_retries):
            try:
                kwargs: dict[str, object] = {"model": self._model, "input": texts}
                if self._dimensions is not None:
                    kwargs["dimensions"] = self._dimensions
                response = await self._client.embeddings.create(**kwargs)
                ordered = sorted(response.data, key=lambda item: item.index)
                vectors = [list(item.embedding) for item in ordered]
                if len(vectors) != len(texts):
                    raise RuntimeError(
                        f"OpenAI returned {len(vectors)} vectors for {len(texts)} inputs"
                    )
                if any(len(vector) != self.dimension for vector in vectors):
                    raise RuntimeError(
                        f"OpenAI returned an unexpected embedding dimension for {self.model_name}"
                    )
                return vectors
            except Exception as error:  # SDK raises several provider-specific types.
                if attempt + 1 >= self._max_retries:
                    raise RuntimeError(
                        f"OpenAI embedding failed after {self._max_retries} attempts: {error}"
                    ) from error
                wait = 2**attempt
                logger.warning(
                    "OpenAI embedding attempt %d failed (%s); retrying in %ds",
                    attempt + 1,
                    type(error).__name__,
                    wait,
                )
                await asyncio.sleep(wait)
        raise AssertionError("unreachable")
