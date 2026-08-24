"""Unit tests for the direct OpenAI embedding provider."""

from types import SimpleNamespace

import pytest

from ep_mcp.embeddings.openai import OpenAIEmbeddingProvider


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        values = [
            SimpleNamespace(index=index, embedding=[float(index), 1.0])
            for index, _ in enumerate(kwargs["input"])
        ]
        return SimpleNamespace(data=values)


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddings()


@pytest.mark.asyncio
async def test_openai_provider_batches_and_preserves_order(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIEmbeddingProvider(model="test-model", dimensions=2, max_parallel_batches=1)
    fake = FakeClient()
    provider._client = fake

    result = await provider.embed(["one", "two"])

    assert result == [[0.0, 1.0], [1.0, 1.0]]
    assert fake.embeddings.calls[0]["model"] == "test-model"
    assert fake.embeddings.calls[0]["input"] == ["one", "two"]


@pytest.mark.asyncio
async def test_openai_provider_query_is_single_request(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIEmbeddingProvider(model="text-embedding-3-small", dimensions=2)
    fake = FakeClient()
    provider._client = fake

    result = await provider.embed_query("pricing")

    assert result == [0.0, 1.0]
    assert len(fake.embeddings.calls) == 1


@pytest.mark.asyncio
async def test_openai_provider_retries_and_rejects_wrong_dimension(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIEmbeddingProvider(model="text-embedding-3-small", dimensions=2, max_retries=2)

    class WrongDimension:
        def __init__(self):
            self.calls = 0

        async def create(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0])])

    wrong = WrongDimension()
    provider._client = SimpleNamespace(embeddings=wrong)
    monkeypatch.setattr("ep_mcp.embeddings.openai.asyncio.sleep", lambda _seconds: _completed_sleep())
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        await provider.embed(["one"])
    assert wrong.calls == 2


def test_openai_provider_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider()


def test_openai_provider_requires_explicit_dimension_for_unknown_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="Explicit dimensions"):
        OpenAIEmbeddingProvider(model="future-embedding-model")


def test_openai_provider_rejects_nonpositive_dimension(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="dimensions must be positive"):
        OpenAIEmbeddingProvider(model="text-embedding-3-small", dimensions=0)


async def _completed_sleep():
    return None
