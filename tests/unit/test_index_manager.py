from pathlib import Path

import pytest

from ep_mcp.embeddings.base import EmbeddingProvider
from ep_mcp.index.manager import IndexManager
from ep_mcp.index.sqlite_store import SQLiteStore
from ep_mcp.pack.loader import load_pack


class FakeProvider(EmbeddingProvider):
    def __init__(self, model_name: str):
        self._model_name = model_name
        self.calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return 4

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(index), 1.0, 0.0, 0.0] for index, _ in enumerate(texts)]


def make_pack(root: Path) -> Path:
    (root / "concepts").mkdir(parents=True)
    (root / "manifest.yaml").write_text(
        """
slug: index-test
name: Index Test
type: product
version: "1.0.0"
description: Index test pack
entry_point: overview.md
context:
  always:
    - overview.md
""",
        encoding="utf-8",
    )
    (root / "overview.md").write_text("---\nid: index-test/overview\n---\n# Overview\nStable content.\n", encoding="utf-8")
    (root / "concepts" / "topic.md").write_text("---\nid: index-test/topic\n---\n# Topic\nIncremental content.\n", encoding="utf-8")
    return root


@pytest.mark.asyncio
async def test_index_manager_incremental_and_model_rebuild(tmp_path):
    pack = load_pack(make_pack(tmp_path / "pack"))
    store = SQLiteStore(str(tmp_path / "index.db"), embedding_dimension=4)
    store.open()
    try:
        provider = FakeProvider("openai/test-model-a")
        manager = IndexManager(pack, store, provider)
        first = await manager.build_index()
        assert first.new_files == 2
        assert first.cache_misses > 0
        first_call_count = len(provider.calls)

        second = await manager.build_index()
        assert second.new_files == 0
        assert second.changed_files == 0
        assert second.unchanged_files == 2
        assert len(provider.calls) == first_call_count

        provider._model_name = "openai/test-model-b"
        rebuilt = await manager.build_index()
        assert rebuilt.full_rebuild is True
        assert rebuilt.new_files == 2
        assert len(provider.calls) > first_call_count
        assert store.get_meta("embedding_model") == "openai/test-model-b"
    finally:
        store.close()
