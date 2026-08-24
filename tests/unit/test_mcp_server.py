"""In-memory MCP SDK v2 smoke tests (legacy + 2026-07-28)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ep_mcp.pack.loader import load_pack
from ep_mcp.retrieval.models import SearchResult
from ep_mcp.server import create_pack_mcp


def _payload(result) -> dict:
    if result.structured_content:
        return result.structured_content
    text = result.content[0].text
    return json.loads(text)


@pytest.fixture
def tiny_pack(tmp_path: Path):
    (tmp_path / "manifest.yaml").write_text(
        """
slug: tiny
name: Tiny Pack
type: product
version: "1.2.0"
description: Tiny fixture pack
entry_point: overview.md
authority_boundary:
  in_scope: Tiny domain
  out_of_scope:
    - Unrelated topics
context:
  always:
    - overview.md
""",
        encoding="utf-8",
    )
    (tmp_path / "overview.md").write_text(
        """---
id: tiny/overview
type: concept
---
# Tiny
Hello.
""",
        encoding="utf-8",
    )
    return load_pack(tmp_path)


def _make_server(pack):
    engine = AsyncMock()
    return create_pack_mcp(pack.slug, pack, engine)


@pytest.mark.asyncio
async def test_list_tools_modern_and_legacy(tiny_pack):
    from mcp import Client

    mcp = _make_server(tiny_pack)
    expected = {"ep_search_tool", "ep_list_topics_tool", "ep_graph_traverse_tool", "ep_read_tool"}

    async with Client(mcp) as modern:
        modern_tools = {t.name for t in (await modern.list_tools()).tools}
        assert expected <= modern_tools
        result = await modern.call_tool("ep_list_topics_tool", {})
        payload = _payload(result)
        assert payload.get("pack", {}).get("slug") == "tiny"
        assert payload["pack"]["authority_boundary"]["in_scope"] == "Tiny domain"

    async with Client(mcp, mode="legacy") as legacy:
        legacy_tools = {t.name for t in (await legacy.list_tools()).tools}
        assert expected <= legacy_tools
        result = await legacy.call_tool("ep_read_tool", {"path": "overview.md"})
        payload = _payload(result)
        assert "Hello" in payload.get("content", "")


@pytest.fixture
def tiny_hormozi_pack(tmp_path: Path):
    pack_dir = tmp_path / "packs" / "brain"
    (pack_dir / "agent-skills").mkdir(parents=True)
    (pack_dir / "meta").mkdir(parents=True)
    (pack_dir / "youtube").mkdir(parents=True)
    (tmp_path / "skills" / "alex-hormozi" / "test-skill").mkdir(parents=True)
    (tmp_path / "skills" / "alex-hormozi" / "test-skill" / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: test\n---\n# Test skill\n",
        encoding="utf-8",
    )
    (pack_dir / "manifest.yaml").write_text(
        """
slug: alex-hormozi-brain
name: Hormozi Brain
type: person
version: "1.0.0"
description: Brain fixture
entry_point: overview.md
context:
  always:
    - overview.md
""",
        encoding="utf-8",
    )
    (pack_dir / "overview.md").write_text("# Brain\n", encoding="utf-8")
    (pack_dir / "agent-skills" / "test-skill.md").write_text(
        "---\n"
        "title: test-skill\n"
        "type: workflow\n"
        "pack: alex-hormozi-brain\n"
        "id: alex-hormozi-brain/agent-skills/test-skill\n"
        "---\n# Test skill\n",
        encoding="utf-8",
    )
    (pack_dir / "meta" / "brain-coverage.json").write_text(
        json.dumps({
            "inventory_records": 1,
            "derived_records": 1,
            "summary": {"indexed_evidence": 1},
            "transcripts": {"unique_videos": 1},
            "skills": {"status": "ready", "packages": ["test-skill"], "invalid": []},
            "extras": {
                "ocr": {"pages": 0},
                "audio": [],
                "containers": {"status": "complete", "source_count": 3, "unique_knowledge_ingested": False},
            },
            "records": [],
        }),
        encoding="utf-8",
    )
    return load_pack(pack_dir)


@pytest.mark.asyncio
async def test_hormozi_brain_tools(tiny_hormozi_pack):
    from mcp import Client

    engine = AsyncMock()
    engine.search.return_value = []
    mcp = create_pack_mcp(tiny_hormozi_pack.slug, tiny_hormozi_pack, engine)
    async with Client(mcp) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
        assert {"search_hormozi_brain", "get_hormozi_source", "get_hormozi_skill", "get_brain_coverage"} <= names
        coverage = _payload(await client.call_tool("get_brain_coverage", {}))
        assert coverage["inventory_records"] == 1
        assert coverage["containers"]["source_count"] == 3
        skill = _payload(await client.call_tool("get_hormozi_skill", {"skill_name": "test-skill"}))
        assert "# Test skill" in skill["content"]
        search = _payload(await client.call_tool("search_hormozi_brain", {"query": "offers"}))
        assert search["results"] == []


@pytest.mark.asyncio
async def test_hormozi_search_returns_cited_provenance_fields(tiny_hormozi_pack):
    from mcp import Client

    engine = AsyncMock()
    engine.search.return_value = [
        SearchResult(
            text=(
                "Evidence boundary: transcript-derived source material.\n"
                "- Video ID: `abc123`\n"
                "- YouTube URL: https://youtube.com/watch?v=abc123\n"
                "- Timestamp range: 12s-20s\n"
            ),
            source_file="youtube/example-abc123-part-001.md",
            id="alex-hormozi-brain/youtube/abc123/part-001",
            content_hash="sha256:test",
            verified_at="2026-08-23",
            score=0.91,
            type="reference",
            title="Example",
            confidence="crawled",
            line_range=(10, 18),
        ),
    ]
    mcp = create_pack_mcp(tiny_hormozi_pack.slug, tiny_hormozi_pack, engine)
    async with Client(mcp) as client:
        payload = _payload(await client.call_tool("search_hormozi_brain", {"query": "offers"}))
    row = payload["results"][0]
    assert row["id"] == "alex-hormozi-brain/youtube/abc123/part-001"
    assert row["title"] == "Example"
    assert row["confidence"] == "crawled"
    assert row["source_url"] == "https://youtube.com/watch?v=abc123"
    assert row["video_id"] == "abc123"
    assert row["locator"] == "12s-20s"
    assert row["citation"] == "youtube/example-abc123-part-001.md lines 10-18"
