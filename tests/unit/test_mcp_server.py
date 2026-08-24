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
    (pack_dir / "youtube" / "example-abc123-part-001.md").write_text(
        "---\n"
        "title: Example\n"
        "type: reference\n"
        "pack: alex-hormozi-brain\n"
        "id: alex-hormozi-brain/youtube/abc123/part-001\n"
        "---\n"
        "# Example\n\n"
        "- Video ID: `abc123`\n"
        "- YouTube URL: https://youtube.com/watch?v=abc123\n"
        "- Timestamp range: 12s-20s\n\n"
        "Transcript evidence.\n",
        encoding="utf-8",
    )
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
            "coverage_categories": {
                "categories": {
                    "included": 1,
                    "duplicate": 0,
                    "incomplete": 2,
                    "unsupported": 0,
                    "quarantined": 0,
                    "missing": 0,
                },
                "missing": {
                    "inventory_records": [],
                    "official_captionless_videos": [{
                        "video_id": "captionless",
                        "title": "Captionless",
                        "status": "caption_unavailable_pending_openai_transcription",
                        "channel_url": "https://youtube.com/@AlexHormozi",
                    }],
                    "audio_pending_transcription": [{
                        "path": r"C:\private\audio.mp3",
                        "status": "metadata_ready_pending_transcription",
                    }],
                },
            },
            "restricted": {
                "status": "pending_external_authorization",
                "unique_work_count": 0,
                "ocr_requested": False,
                "ocr_atoms": 0,
                "authorization_required": True,
                "reason": "authorization is required",
                "sources": [
                    {"source_id": "qsrc-b"},
                    {"source_id": "qsrc-a"},
                ],
            },
            "transcripts": {
                "unique_videos": 1,
                "source_files": [r"C:\\private\\transcripts.txt"],
            },
            "skills": {"status": "ready", "packages": ["test-skill"], "invalid": []},
            "extras": {
                "ocr": {
                    "status": "indexed_ocr_recovered_manual_visual_qa_complete",
                    "requested_pages": 442,
                    "recovered_pages": 442,
                    "visual_qa_status": "manual_review_complete",
                    "manual_review_pages": 340,
                    "manual_review_decision_counts": {"accept_ocr": 285, "graphic_or_blank": 55},
                },
                "audio": [{"path": r"C:\\private\\audio.mp3", "status": "metadata_ready_pending_transcription"}],
                "restricted": {
                    "status": "pending_external_authorization",
                    "unique_work_count": 0,
                    "ocr_requested": False,
                    "ocr_atoms": 0,
                    "authorization_required": True,
                    "reason": "authorization is required",
                    "sources": [
                        {"source_id": "qsrc-b"},
                        {"source_id": "qsrc-a"},
                    ],
                },
                "containers": {
                    "status": "complete",
                    "source_count": 3,
                    "unique_knowledge_ingested": False,
                    "sources": [{"source_id": "src-1", "path": r"C:\\private\\book.zip", "status": "inspected"}],
                },
            },
            "records": [{
                "record_id": "derived-audio-audio",
                "kind": "derived_audio",
                "title": "audio",
                "status": "metadata_ready_pending_transcription",
            }],
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
        assert "book_to_skills" in coverage
        full_coverage = _payload(await client.call_tool("get_brain_coverage", {"include_records": True}))
        assert full_coverage["source_aliases"] == []
        assert coverage["ocr"]["requested_pages"] == 442
        assert coverage["ocr"]["recovered_pages"] == 442
        assert coverage["pending"]["ocr_pages"] == 0
        assert coverage["pending"]["ocr_manual_review_pages"] == 0
        assert coverage["coverage_categories"]["categories"]["included"] == 1
        assert coverage["coverage_categories"]["missing"]["official_captionless_videos"][0]["video_id"] == "captionless"
        assert coverage["coverage_categories"]["missing"]["audio_pending_transcription"] == [{"status": "metadata_ready_pending_transcription"}]
        assert coverage["restricted"]["status"] == "pending_external_authorization"
        assert coverage["restricted"]["authorization_required"] is True
        assert coverage["restricted"]["source_ids"] == ["qsrc-a", "qsrc-b"]
        assert coverage["pending"]["restricted_resolution_status"] == "pending_external_authorization"
        assert coverage["pending"]["audio"][0]["source_id"] == "derived-audio-audio"
        assert coverage["pending"]["audio"][0]["content_type"] == "audio"
        serialized_coverage = json.dumps(coverage)
        assert "C:\\private" not in serialized_coverage
        assert "source_files" not in serialized_coverage
        assert '"path"' not in serialized_coverage
        skill = _payload(await client.call_tool("get_hormozi_skill", {"skill_name": "test-skill"}))
        assert "# Test skill" in skill["content"]
        assert skill["source_id"] == "alex-hormozi-brain/agent-skills/test-skill"
        assert skill["content_type"] == "workflow"
        assert skill["file_provenance"] == "agent-skills/test-skill.md"
        assert "package_root" not in skill
        assert "C:\\" not in json.dumps(skill)
        search = _payload(await client.call_tool("search_hormozi_brain", {"query": "offers"}))
        assert search["results"] == []


@pytest.mark.asyncio
async def test_get_hormozi_source_returns_cited_locator(tiny_hormozi_pack):
    from mcp import Client

    engine = AsyncMock()
    mcp = create_pack_mcp(tiny_hormozi_pack.slug, tiny_hormozi_pack, engine)
    async with Client(mcp) as client:
        payload = _payload(await client.call_tool("get_hormozi_source", {
            "source_id": "alex-hormozi-brain/youtube/abc123/part-001",
        }))
    assert payload["source_id"] == "alex-hormozi-brain/youtube/abc123/part-001"
    assert payload["content_type"] == "reference"
    assert payload["file_provenance"] == "youtube/example-abc123-part-001.md"
    assert payload["locator"] == "timestamp 12s-20s"
    assert payload["citation_url"] == "https://youtube.com/watch?v=abc123&t=12s"


def test_source_enrichment_normalizes_title_only_page_locator():
    from ep_mcp.server import _enrich_hormozi_source

    payload = _enrich_hormozi_source(
        {
            "title": "Pricing Playbook — pages 4-4",
            "content": "Pricing guidance",
            "id": "alex-hormozi-brain/evidence/pricing-pages-4-4",
            "path": "evidence/pricing-pages-4-4.md",
        }
    )
    assert payload["page"] == 4
    assert payload["locator"] == "page 4"
    assert payload["citation"] == "evidence/pricing-pages-4-4.md page 4"


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
    assert row["source_id"] == row["id"]
    assert row["content_type"] == "reference"
    assert row["file_provenance"] == row["source_file"]
    assert row["title"] == "Example"
    assert row["confidence"] == "crawled"
    assert row["source_url"] == "https://youtube.com/watch?v=abc123"
    assert row["citation_url"] == "https://youtube.com/watch?v=abc123&t=12s"
    assert row["video_id"] == "abc123"
    assert row["timestamp"] == "12s-20s"
    assert row["locator"] == "timestamp 12s-20s"
    assert row["citation"] == "youtube/example-abc123-part-001.md timestamp 12s-20s"


@pytest.mark.asyncio
async def test_hormozi_search_and_source_use_fallback_id_for_unattributed_file(tiny_hormozi_pack):
    from mcp import Client

    engine = AsyncMock()
    engine.search.return_value = [
        SearchResult(
            text="Generated overview without frontmatter ID.",
            source_file="overview.md",
            id=None,
            content_hash="sha256:overview",
            score=0.7,
            type=None,
            title="Brain overview",
        ),
    ]
    mcp = create_pack_mcp(tiny_hormozi_pack.slug, tiny_hormozi_pack, engine)
    async with Client(mcp) as client:
        search = _payload(await client.call_tool("search_hormozi_brain", {"query": "overview"}))
        row = search["results"][0]
        assert row["source_id"] == "alex-hormozi-brain/file/overview.md"
        assert row["content_type"] == "untyped"
        assert row["confidence"] == "ungraded"
        source = _payload(await client.call_tool("get_hormozi_source", {"source_id": row["source_id"]}))
    assert source["source_id"] == row["source_id"]
    assert source["file_provenance"] == "overview.md"
    assert source["content_type"] == "untyped"
    assert source["confidence"] == "ungraded"


@pytest.mark.asyncio
async def test_hormozi_search_expands_path_scopes_before_truncating(tiny_hormozi_pack):
    from mcp import Client

    engine = AsyncMock()
    engine.search.return_value = [
        SearchResult(text="wrong scope", source_file="evidence/a.md", id="evidence/a", score=0.9),
        SearchResult(text="right scope", source_file="youtube/b.md", id="youtube/b", score=0.8),
    ]
    mcp = create_pack_mcp(tiny_hormozi_pack.slug, tiny_hormozi_pack, engine)
    async with Client(mcp) as client:
        payload = _payload(await client.call_tool("search_hormozi_brain", {
            "query": "offers",
            "source_scope": "youtube",
            "max_results": 1,
        }))
    assert [row["source_file"] for row in payload["results"]] == ["youtube/b.md"]
    request = engine.search.await_args.args[0]
    assert request.max_results == 4


@pytest.mark.asyncio
async def test_hormozi_search_normalizes_page_locator(tiny_hormozi_pack):
    from mcp import Client

    engine = AsyncMock()
    engine.search.return_value = [
        SearchResult(
            text="- Page: 17\nPricing guidance",
            source_file="ocr/src-example-page-0017.md",
            id="alex-hormozi-brain/ocr/src-example/page-0017",
            content_hash="sha256:page",
            score=0.82,
            type="reference",
            title="Pricing — OCR page 17",
            confidence="manually_transcribed",
        ),
    ]
    mcp = create_pack_mcp(tiny_hormozi_pack.slug, tiny_hormozi_pack, engine)
    async with Client(mcp) as client:
        payload = _payload(await client.call_tool("search_hormozi_brain", {"query": "pricing"}))
    row = payload["results"][0]
    assert row["page"] == 17
    assert row["locator"] == "page 17"
    assert row["citation"] == "ocr/src-example-page-0017.md page 17"
    assert row["confidence"] == "manually_transcribed"


@pytest.mark.asyncio
async def test_hormozi_search_normalizes_title_only_page_locator(tiny_hormozi_pack):
    from mcp import Client

    engine = AsyncMock()
    engine.search.return_value = [
        SearchResult(
            text="Pricing guidance without an inline page marker",
            source_file="evidence/pricing-pages-17-17.md",
            id="alex-hormozi-brain/evidence/pricing-pages-17-17",
            content_hash="sha256:page-title",
            score=0.82,
            type="concept",
            title="Pricing Playbook — pages 17-17",
            confidence="crawled",
        ),
    ]
    mcp = create_pack_mcp(tiny_hormozi_pack.slug, tiny_hormozi_pack, engine)
    async with Client(mcp) as client:
        payload = _payload(await client.call_tool("search_hormozi_brain", {"query": "pricing"}))
    row = payload["results"][0]
    assert row["page"] == 17
    assert row["locator"] == "page 17"
    assert row["citation"] == "evidence/pricing-pages-17-17.md page 17"


@pytest.mark.asyncio
async def test_hormozi_search_supports_ocr_scope(tiny_hormozi_pack):
    from mcp import Client

    engine = AsyncMock()
    engine.search.return_value = [
        SearchResult(text="OCR page", source_file="ocr/source-page-0001.md", id="ocr/source/page-0001", score=0.8),
        SearchResult(text="Book page", source_file="ebook/book.md", id="ebook/book", score=0.7),
    ]
    mcp = create_pack_mcp(tiny_hormozi_pack.slug, tiny_hormozi_pack, engine)
    async with Client(mcp) as client:
        payload = _payload(await client.call_tool("search_hormozi_brain", {
            "query": "pricing",
            "source_scope": "ocr",
            "max_results": 1,
        }))
    assert [row["source_file"] for row in payload["results"]] == ["ocr/source-page-0001.md"]
