"""MCPServer setup, tool/resource registration, multi-pack routing."""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from .auth import APIKeyAuth
from .config import ServerConfig
from .embeddings.base import EmbeddingProvider
from .embeddings.gemini import GeminiEmbeddingProvider
from .index.manager import IndexManager
from .index.sqlite_store import SQLiteStore
from .pack.loader import load_pack
from .pack.models import Pack
from .prompts.pack_prompts import register_prompts
from .resources.pack_resources import register_resources
from .retrieval.engine import RetrievalEngine
from .retrieval.graph_helpers import GraphLookup
from .retrieval.reranker import Reranker
from .tools.ep_graph_traverse import ep_graph_traverse
from .tools.ep_list_topics import ep_list_topics
from .tools.ep_read import ep_read
from .tools.ep_search import ep_search, log_query

logger = logging.getLogger(__name__)


class _TokenBucketRateLimiter:
    """Small process-local token bucket used as a deployment safety valve."""

    def __init__(self, requests_per_minute: int, burst: int):
        self.rate = max(1, requests_per_minute) / 60.0
        self.capacity = max(1, burst)
        self._buckets: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, [float(self.capacity), now])
            tokens = min(float(self.capacity), tokens + (now - last) * self.rate)
            if tokens < 1.0:
                retry_after = max(1, int((1.0 - tokens) / self.rate + 0.999))
                self._buckets[key] = [tokens, now]
                return False, retry_after
            self._buckets[key] = [tokens - 1.0, now]
            return True, 0


def _loopback_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


class PackInstance:
    """A fully initialized pack with all its components."""

    def __init__(self, pack, store, engine, mcp, index_manager=None):
        self.pack = pack
        self.store = store
        self.engine = engine
        self.mcp = mcp
        self.index_manager = index_manager  # retained for file watcher reindex


def create_embedding_provider(config: ServerConfig) -> EmbeddingProvider:
    """Create the configured embedding provider."""
    from .embeddings.azure_openai import AzureOpenAIEmbeddingProvider
    from .embeddings.openai import OpenAIEmbeddingProvider

    emb = config.embedding
    if emb.provider == "gemini":
        return GeminiEmbeddingProvider(
            model=emb.model,
            output_dimensionality=emb.output_dimensionality,
        )
    if emb.provider == "azure-openai":
        return AzureOpenAIEmbeddingProvider(
            model=emb.model or "text-embedding-3-large",
            azure_endpoint=emb.azure_endpoint,
            api_key=emb.azure_api_key,
            api_version=emb.azure_api_version,
            azure_deployment=emb.azure_deployment,
            output_dimensionality=emb.output_dimensionality,
        )
    if emb.provider == "openai":
        return OpenAIEmbeddingProvider(
            model=emb.model or "text-embedding-3-small",
            api_key=emb.api_key,
            base_url=emb.base_url,
            dimensions=emb.output_dimensionality,
        )
    raise ValueError(
        f"Unsupported embedding provider: {emb.provider!r}. "
        f"Supported: 'gemini', 'openai', 'azure-openai'"
    )


def _get_server_instructions(pack: Pack) -> str:
    """Derive the MCP server instructions string for a pack.

    Priority: mcp.instructions in manifest > manifest.description > generic fallback.
    """
    parts: list[str] = []
    if pack.mcp_config.instructions:
        parts.append(pack.mcp_config.instructions)
    elif pack.description:
        parts.append(pack.description)
    else:
        parts.append(f"{pack.name} ExpertPack knowledge service.")

    parts.append(
        "Consume loop: ep_search for candidate atom ids, then ep_read to load the "
        "whole atom. requires: dependencies expand automatically on search. "
        "Stop when the atom answers the question, or after 3 steps (hard cap 7). "
        "Escalate only for multi-hop, contradiction, or a navigation miss. "
        "Use ep_list_topics to browse structure and ep_graph_traverse for graph hops."
    )
    boundary = pack.manifest.authority_boundary
    if boundary is not None:
        refuse = "; ".join(boundary.out_of_scope[:4]) if boundary.out_of_scope else "topics outside in_scope"
        parts.append(
            f"Authority: in scope — {boundary.in_scope} "
            f"Refuse: {refuse}."
        )
        if boundary.no_source_no_claim:
            parts.append("Do not assert a claim unless a retrieved atom supports it.")
    return " ".join(parts)


def _fallback_hormozi_source_id(pack_slug: str, path: str | None) -> str | None:
    """Return a stable ID for generated files that lack frontmatter IDs."""

    normalized = str(path or "").replace("\\", "/").lstrip("/")
    if not normalized or ".." in Path(normalized).parts:
        return None
    return f"{pack_slug}/file/{normalized}"


def _enrich_hormozi_source(payload: dict, pack_slug: str = "alex-hormozi-brain") -> dict:
    """Add stable citation fields when a complete Hormozi source is read."""

    if payload.get("error"):
        return payload
    source_text = str(payload.get("content", ""))
    source_url_match = re.search(r"YouTube URL:\s*(https?://\S+)", source_text)
    video_match = re.search(r"Video ID:\s*`([^`]+)`", source_text)
    timestamp_match = re.search(r"Timestamp range:\s*([^\n]+)", source_text)
    page_match = re.search(
        r"(?:\b(?:Source )?Page:\s*|\bpages?\s+)([0-9]+)",
        f"{payload.get('title', '')} {source_text}",
        re.IGNORECASE,
    )
    chapter_match = re.search(
        r"(?:EPUB )?chapter\s+([0-9]+)",
        f"{payload.get('title', '')} {source_text}",
        re.IGNORECASE,
    )
    if source_url_match:
        payload["source_url"] = source_url_match.group(1).rstrip("`),")
    if video_match:
        payload["video_id"] = video_match.group(1)
    if timestamp_match:
        timestamp = timestamp_match.group(1).strip()
        payload["timestamp"] = timestamp
        payload["locator"] = f"timestamp {timestamp}"
        if payload.get("source_url"):
            start = timestamp.split("-", 1)[0].strip()
            seconds: int | None = None
            seconds_match = re.fullmatch(r"(\d+(?:\.\d+)?)s", start)
            clock_match = re.fullmatch(r"(?:(\d+):)?(\d{1,2}):(\d{2})", start)
            if seconds_match:
                seconds = max(0, int(float(seconds_match.group(1))))
            elif clock_match:
                hours = int(clock_match.group(1) or 0)
                minutes = int(clock_match.group(2))
                seconds = hours * 3600 + minutes * 60 + int(clock_match.group(3))
            if seconds is not None:
                parsed = urlsplit(payload["source_url"])
                query = dict(parse_qsl(parsed.query, keep_blank_values=True))
                query["t"] = f"{seconds}s"
                payload["citation_url"] = urlunsplit(
                    (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
                )
    elif page_match:
        payload["page"] = int(page_match.group(1))
        payload["locator"] = f"page {payload['page']}"
    elif chapter_match:
        payload["chapter"] = int(chapter_match.group(1))
        payload["locator"] = f"chapter {payload['chapter']}"
    payload["source_id"] = payload.get("id") or _fallback_hormozi_source_id(pack_slug, payload.get("path"))
    payload["title"] = payload.get("title") or payload.get("path")
    payload["content_type"] = payload.get("type") or "untyped"
    payload["confidence"] = payload.get("confidence") or "ungraded"
    payload["file_provenance"] = payload.get("path")
    if payload.get("locator") and payload.get("path"):
        payload["citation"] = f"{payload['path']} {payload['locator']}"
    return payload


def create_pack_mcp(
    slug: str,
    pack: Pack,
    engine: RetrievalEngine,
    graph_lookup: GraphLookup | None = None,
    query_log_path: str | None = None,
):
    """Create an MCPServer instance with tools, resources, and prompts for a pack."""
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer(
        f"ep-mcp-{slug}",
        instructions=_get_server_instructions(pack),
        version=pack.version,
    )

    @mcp.tool(
        annotations={
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def ep_search_tool(
        query: str,
        type: str | None = None,
        tags: list[str] | None = None,
        max_results: int = 10,
        reconstruct: bool = False,
    ) -> list | dict:
        """Search the ExpertPack for relevant domain expertise.

        Args:
            query: Natural language search query.
            type: Filter by content type (concept, workflow, reference,
                  troubleshooting, faq, specification, etc.)
            tags: Filter by content tags. Results must match at least one.
            max_results: Maximum results to return (1-50, default 10).
            reconstruct: Include original markdown spans and provenance blocks
                for verification/reconstruction (default false).

        Returns:
            Ranked results with provenance metadata.
        """
        try:
            return await ep_search(
                engine, query, type, tags, max_results,
                query_log_path=query_log_path,
                reconstruct=reconstruct,
            )
        except Exception as e:
            logger.exception(
                "ep_search_tool error | pack=%s query=%r", slug, query,
            )
            return {"error": str(e), "pack": slug, "query": query}

    @mcp.tool(
        annotations={
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def ep_list_topics_tool(
        type: str | None = None,
    ) -> dict:
        """List available topics and content structure in the ExpertPack.

        Args:
            type: Filter by content type. If omitted, returns all types.

        Returns:
            Pack metadata and grouped file listing.
        """
        try:
            return ep_list_topics(pack, type)
        except Exception as e:
            logger.exception(
                "ep_list_topics_tool error | pack=%s type=%s", slug, type,
            )
            return {"error": str(e), "pack": slug}

    @mcp.tool(
        annotations={
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def ep_graph_traverse_tool(
        file_path: str,
        depth: int = 1,
        edge_kinds: list[str] | None = None,
    ) -> dict:
        """Traverse the ExpertPack knowledge graph from a starting file.

        Explores connections between content files (concepts, workflows,
        references, etc.) through the pack's knowledge graph.

        Args:
            file_path: Starting file path (e.g. 'concepts/auto-build.md').
            depth: Number of hops to follow (1-3, default 1).
            edge_kinds: Filter by edge types (wikilink, related, context).
                       If omitted, follows all edge types.

        Returns:
            Start node info, connected nodes, and traversal stats.
        """
        try:
            return ep_graph_traverse(
                pack=pack,
                graph_lookup=graph_lookup,
                file_path=file_path,
                depth=depth,
                edge_kinds=edge_kinds,
            )
        except Exception as e:
            logger.exception(
                "ep_graph_traverse_tool error | pack=%s file_path=%r",
                slug, file_path,
            )
            return {"error": str(e), "pack": slug, "file_path": file_path}

    @mcp.tool(
        annotations={
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def ep_read_tool(
        path: str | None = None,
        id: str | None = None,
        reconstruct: bool = False,
    ) -> dict:
        """Read a whole ExpertPack atom by path or provenance id.

        Search hits are locators. Call this after ep_search to load the
        complete atom (opening paragraph plus body), not a sidecar fragment.

        Args:
            path: Pack-relative file path (e.g. 'concepts/routing.md').
            id: Provenance id (e.g. 'my-pack/concepts/routing').
            reconstruct: Include original markdown and provenance block.

        Returns:
            Full atom content plus requires/activation metadata.
        """
        try:
            return ep_read(pack, path=path, id=id, reconstruct=reconstruct)
        except Exception as e:
            logger.exception(
                "ep_read_tool error | pack=%s path=%r id=%r", slug, path, id,
            )
            return {"error": str(e), "pack": slug, "path": path, "id": id}

    # The Hormozi brain is a single composite corpus.  Keep the standard EP
    # tools available, and add stable agent-first names that do not require a
    # caller to know the internal pack layout.
    if slug == "alex-hormozi-brain":
        @mcp.tool(
            annotations={
                "readOnlyHint": True,
                "idempotentHint": True,
                "openWorldHint": False,
            }
        )
        async def search_hormozi_brain(
            query: str,
            source_scope: str | None = None,
            tags: list[str] | None = None,
            max_results: int = 10,
        ) -> dict:
            """Search the unified Hormozi evidence, transcript, and skills brain.

            Results always include a source file, content hash, confidence, and
            a citation locator.  The result is a source-grounded perspective,
            not an assertion that Alex Hormozi personally made the answer.
            """
            normalized_scope = source_scope.strip().casefold() if source_scope else None
            scoped_prefix = normalized_scope in {
                "evidence",
                "youtube",
                "curated-skills",
                "agent-skills",
                "ebook",
                "audio",
                "ocr",
            }
            # Path scopes are applied after retrieval because they are pack
            # layout filters rather than ExpertPack frontmatter types. Fetch a
            # wider candidate window first so a narrow scope is not starved by
            # unrelated higher-ranked files, then honor the caller's limit.
            retrieval_limit = max_results
            if scoped_prefix:
                retrieval_limit = min(50, max(max_results, max_results * 4))
            results = await ep_search(
                engine,
                query,
                type=normalized_scope if normalized_scope in {"reference", "workflow", "concept", "decision", "gotcha", "phase"} else None,
                tags=tags,
                max_results=retrieval_limit,
                query_log_path=query_log_path,
            )
            if scoped_prefix:
                results = [
                    result for result in results
                    if str(result.get("source_file", "")).startswith(normalized_scope + "/")
                ]
                results = results[:max_results]
            for result in results:
                result_text = str(result.get("text", ""))
                url_match = re.search(r"YouTube URL:\s*(https?://\S+)", result_text)
                video_match = re.search(r"Video ID:\s*`([^`]+)`", result_text)
                timestamp_match = re.search(r"Timestamp range:\s*([^\n]+)", result_text)
                page_match = re.search(
                    r"(?:\b(?:Source )?Page:\s*|\bpages?\s+)([0-9]+)",
                    f"{result.get('title', '')} {result_text}",
                    re.IGNORECASE,
                )
                chapter_match = re.search(r"(?:EPUB )?chapter\s+([0-9]+)", f"{result.get('title', '')} {result_text}", re.IGNORECASE)
                if url_match:
                    result["source_url"] = url_match.group(1).rstrip("`),")
                if video_match:
                    result["video_id"] = video_match.group(1)
                if timestamp_match:
                    result["timestamp"] = timestamp_match.group(1).strip()
                    result["locator"] = f"timestamp {result['timestamp']}"
                    # Keep the human-readable range, but also emit a direct
                    # YouTube moment link whenever the transcript has a
                    # numeric start locator.  This makes citations actionable
                    # without asking an agent to reconstruct the URL itself.
                    if result.get("source_url"):
                        start = result["timestamp"].split("-", 1)[0].strip()
                        seconds: int | None = None
                        seconds_match = re.fullmatch(r"(\d+(?:\.\d+)?)s", start)
                        clock_match = re.fullmatch(r"(?:(\d+):)?(\d{1,2}):(\d{2})", start)
                        if seconds_match:
                            seconds = max(0, int(float(seconds_match.group(1))))
                        elif clock_match:
                            hours = int(clock_match.group(1) or 0)
                            minutes = int(clock_match.group(2))
                            seconds = hours * 3600 + minutes * 60 + int(clock_match.group(3))
                        if seconds is not None:
                            parsed = urlsplit(result["source_url"])
                            query = dict(parse_qsl(parsed.query, keep_blank_values=True))
                            query["t"] = f"{seconds}s"
                            result["citation_url"] = urlunsplit(
                                (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
                            )
                elif page_match:
                    result["page"] = int(page_match.group(1))
                    result["locator"] = f"page {result['page']}"
                elif chapter_match:
                    result["chapter"] = int(chapter_match.group(1))
                    result["locator"] = f"chapter {result['chapter']}"
            for result in results:
                locator = result.get("line_range")
                line_locator = f"lines {locator[0]}-{locator[1]}" if locator else None
                if line_locator and not result.get("locator"):
                    result["locator"] = line_locator
                citation_locator = result.get("locator") or line_locator or f"chunk {result.get('chunk_index', 0)}"
                citation = f"{result.get('source_file')} {citation_locator}"
                result["source_id"] = result.get("id") or _fallback_hormozi_source_id(slug, result.get("source_file"))
                result["title"] = result.get("title") or result.get("source_file")
                result["content_type"] = result.get("type") or "untyped"
                result["confidence"] = result.get("confidence") or "ungraded"
                result["file_provenance"] = result.get("source_file")
                result["citation"] = citation
                result["source_scope"] = source_scope
            return {
                "query": query,
                "perspective_disclaimer": "Synthesized from retrieved source material; not a current statement or endorsement by Alex Hormozi.",
                "results": results,
            }

        @mcp.tool(
            annotations={
                "readOnlyHint": True,
                "idempotentHint": True,
                "openWorldHint": False,
            }
        )
        async def get_hormozi_source(
            source_id: str | None = None,
            path: str | None = None,
            reconstruct: bool = False,
        ) -> dict:
            """Read a complete cited source atom by ID or pack-relative path."""
            resolved_id = source_id
            file_prefix = f"{slug}/file/"
            if resolved_id and resolved_id.startswith(file_prefix):
                path = resolved_id.removeprefix(file_prefix)
                resolved_id = None
            if resolved_id and resolved_id.startswith("youtube-"):
                video_id = resolved_id.removeprefix("youtube-")
                prefix = "youtube/"
                candidates = [
                    file_path for file_path in pack.files
                    if file_path.startswith(prefix) and file_path.endswith(f"-{video_id}-part-001.md")
                ]
                if candidates:
                    path = min(candidates)
                    return _enrich_hormozi_source(ep_read(pack, path=path, reconstruct=reconstruct), slug)
                resolved_id = f"alex-hormozi-brain/youtube/{video_id}"
            if path and (".." in Path(path).parts or path.startswith(("/", "\\"))):
                return {"error": "path must be pack-relative"}
            return _enrich_hormozi_source(ep_read(pack, path=path, id=resolved_id, reconstruct=reconstruct), slug)

        @mcp.tool(
            annotations={
                "readOnlyHint": True,
                "idempotentHint": True,
                "openWorldHint": False,
            }
        )
        async def get_hormozi_skill(skill_name: str) -> dict:
            """Return an executable Paperclip skill package entrypoint."""
            safe_name = Path(skill_name).name
            if safe_name != skill_name or safe_name in {"", ".", ".."}:
                return {"error": "skill_name must be a package name"}
            package_root = Path(pack.pack_dir).parents[1] / "skills" / "alex-hormozi" / safe_name
            skill_file = package_root / "SKILL.md"
            if not skill_file.is_file():
                # The concise MCP atom still gives a useful error for callers
                # that have not generated the executable mirror.
                return ep_read(pack, path=f"agent-skills/{safe_name}.md")
            raw_skill = skill_file.read_text(encoding="utf-8", errors="replace")
            skill_source_id = f"{pack.slug}/agent-skills/{safe_name}"
            return {
                "skill_name": safe_name,
                "source_id": skill_source_id,
                "title": safe_name,
                "content_type": "workflow",
                "confidence": "curated",
                "file_provenance": f"agent-skills/{safe_name}.md",
                "content": raw_skill,
                "files": sorted(str(path.relative_to(package_root)).replace("\\", "/") for path in package_root.rglob("*") if path.is_file()),
                "execution_note": "Use this workflow with retrieved citations; do not represent the output as Alex Hormozi's current personal statement.",
            }

        @mcp.tool(
            annotations={
                "readOnlyHint": True,
                "idempotentHint": True,
                "openWorldHint": False,
            }
        )
        async def get_brain_coverage(include_records: bool = False) -> dict:
            """Report source coverage, rights status, and freshness without raw paths."""
            import json
            from pathlib import Path

            report_path = Path(pack.pack_dir) / "meta" / "brain-coverage.json"
            if not report_path.is_file():
                return {"error": "coverage report is not present"}
            report = json.loads(report_path.read_text(encoding="utf-8"))
            transcript_report = report.get("transcripts", {})
            safe_transcripts = {
                key: transcript_report.get(key)
                for key in (
                    "sections_found", "unique_videos", "duplicate_sections_removed",
                    "transcript_atoms", "official_channel_enumeration",
                    "official_channel_video_count", "official_catalog_status_counts",
                    "official_caption_videos",
                )
                if key in transcript_report
            }
            container_report = report.get("extras", {}).get("containers", {})
            ocr_report = report.get("extras", {}).get("ocr", {})
            safe_ocr = {
                key: ocr_report.get(key)
                for key in (
                    "status", "requested_pages", "recovered_pages", "visual_qa_status",
                    "manual_review_pages", "manual_review_decision_counts",
                )
                if key in ocr_report
            }
            safe_containers = {
                "status": container_report.get("status"),
                "source_count": container_report.get("source_count", 0),
                "unique_knowledge_ingested": container_report.get("unique_knowledge_ingested"),
                "sources": [
                    {
                        key: item.get(key)
                        for key in (
                            "source_id", "type", "source_hash", "status", "member_count",
                            "manifest_member_matches", "row_count", "unique_knowledge_ingested",
                            "reason", "duplicate_group",
                        )
                        if key in item
                    }
                    for item in container_report.get("sources", [])
                    if isinstance(item, dict)
                ],
            }
            raw_coverage = report.get("coverage_categories", {})
            raw_categories = raw_coverage.get("categories", {}) if isinstance(raw_coverage, dict) else {}
            raw_missing = raw_coverage.get("missing", {}) if isinstance(raw_coverage, dict) else {}
            safe_coverage = {
                "categories": {
                    key: int(raw_categories.get(key, 0) or 0)
                    for key in ("included", "duplicate", "incomplete", "unsupported", "quarantined", "missing")
                },
                "missing": {
                    "inventory_records": len(raw_missing.get("inventory_records", []) or []),
                    "official_captionless_videos": [
                        {
                            key: item.get(key)
                            for key in ("video_id", "title", "status")
                            if key in item
                        }
                        for item in (raw_missing.get("official_captionless_videos", []) or [])
                        if isinstance(item, dict)
                    ],
                    "audio_pending_transcription": [
                        {"status": item.get("status")}
                        for item in (raw_missing.get("audio_pending_transcription", []) or [])
                        if isinstance(item, dict)
                    ],
                },
            }
            restricted_report = report.get("extras", {}).get("restricted", {})
            safe_restricted = {
                "status": restricted_report.get("status", "pending_external_authorization"),
                "unique_work_count": int(restricted_report.get("unique_work_count", 0) or 0),
                "ocr_requested": bool(restricted_report.get("ocr_requested", False)),
                "ocr_atoms": int(restricted_report.get("ocr_atoms", 0) or 0),
                "authorization_required": bool(restricted_report.get("authorization_required", False)),
                "reason": restricted_report.get("reason"),
                "source_ids": sorted(
                    str(item.get("source_id"))
                    for item in restricted_report.get("sources", [])
                    if isinstance(item, dict) and item.get("source_id")
                ),
            }
            audio_records = {
                str(record.get("title", "")): record
                for record in report.get("records", [])
                if isinstance(record, dict) and record.get("kind") == "derived_audio"
            }
            safe_audio = []
            for item in report.get("extras", {}).get("audio", []):
                if not isinstance(item, dict):
                    continue
                raw_path = str(item.get("path", ""))
                title = raw_path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
                record = audio_records.get(title, {})
                plan = item.get("transcription_plan", {})
                metadata = item.get("metadata", {})
                safe_audio.append({
                    "source_id": item.get("source_id") or record.get("record_id"),
                    "title": record.get("title") or title,
                    "content_type": "audio",
                    "status": item.get("status"),
                    "duration_seconds": metadata.get("duration"),
                    "estimated_chunks": plan.get("chunk_count"),
                    "model": plan.get("model"),
                    "timestamped": plan.get("timestamped_segments"),
                })
            response = {
                "inventory_records": report.get("inventory_records", 0),
                "derived_records": report.get("derived_records", 0),
                "summary": report.get("summary", {}),
                "transcripts": safe_transcripts,
                "skills": {
                    "status": report.get("skills", {}).get("status"),
                    "package_count": len(report.get("skills", {}).get("packages", [])),
                    "invalid": report.get("skills", {}).get("invalid", []),
                },
                "containers": safe_containers,
                "ocr": safe_ocr,
                "coverage_categories": safe_coverage,
                "restricted": safe_restricted,
                "freshness": pack.freshness.model_dump() if pack.freshness else {},
                "pending": {
                    "restricted_sources": sum(
                        1
                        for record in report.get("records", [])
                        if isinstance(record, dict)
                        and record.get("status") == "quarantined_restricted_authorization_required"
                    ),
                    "restricted_resolution_status": safe_restricted["status"],
                    "ocr_pages": max(
                        int(ocr_report.get("requested_pages", 0) or 0)
                        - int(ocr_report.get("recovered_pages", 0) or 0),
                        0,
                    ),
                    "ocr_manual_review_pages": (
                        0
                        if ocr_report.get("visual_qa_status") == "manual_review_complete"
                        else int(ocr_report.get("manual_review_pages", 0) or 0)
                    ),
                    "audio": safe_audio,
                    "official_channel_enumeration": report.get("transcripts", {}).get("official_channel_enumeration"),
                },
            }
            if include_records:
                response["records"] = [
                    {
                        key: record.get(key)
                        for key in ("record_id", "kind", "title", "format", "status", "rights_status", "source_url", "video_id", "duplicate_of", "pack_membership")
                        if key in record
                    }
                    for record in report.get("records", [])
                ]
            return response

    # Register resources (always-tier files, overview, manifest, additional declared)
    register_resources(mcp, pack)

    # Register prompts (from mcp.prompts manifest declarations or auto-discovered workflows)
    register_prompts(mcp, pack)

    return mcp


async def init_pack(
    slug: str,
    pack_path: str,
    provider: EmbeddingProvider,
    config: ServerConfig,
    index_dir: str | None = None,
) -> PackInstance:
    """Load, index, and initialize a pack with MCP tools."""
    pack = load_pack(pack_path, slug_override=slug, index_dir_override=index_dir)
    store = SQLiteStore(pack.index_path, embedding_dimension=provider.dimension)
    store.open()

    manager = IndexManager(pack, store, provider)
    stats = await manager.build_index()
    logger.info("Pack '%s' indexed: %s", slug, stats)

    graph_lookup = GraphLookup.from_pack(pack)

    # Build effective retrieval config: start with global, apply pack-level overrides
    pack_cfg = next((p for p in config.packs if p.slug == slug), None)
    retrieval_cfg = config.retrieval
    if pack_cfg is not None:
        overrides = {}
        if pack_cfg.graph_expansion_enabled is not None:
            overrides["graph_expansion_enabled"] = pack_cfg.graph_expansion_enabled
        if pack_cfg.graph_expansion_confidence_threshold is not None:
            overrides["graph_expansion_confidence_threshold"] = pack_cfg.graph_expansion_confidence_threshold
        if pack_cfg.graph_expansion_min_score is not None:
            overrides["graph_expansion_min_score"] = pack_cfg.graph_expansion_min_score
        if pack_cfg.graph_expansion_structural_bonus is not None:
            overrides["graph_expansion_structural_bonus"] = pack_cfg.graph_expansion_structural_bonus
        if overrides:
            retrieval_cfg = retrieval_cfg.model_copy(update=overrides)
            logger.info("Pack '%s' applying retrieval overrides: %s", slug, overrides)

    reranker_cfg = config.reranker
    reranker = Reranker(
        model_name=reranker_cfg.model,
        candidate_pool_size=reranker_cfg.candidate_pool_size,
        enabled=reranker_cfg.enabled,
        max_chars=reranker_cfg.max_chars,
        batch_size=reranker_cfg.batch_size,
    )
    engine = RetrievalEngine(pack, store, provider, retrieval_cfg, graph_lookup, reranker=reranker)
    mcp = create_pack_mcp(slug, pack, engine, graph_lookup, query_log_path=config.query_log_path)

    return PackInstance(pack=pack, store=store, engine=engine, mcp=mcp, index_manager=manager)


def build_app(
    config: ServerConfig,
    pack_instances: dict[str, PackInstance],
    dev_watch: bool = False,
) -> Starlette:
    """Build the Starlette ASGI application with pack routing.

    Each pack's MCPServer Streamable HTTP app is mounted as a sub-application
    with its own lifespan managed through the parent app's lifespan.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    # Loopback development may intentionally omit a key. Any network bind is
    # fail-closed when EP_MCP_KEY_<PACK> is missing.
    auth = APIKeyAuth(allow_open=_loopback_host(config.host))
    for pack_config in config.packs:
        # Register every configured pack so EP_MCP_KEY_<SLUG> works even when
        # the secret is supplied only through the environment.
        auth.add_pack_keys(pack_config.slug, pack_config.api_keys)

    limiter = None
    if config.rate_limit.enabled:
        limiter = _TokenBucketRateLimiter(
            config.rate_limit.requests_per_minute,
            config.rate_limit.burst,
        )

    def rate_limit_response(scope: dict, pack_slug: str) -> JSONResponse | None:
        if limiter is None:
            return None
        client = scope.get("client")
        client_host = client[0] if isinstance(client, (tuple, list)) and client else "unknown"
        allowed, retry_after = limiter.allow(f"{pack_slug}:{client_host}")
        if allowed:
            return None
        return JSONResponse(
            {"error": "Rate limit exceeded", "pack": pack_slug},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )

    transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(config.mcp_allowed_hosts),
        allowed_origins=list(config.mcp_allowed_origins),
    )

    # Collect MCP session managers for lifespan management
    session_managers = []
    for slug, inst in pack_instances.items():
        mcp_app = inst.mcp.streamable_http_app(
            stateless_http=True,
            transport_security=transport_security,
            host=config.host,
        )
        session_managers.append((slug, inst.mcp.session_manager, mcp_app))

    @asynccontextmanager
    async def lifespan(app):
        """Manage all pack MCP session managers and optional file watchers."""
        from contextlib import AsyncExitStack
        async with AsyncExitStack() as stack:
            for slug, sm, _ in session_managers:
                logger.info("Starting session manager for pack '%s'", slug)
                await stack.enter_async_context(sm.run())

            # Start file watchers in dev mode
            watchers = []
            if dev_watch:
                try:
                    from .index.watcher import start_watchers
                    loop = asyncio.get_event_loop()
                    watchers = start_watchers(
                        list(pack_instances.values()), loop
                    )
                    if watchers:
                        logger.info(
                            "Dev file watch enabled for %d pack(s)", len(watchers)
                        )
                    else:
                        logger.warning(
                            "Dev file watch requested but no watchers started "
                            "(watchdog installed?)"
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not start file watchers: %s", exc)

            yield

            for w in watchers:
                w.stop()

    async def health(request: Request) -> JSONResponse:
        packs_info = {}
        for slug, inst in pack_instances.items():
            packs_info[slug] = {
                "name": inst.pack.name,
                "files": len(inst.pack.files),
                "chunks": inst.store.chunk_count(),
                "version": inst.pack.version,
            }
        return JSONResponse({"status": "healthy", "packs": packs_info})

    async def list_packs(request: Request) -> JSONResponse:
        packs = []
        for slug, inst in pack_instances.items():
            packs.append({
                "slug": slug,
                "name": inst.pack.name,
                "type": inst.pack.type,
                "version": inst.pack.version,
                "file_count": len(inst.pack.files),
            })
        return JSONResponse({"packs": packs})

    async def search(request: Request) -> JSONResponse:
        """GET /search?q=<query>&pack=<slug>&n=<max_results>&type=<type>&tags=<tag1,tag2>

        Lightweight HTTP search endpoint for non-MCP clients (e.g. web_fetch, curl).
        Requires Bearer token if API keys are configured for the pack.

        Tuning overrides (optional, for eval/tuning only — not for production use):
          graph_expansion_confidence_threshold=<float>
          graph_expansion_min_score=<float>
        """
        q = request.query_params.get("q", "").strip()
        slug = request.query_params.get("pack", "").strip()
        n = int(request.query_params.get("n", "10"))
        type_filter = request.query_params.get("type", None)
        tags_raw = request.query_params.get("tags", None)
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else None
        reconstruct = request.query_params.get("reconstruct", "").lower() in {"1", "true", "yes"}

        # Optional tuning overrides
        conf_raw = request.query_params.get("graph_expansion_confidence_threshold", None)
        min_raw = request.query_params.get("graph_expansion_min_score", None)
        conf_override = float(conf_raw) if conf_raw is not None else None
        min_override = float(min_raw) if min_raw is not None else None

        if not q:
            return JSONResponse({"error": "Missing required parameter: q"}, status_code=400)
        if not slug:
            return JSONResponse({"error": "Missing required parameter: pack"}, status_code=400)
        if slug not in pack_instances:
            return JSONResponse({"error": f"Unknown pack: {slug}"}, status_code=404)

        # Auth check
        auth_header = request.headers.get("Authorization", "")
        if not auth.authenticate(auth_header, slug):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        limited = rate_limit_response(request.scope, slug)
        if limited is not None:
            return limited

        try:
            engine = pack_instances[slug].engine
            from .retrieval.models import SearchRequest
            search_req = SearchRequest(
                query=q,
                type=type_filter,
                tags=tags,
                max_results=n,
                reconstruct=reconstruct,
            )
            _t0 = time.monotonic()
            raw_results = await engine.search(
                search_req,
                graph_expansion_confidence_threshold=conf_override,
                graph_expansion_min_score=min_override,
            )
            _elapsed_ms = (time.monotonic() - _t0) * 1000
            if config.query_log_path:
                _embed_cached = getattr(engine.provider, "last_cache_hit", None)
                log_query(
                    config.query_log_path, slug, q, raw_results,
                    _elapsed_ms, _embed_cached, type=type_filter, tags=tags,
                )
            return JSONResponse({
                "query": q,
                "pack": slug,
                "results": [_result_to_http_dict(r) for r in raw_results],
            })
        except Exception as exc:
            logger.exception("search endpoint error | pack=%s query=%r", slug, q)
            return JSONResponse({"error": str(exc)}, status_code=500)

    async def search_post(request: Request) -> JSONResponse:
        """POST /search with JSON body.

        Same semantics as ``GET /search`` but accepts a JSON body so clients can
        pass a pre-computed query embedding (``vector``) without blowing out the
        URL. This is the entry point used by OpenClaw's memory plugin to avoid
        a second round-trip to Gemini when the caller already embedded the
        query upstream.

        Body schema::

            {
              "query": "search string",
              "pack": "ezt-designer",
              "n": 10,
              "type": null,
              "tags": null,
              "vector": [float, ...],          // optional
              "reconstruct": false,            // optional
              "graph_expansion_confidence_threshold": 0.55,  // optional
              "graph_expansion_min_score": 0.45              // optional
            }
        """
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "JSON body must be an object"}, status_code=400)

        q = (body.get("query") or "").strip()
        slug = (body.get("pack") or "").strip()
        n = int(body.get("n", 10))
        type_filter = body.get("type")
        tags_raw = body.get("tags")
        if isinstance(tags_raw, str):
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        elif isinstance(tags_raw, list):
            tags = [str(t).strip() for t in tags_raw if str(t).strip()]
        else:
            tags = None

        conf_raw = body.get("graph_expansion_confidence_threshold")
        min_raw = body.get("graph_expansion_min_score")
        conf_override = float(conf_raw) if conf_raw is not None else None
        min_override = float(min_raw) if min_raw is not None else None
        reconstruct = bool(body.get("reconstruct", False))

        vector = body.get("vector")
        if vector is not None:
            if not isinstance(vector, list) or not all(
                isinstance(v, (int, float)) for v in vector
            ):
                return JSONResponse(
                    {"error": "vector must be a list of numbers"}, status_code=400,
                )
            # Convert ints to floats for downstream arithmetic safety.
            vector = [float(v) for v in vector]

        if not q:
            return JSONResponse({"error": "Missing required parameter: query"}, status_code=400)
        if not slug:
            return JSONResponse({"error": "Missing required parameter: pack"}, status_code=400)
        if slug not in pack_instances:
            return JSONResponse({"error": f"Unknown pack: {slug}"}, status_code=404)

        # Auth check
        auth_header = request.headers.get("Authorization", "")
        if not auth.authenticate(auth_header, slug):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        limited = rate_limit_response(request.scope, slug)
        if limited is not None:
            return limited

        # Dimension validation against the pack's configured embedding provider.
        if vector is not None:
            inst = pack_instances[slug]
            expected_dim = getattr(inst.engine.provider, "dimension", None)
            if expected_dim is not None and len(vector) != expected_dim:
                return JSONResponse(
                    {
                        "error": (
                            f"vector dimension mismatch: got {len(vector)}, "
                            f"expected {expected_dim} for pack '{slug}'"
                        )
                    },
                    status_code=400,
                )

        try:
            engine = pack_instances[slug].engine
            from .retrieval.models import SearchRequest
            search_req = SearchRequest(
                query=q,
                type=type_filter,
                tags=tags,
                max_results=n,
                vector=vector,
                reconstruct=reconstruct,
            )
            _t0 = time.monotonic()
            raw_results = await engine.search(
                search_req,
                graph_expansion_confidence_threshold=conf_override,
                graph_expansion_min_score=min_override,
            )
            _elapsed_ms = (time.monotonic() - _t0) * 1000
            if config.query_log_path:
                _embed_cached = getattr(engine.provider, "last_cache_hit", None)
                log_query(
                    config.query_log_path, slug, q, raw_results,
                    _elapsed_ms, _embed_cached, type=type_filter, tags=tags,
                )
            return JSONResponse({
                "query": q,
                "pack": slug,
                "vector_supplied": vector is not None,
                "results": [_result_to_http_dict(r) for r in raw_results],
            })
        except Exception as exc:
            logger.exception("search(POST) endpoint error | pack=%s query=%r", slug, q)
            return JSONResponse({"error": str(exc)}, status_code=500)

    def _result_to_http_dict(result) -> dict:
        """Serialize SearchResult for lightweight HTTP endpoints."""
        return {
            "source_file": result.source_file,
            "title": result.title,
            "text": result.text,
            "score": round(result.score, 4),
            "type": result.type,
            "tags": result.tags,
            "graph_expanded": result.graph_expanded,
            "requires_expanded": result.requires_expanded,
            "id": result.id,
            "content_hash": result.content_hash,
            "verified_at": result.verified_at,
            "chunk_index": result.chunk_index,
            "original_span": result.original_span,
            "original_markdown": result.original_markdown,
            "provenance_block": result.provenance_block,
            "byte_offset": result.byte_offset,
            "fragment_id": result.fragment_id,
            "line_range": list(result.line_range) if result.line_range else None,
            "confidence": result.confidence,
            "excerpt": result.excerpt,
            "stale": result.stale,
        }

    routes = [
        Route("/health", health),
        Route("/packs", list_packs),
        Route("/search", search),
        Route("/search", search_post, methods=["POST"]),
    ]

    def authenticated_mcp_app(app, pack_slug: str):
        """Protect mounted MCP routes with the same per-pack API-key policy.

        The direct REST search route already checks ``APIKeyAuth``.  The MCP
        transport is mounted as a sub-application, so it needs an explicit
        ASGI wrapper as well; otherwise a company-network deployment would
        accidentally expose the tool surface while REST remained protected.
        """

        async def wrapped(scope, receive, send):
            if scope.get("type") == "http":
                headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}
                if not auth.authenticate(headers.get("authorization", ""), pack_slug):
                    response = JSONResponse({"error": "Unauthorized"}, status_code=401)
                    await response(scope, receive, send)
                    return
                limited = rate_limit_response(scope, pack_slug)
                if limited is not None:
                    await limited(scope, receive, send)
                    return
            await app(scope, receive, send)

        return wrapped

    for slug, sm, mcp_app in session_managers:
        routes.append(Mount(f"/packs/{slug}", app=authenticated_mcp_app(mcp_app, slug)))
        logger.info("Mounted MCP endpoint: /packs/%s/mcp", slug)

    return Starlette(routes=routes, lifespan=lifespan)
