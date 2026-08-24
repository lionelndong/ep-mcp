import json

from ep_mcp.retrieval.models import SearchResult
from ep_mcp.tools.ep_search import log_query


def test_query_log_excludes_retrieved_source_text(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    result = SearchResult(
        text="SECRET_SOURCE_SENTENCE",
        source_file="evidence/source.md",
        id="source-1",
        score=0.9,
        title="Source",
        excerpt="SECRET_SOURCE_SENTENCE",
    )
    log_query(str(log_path), "alex-hormozi-brain", "pricing question", [result], 1.2, False)
    record = json.loads(log_path.read_text(encoding="utf-8"))
    serialized = json.dumps(record)
    assert "SECRET_SOURCE_SENTENCE" not in serialized
    assert record["chunks"] == ["evidence/source.md"]
    assert record["pack"] == "alex-hormozi-brain"
