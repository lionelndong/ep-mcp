from ep_mcp.retrieval.scorer import normalize_bm25_scores, score_bm25_fallback
from ep_mcp.retrieval.stopwords import STOPWORDS


def test_bm25_fallback_uses_shared_stopwords_without_name_error():
    results = normalize_bm25_scores([{"chunk_id": 1, "bm25_score": -1.0}])
    chunks = {
        1: {
            "content": "A pricing framework for increasing offer value",
            "file_path": "pricing-framework.md",
        }
    }

    scores = score_bm25_fallback(results, chunks, "How do I improve pricing?", path_boost=0.0)

    assert "how" in STOPWORDS
    assert scores[1] > results[0]["bm25_norm"]
