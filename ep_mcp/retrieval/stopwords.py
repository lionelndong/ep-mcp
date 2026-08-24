"""Stopwords shared by FTS5 query sanitization and lexical fallback scoring."""

from __future__ import annotations

# English words that tend to break strict FTS matching without adding useful
# retrieval signal. Keep this in one module so BM25 and fallback scoring use
# identical query-token semantics.
STOPWORDS = frozenset({
    # Question / auxiliary words
    "what", "which", "how", "why", "when", "where", "who", "whom", "whose",
    "does", "do", "did", "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "will", "would", "could", "should", "can", "may",
    "might", "shall", "must", "need", "dare", "used",
    # Articles / determiners
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "its",
    "our", "their", "his", "her", "some", "any", "all", "both", "each",
    "every", "few", "more", "most", "other", "such", "no", "not", "only",
    "same", "so", "than", "too", "very",
    # Prepositions / conjunctions
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "from",
    "up", "down", "out", "off", "over", "under", "again", "then", "once",
    "of", "to", "as", "if", "or", "and", "but", "nor", "yet", "while",
    "although", "because", "since", "unless", "until", "whether",
    # Common filler
    "i", "me", "we", "us", "you", "he", "she", "they", "them", "it",
    "get", "use", "make", "tell", "know", "want", "like", "just",
    "also", "back", "even", "still", "way", "well", "new", "old",
    "please", "help", "show", "give", "look", "see",
})
