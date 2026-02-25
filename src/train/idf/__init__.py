"""
IDF computation infrastructure for V25/V26/V28 SPLADE training.

Provides:
- IDFComputer: Efficient IDF computation with caching
- Korean stopword handling for XLM-RoBERTa tokenizer
- V26: Enhanced stopword handling with special token separation
- V28: Korean token identification for language filtering
"""

from .idf_computer import IDFComputer, compute_idf_from_corpus, load_or_compute_idf
from .korean_stopwords import (
    create_stopword_mask,
    create_stopword_mask_v26,
    expand_stopwords_by_idf,
    get_korean_stopword_ids,
    get_korean_stopword_ids_v26,
    get_special_token_ids_only,
    KOREAN_STOPWORDS,
    KOREAN_STOPWORDS_V26,
)
from .korean_tokens import (
    build_korean_token_ids,
    build_non_korean_token_ids,
    create_language_penalty_mask,
    load_or_compute_korean_tokens,
    is_korean_char,
    get_token_language,
    analyze_token_language_distribution,
)

__all__ = [
    # IDF computation
    "IDFComputer",
    "compute_idf_from_corpus",
    "load_or_compute_idf",
    # V25 stopword handling
    "create_stopword_mask",
    "get_korean_stopword_ids",
    "KOREAN_STOPWORDS",
    # V26 enhanced stopword handling
    "create_stopword_mask_v26",
    "expand_stopwords_by_idf",
    "get_korean_stopword_ids_v26",
    "get_special_token_ids_only",
    "KOREAN_STOPWORDS_V26",
    # V28 Korean token identification
    "build_korean_token_ids",
    "build_non_korean_token_ids",
    "create_language_penalty_mask",
    "load_or_compute_korean_tokens",
    "is_korean_char",
    "get_token_language",
    "analyze_token_language_distribution",
]
