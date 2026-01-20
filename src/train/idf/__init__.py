"""
IDF computation infrastructure for V25 SPLADE training.

Provides:
- IDFComputer: Efficient IDF computation with caching
- Korean stopword handling for XLM-RoBERTa tokenizer
"""

from .idf_computer import IDFComputer, compute_idf_from_corpus, load_or_compute_idf
from .korean_stopwords import (
    create_stopword_mask,
    get_korean_stopword_ids,
    KOREAN_STOPWORDS,
)

__all__ = [
    "IDFComputer",
    "compute_idf_from_corpus",
    "load_or_compute_idf",
    "create_stopword_mask",
    "get_korean_stopword_ids",
    "KOREAN_STOPWORDS",
]
