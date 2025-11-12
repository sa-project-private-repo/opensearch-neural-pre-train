"""
OpenSearch Neural Sparse Training - Source Package

This package contains core modules for training neural sparse retrieval models
with temporal analysis and unsupervised synonym discovery.

Modules:
    - losses: Improved contrastive loss functions for neural sparse training
    - data_loader: News data loading with temporal information preserved
    - temporal_analysis: Time-weighted IDF and automatic trend detection
    - negative_sampling: Hard negative mining with BM25
    - temporal_clustering: Time-based clustering for synonym discovery
    - cross_lingual_synonyms: Korean-English bilingual synonym discovery

Key Features:
    ✓ In-batch negatives contrastive loss (fixes BCE issue)
    ✓ Temporal IDF with exponential decay weighting
    ✓ Automatic trend detection (replaces hardcoded TREND_BOOST)
    ✓ BM25-based hard negative mining
    ✓ Bilingual synonyms (Korean ↔ English, e.g., '모델' ↔ 'model')
    ✓ Fully unsupervised approach

Quick Start:
    >>> from src import (
    ...     load_korean_news_with_dates,
    ...     calculate_temporal_idf,
    ...     neural_sparse_loss_with_regularization,
    ...     build_comprehensive_bilingual_dictionary,
    ... )
"""

__version__ = "0.3.0"
__author__ = "OpenSearch Neural Sparse Team"

# === Loss functions ===
from src.losses import (
    in_batch_negatives_loss,
    margin_ranking_loss,
    contrastive_loss_with_hard_negatives,
    neural_sparse_loss_with_regularization,
    compute_sparsity_metrics,
)

# === Data loading ===
from src.data_loader import (
    load_korean_news_with_dates,
    load_multiple_korean_datasets,
    create_time_windows,
    get_recent_documents,
)

# === Temporal analysis ===
from src.temporal_analysis import (
    calculate_temporal_idf,
    calculate_windowed_idf,
    detect_trending_tokens,
    build_trend_boost_dict,
    apply_temporal_boost_to_idf,
    analyze_token_frequency_over_time,
)

# === Negative sampling ===
from src.negative_sampling import (
    add_hard_negatives_bm25,
    add_random_negatives,
    add_mixed_negatives,
    balance_positive_negative_ratio,
)

# === Temporal clustering (synonym discovery) ===
from src.temporal_clustering import (
    cluster_tokens_temporal,
    build_synonym_groups_from_clusters,
    discover_synonyms_temporal,
    merge_synonym_dictionaries,
    filter_synonyms_by_frequency,
)

# === Cross-lingual synonyms ===
from src.cross_lingual_synonyms import (
    extract_bilingual_terms,
    discover_cross_lingual_synonyms_by_embedding,
    build_comprehensive_bilingual_dictionary,
    get_default_korean_english_pairs,
    apply_bilingual_synonyms_to_idf,
)

__all__ = [
    # === Version info ===
    "__version__",
    "__author__",

    # === Loss functions ===
    "in_batch_negatives_loss",
    "margin_ranking_loss",
    "contrastive_loss_with_hard_negatives",
    "neural_sparse_loss_with_regularization",
    "compute_sparsity_metrics",

    # === Data loading ===
    "load_korean_news_with_dates",
    "load_multiple_korean_datasets",
    "create_time_windows",
    "get_recent_documents",

    # === Temporal analysis ===
    "calculate_temporal_idf",
    "calculate_windowed_idf",
    "detect_trending_tokens",
    "build_trend_boost_dict",
    "apply_temporal_boost_to_idf",
    "analyze_token_frequency_over_time",

    # === Negative sampling ===
    "add_hard_negatives_bm25",
    "add_random_negatives",
    "add_mixed_negatives",
    "balance_positive_negative_ratio",

    # === Temporal clustering ===
    "cluster_tokens_temporal",
    "build_synonym_groups_from_clusters",
    "discover_synonyms_temporal",
    "merge_synonym_dictionaries",
    "filter_synonyms_by_frequency",

    # === Cross-lingual synonyms ===
    "extract_bilingual_terms",
    "discover_cross_lingual_synonyms_by_embedding",
    "build_comprehensive_bilingual_dictionary",
    "get_default_korean_english_pairs",
    "apply_bilingual_synonyms_to_idf",
]
