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

Key Features:
    ✓ In-batch negatives contrastive loss (fixes BCE issue)
    ✓ Temporal IDF with exponential decay weighting
    ✓ Automatic trend detection (replaces hardcoded TREND_BOOST)
    ✓ BM25-based hard negative mining
    ✓ Fully unsupervised approach
"""

__version__ = "0.3.0"
__author__ = "OpenSearch Neural Sparse Team"

# Loss functions
from src.losses import (
    in_batch_negatives_loss,
    margin_ranking_loss,
    contrastive_loss_with_hard_negatives,
    neural_sparse_loss_with_regularization,
    compute_sparsity_metrics,
)

# Data loading
from src.data_loader import (
    load_korean_news_with_dates,
    load_multiple_korean_datasets,
)

# Temporal analysis
from src.temporal_analysis import (
    calculate_temporal_idf,
    detect_trending_tokens,
    build_trend_boost_dict,
)

# Negative sampling
from src.negative_sampling import (
    add_hard_negatives_bm25,
    add_mixed_negatives,
)

__all__ = [
    # Losses
    "in_batch_negatives_loss",
    "margin_ranking_loss",
    "contrastive_loss_with_hard_negatives",
    "neural_sparse_loss_with_regularization",
    "compute_sparsity_metrics",
    # Data loading
    "load_korean_news_with_dates",
    "load_multiple_korean_datasets",
    # Temporal analysis
    "calculate_temporal_idf",
    "detect_trending_tokens",
    "build_trend_boost_dict",
    # Negative sampling
    "add_hard_negatives_bm25",
    "add_mixed_negatives",
]
