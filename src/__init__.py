"""
OpenSearch Neural Sparse Training - Source Package

This package contains core modules for training neural sparse retrieval models
with temporal analysis and unsupervised synonym discovery.
"""

__version__ = "0.2.0"
__author__ = "OpenSearch Neural Sparse Team"

from src.losses import (
    in_batch_negatives_loss,
    margin_ranking_loss,
    contrastive_loss_with_hard_negatives,
)

__all__ = [
    "in_batch_negatives_loss",
    "margin_ranking_loss",
    "contrastive_loss_with_hard_negatives",
]
