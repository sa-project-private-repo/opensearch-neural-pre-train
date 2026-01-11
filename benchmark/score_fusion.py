"""
Score fusion methods for hybrid search.

Implements RRF, Linear, and Learned fusion for combining
Dense (Semantic) and Sparse search results.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RankedResult:
    """A search result with rank information."""

    doc_id: str
    score: float
    rank: int


class ScoreFusion(ABC):
    """Abstract base class for score fusion methods."""

    @abstractmethod
    def fuse(
        self,
        sparse_results: List[RankedResult],
        dense_results: List[RankedResult],
    ) -> List[RankedResult]:
        """
        Fuse sparse and dense search results.

        Args:
            sparse_results: Results from neural sparse search
            dense_results: Results from dense semantic search

        Returns:
            Fused and re-ranked results
        """
        pass


class RRFFusion(ScoreFusion):
    """
    Reciprocal Rank Fusion (RRF).

    Combines rankings without requiring score normalization.
    Formula: score = sum(1 / (k + rank)) for each system

    Reference: Cormack et al., "Reciprocal Rank Fusion Outperforms
    Condorcet and Individual Rank Learning Methods" (SIGIR 2009)
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF fusion.

        Args:
            k: Rank constant (default 60, standard in literature)
        """
        self.k = k

    def fuse(
        self,
        sparse_results: List[RankedResult],
        dense_results: List[RankedResult],
    ) -> List[RankedResult]:
        """Fuse results using RRF."""
        # Build rank maps
        sparse_ranks: Dict[str, int] = {r.doc_id: r.rank for r in sparse_results}
        dense_ranks: Dict[str, int] = {r.doc_id: r.rank for r in dense_results}

        # Get all unique doc_ids
        all_docs = set(sparse_ranks.keys()) | set(dense_ranks.keys())

        # Calculate RRF scores
        # For missing docs, use a high rank (penalizes absence)
        max_rank = max(
            len(sparse_results) + 1,
            len(dense_results) + 1,
            100,  # Default penalty rank
        )

        fused_scores: Dict[str, float] = {}
        for doc_id in all_docs:
            sparse_rank = sparse_ranks.get(doc_id, max_rank)
            dense_rank = dense_ranks.get(doc_id, max_rank)

            # RRF formula
            rrf_score = 1 / (self.k + sparse_rank) + 1 / (self.k + dense_rank)
            fused_scores[doc_id] = rrf_score

        # Sort by fused score (descending)
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            RankedResult(doc_id=doc_id, score=score, rank=i + 1)
            for i, (doc_id, score) in enumerate(sorted_docs)
        ]


class LinearFusion(ScoreFusion):
    """
    Linear score combination with min-max normalization.

    Formula: score = alpha * norm(sparse_score) + (1 - alpha) * norm(dense_score)
    """

    def __init__(self, alpha: float = 0.4):
        """
        Initialize linear fusion.

        Args:
            alpha: Weight for sparse scores (0-1).
                   Dense weight = (1 - alpha).
                   Default 0.4 favors dense slightly.
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha

    def _normalize_scores(
        self, results: List[RankedResult]
    ) -> Dict[str, float]:
        """Min-max normalize scores to [0, 1]."""
        if not results:
            return {}

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)

        # Handle edge case where all scores are the same
        if max_score == min_score:
            return {r.doc_id: 1.0 for r in results}

        return {
            r.doc_id: (r.score - min_score) / (max_score - min_score)
            for r in results
        }

    def fuse(
        self,
        sparse_results: List[RankedResult],
        dense_results: List[RankedResult],
    ) -> List[RankedResult]:
        """Fuse results using linear combination."""
        # Normalize scores
        sparse_norm = self._normalize_scores(sparse_results)
        dense_norm = self._normalize_scores(dense_results)

        # Get all unique doc_ids
        all_docs = set(sparse_norm.keys()) | set(dense_norm.keys())

        # Calculate linear combination
        fused_scores: Dict[str, float] = {}
        for doc_id in all_docs:
            sparse_score = sparse_norm.get(doc_id, 0.0)
            dense_score = dense_norm.get(doc_id, 0.0)

            fused_score = self.alpha * sparse_score + (1 - self.alpha) * dense_score
            fused_scores[doc_id] = fused_score

        # Sort by fused score (descending)
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            RankedResult(doc_id=doc_id, score=score, rank=i + 1)
            for i, (doc_id, score) in enumerate(sorted_docs)
        ]


class WeightedRRFFusion(ScoreFusion):
    """
    Weighted RRF with different weights for each ranker.

    Formula: score = w1 / (k + rank_sparse) + w2 / (k + rank_dense)
    """

    def __init__(
        self,
        k: int = 60,
        sparse_weight: float = 0.4,
        dense_weight: float = 0.6,
    ):
        """
        Initialize weighted RRF.

        Args:
            k: Rank constant
            sparse_weight: Weight for sparse ranker
            dense_weight: Weight for dense ranker
        """
        self.k = k
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight

    def fuse(
        self,
        sparse_results: List[RankedResult],
        dense_results: List[RankedResult],
    ) -> List[RankedResult]:
        """Fuse results using weighted RRF."""
        sparse_ranks: Dict[str, int] = {r.doc_id: r.rank for r in sparse_results}
        dense_ranks: Dict[str, int] = {r.doc_id: r.rank for r in dense_results}

        all_docs = set(sparse_ranks.keys()) | set(dense_ranks.keys())
        max_rank = max(len(sparse_results) + 1, len(dense_results) + 1, 100)

        fused_scores: Dict[str, float] = {}
        for doc_id in all_docs:
            sparse_rank = sparse_ranks.get(doc_id, max_rank)
            dense_rank = dense_ranks.get(doc_id, max_rank)

            # Weighted RRF
            rrf_score = (
                self.sparse_weight / (self.k + sparse_rank)
                + self.dense_weight / (self.k + dense_rank)
            )
            fused_scores[doc_id] = rrf_score

        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            RankedResult(doc_id=doc_id, score=score, rank=i + 1)
            for i, (doc_id, score) in enumerate(sorted_docs)
        ]


def create_fusion_method(
    method: str,
    **kwargs,
) -> ScoreFusion:
    """
    Factory function to create fusion methods.

    Args:
        method: One of "rrf", "linear", "weighted_rrf"
        **kwargs: Method-specific parameters

    Returns:
        ScoreFusion instance
    """
    methods = {
        "rrf": RRFFusion,
        "linear": LinearFusion,
        "weighted_rrf": WeightedRRFFusion,
    }

    if method not in methods:
        raise ValueError(f"Unknown fusion method: {method}. Choose from {list(methods.keys())}")

    return methods[method](**kwargs)
