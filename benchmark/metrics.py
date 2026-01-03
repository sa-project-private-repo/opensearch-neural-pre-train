"""
Benchmark evaluation metrics.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of a single query."""

    query: str
    target_doc_id: str
    retrieved_doc_ids: List[str]
    latency_ms: float
    hit_rank: Optional[int] = None  # Rank where target was found (1-indexed)

    def __post_init__(self):
        """Calculate hit rank if not provided."""
        if self.hit_rank is None:
            try:
                self.hit_rank = self.retrieved_doc_ids.index(
                    self.target_doc_id
                ) + 1
            except ValueError:
                self.hit_rank = None


@dataclass
class BenchmarkMetrics:
    """Aggregated benchmark metrics."""

    method: str
    num_queries: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float


def compute_recall_at_k(results: List[QueryResult], k: int) -> float:
    """
    Compute Recall@K.

    Recall@K = # queries with target in top-K / total queries
    """
    hits = sum(
        1 for r in results
        if r.hit_rank is not None and r.hit_rank <= k
    )
    return hits / len(results) if results else 0.0


def compute_mrr(results: List[QueryResult]) -> float:
    """
    Compute Mean Reciprocal Rank.

    MRR = (1/N) * sum(1/rank_i)
    """
    reciprocal_ranks = []
    for r in results:
        if r.hit_rank is not None:
            reciprocal_ranks.append(1.0 / r.hit_rank)
        else:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def compute_ndcg_at_k(results: List[QueryResult], k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain @ K.

    For binary relevance (1 relevant doc per query):
    DCG@K = 1 / log2(rank + 1) if hit else 0
    IDCG@K = 1 / log2(2) = 1.0 (perfect case: hit at rank 1)
    nDCG@K = DCG@K / IDCG@K
    """
    ndcg_scores = []
    idcg = 1.0  # Best case: relevant doc at rank 1

    for r in results:
        if r.hit_rank is not None and r.hit_rank <= k:
            dcg = 1.0 / np.log2(r.hit_rank + 1)
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def compute_latency_percentiles(
    results: List[QueryResult],
) -> Dict[str, float]:
    """Compute latency percentiles."""
    latencies = [r.latency_ms for r in results]
    if not latencies:
        return {"p50": 0, "p95": 0, "p99": 0, "mean": 0}

    return {
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "mean": np.mean(latencies),
    }


def compute_metrics(
    method: str,
    results: List[QueryResult],
) -> BenchmarkMetrics:
    """
    Compute all benchmark metrics for a method.

    Args:
        method: Search method name
        results: List of query results

    Returns:
        BenchmarkMetrics with all computed values
    """
    latency_stats = compute_latency_percentiles(results)

    return BenchmarkMetrics(
        method=method,
        num_queries=len(results),
        recall_at_1=compute_recall_at_k(results, k=1),
        recall_at_5=compute_recall_at_k(results, k=5),
        recall_at_10=compute_recall_at_k(results, k=10),
        mrr=compute_mrr(results),
        ndcg_at_10=compute_ndcg_at_k(results, k=10),
        latency_p50_ms=latency_stats["p50"],
        latency_p95_ms=latency_stats["p95"],
        latency_p99_ms=latency_stats["p99"],
        latency_mean_ms=latency_stats["mean"],
    )


def paired_t_test(
    results_a: List[QueryResult],
    results_b: List[QueryResult],
) -> Dict[str, float]:
    """
    Perform paired t-test between two methods.

    Uses hit rank as the comparison metric.
    """
    if len(results_a) != len(results_b):
        raise ValueError("Result lists must have same length for paired test")

    # Use reciprocal rank as metric (higher is better)
    scores_a = [
        1.0 / r.hit_rank if r.hit_rank else 0.0
        for r in results_a
    ]
    scores_b = [
        1.0 / r.hit_rank if r.hit_rank else 0.0
        for r in results_b
    ]

    statistic, p_value = stats.ttest_rel(scores_a, scores_b)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def bootstrap_confidence_interval(
    results: List[QueryResult],
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        results: Query results
        metric_fn: Function to compute metric from results
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Dict with lower, upper bounds and point estimate
    """
    np.random.seed(42)
    n = len(results)
    bootstrap_values = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        sample = [results[i] for i in indices]
        bootstrap_values.append(metric_fn(sample))

    lower_pct = (1 - confidence) / 2 * 100
    upper_pct = (1 + confidence) / 2 * 100

    return {
        "point_estimate": metric_fn(results),
        "lower": np.percentile(bootstrap_values, lower_pct),
        "upper": np.percentile(bootstrap_values, upper_pct),
    }
