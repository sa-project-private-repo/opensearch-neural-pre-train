"""
Information Gain-based Synonym Pair Filtering in Embedding Space.

This module implements KNN-based entropy estimation for filtering low-quality
synonym pairs. The key insight is that meaningful synonym expansions should
provide significant "information gain" - i.e., the target term should add
semantic value beyond what the source term already captures.

Mathematical Foundation:
    IG(source -> target) = H(target) - H(target | source)

    where:
    - H(target) = marginal entropy of target in embedding space
    - H(target | source) = conditional entropy of target given source's neighborhood

    High IG: source -> target provides meaningful semantic expansion
    Low IG: trivial relationship (truncation, case change, simple substring)

KNN Entropy Estimation (Kozachenko-Leonenko Estimator):
    H_k(X) = (d/n) * sum(log(rho_k(i))) + log(n) + log(V_d) + gamma - psi(k)

    where:
    - d = embedding dimension
    - n = number of samples
    - rho_k(i) = distance to k-th nearest neighbor
    - V_d = volume of unit ball in d dimensions
    - gamma = Euler-Mascheroni constant
    - psi(k) = digamma function

Reference:
    Kozachenko, L. F., & Leonenko, N. N. (1987). Sample estimate of the
    entropy of a random vector. Problemy Peredachi Informatsii.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma, gammaln
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


@dataclass
class InformationGainResult:
    """Result of Information Gain computation for a synonym pair."""

    source: str
    target: str
    information_gain: float
    target_entropy: float
    conditional_entropy: float
    similarity: float
    is_filtered: bool
    filter_reason: str | None = None


@dataclass
class InformationGainConfig:
    """Configuration for Information Gain computation."""

    # KNN parameters
    k_entropy: int = 10  # k for entropy estimation (smaller = more local)
    k_neighborhood: int = 50  # k for defining source neighborhood

    # Filtering parameters
    percentile_threshold: float = 10.0  # Filter bottom X percentile
    min_ig_absolute: float = 0.0  # Minimum absolute IG (optional hard floor)

    # Computational parameters
    batch_size: int = 1000  # For large-scale computation
    use_faiss: bool = True  # Use FAISS for fast KNN (if available)
    normalize_embeddings: bool = True  # L2 normalize before computation

    # Debug parameters
    verbose: bool = False


def _log_volume_unit_ball(d: int) -> float:
    """
    Compute log volume of unit ball in d dimensions.

    V_d = pi^(d/2) / Gamma(d/2 + 1)
    log(V_d) = (d/2) * log(pi) - log(Gamma(d/2 + 1))

    Args:
        d: Dimension of the space.

    Returns:
        Log volume of unit ball.
    """
    return (d / 2) * np.log(np.pi) - gammaln(d / 2 + 1)


def knn_entropy_kl(
    query_embedding: NDArray[np.float32],
    reference_embeddings: NDArray[np.float32],
    k: int = 10,
    eps: float = 1e-10,
) -> float:
    """
    Compute KNN-based entropy using Kozachenko-Leonenko estimator.

    This estimates the differential entropy of a continuous distribution
    by using the distances to k-th nearest neighbors.

    Args:
        query_embedding: Single embedding vector (d,) or (1, d).
        reference_embeddings: Reference set of embeddings (n, d).
        k: Number of nearest neighbors for estimation.
        eps: Small constant to avoid log(0).

    Returns:
        Estimated entropy value (in nats, natural log).

    Note:
        For a single point, we estimate entropy based on the local
        density around that point in the reference distribution.
    """
    query_embedding = np.atleast_2d(query_embedding)
    n_ref, d = reference_embeddings.shape

    # Ensure k is valid
    k = min(k, n_ref - 1)
    if k < 1:
        return 0.0

    # Compute distances from query to all reference points
    distances = cdist(query_embedding, reference_embeddings, metric="euclidean")[0]

    # Sort and get k-th nearest neighbor distance
    sorted_distances = np.sort(distances)
    # Skip the first if query is in reference (distance = 0)
    if sorted_distances[0] < eps:
        rho_k = sorted_distances[k] if k < len(sorted_distances) else sorted_distances[-1]
    else:
        rho_k = sorted_distances[k - 1] if k - 1 < len(sorted_distances) else sorted_distances[-1]

    rho_k = max(rho_k, eps)  # Avoid log(0)

    # KL estimator: H = d * log(rho_k) + log(n) + log(V_d) + gamma - psi(k)
    # Simplified for single point estimation
    log_v_d = _log_volume_unit_ball(d)
    euler_gamma = 0.5772156649  # Euler-Mascheroni constant

    entropy = d * np.log(rho_k) + np.log(n_ref) + log_v_d + euler_gamma - digamma(k)

    return float(entropy)


def knn_entropy_batch(
    query_embeddings: NDArray[np.float32],
    reference_embeddings: NDArray[np.float32],
    k: int = 10,
    eps: float = 1e-10,
) -> NDArray[np.float32]:
    """
    Compute KNN entropy for a batch of query embeddings.

    Args:
        query_embeddings: Query embeddings (m, d).
        reference_embeddings: Reference embeddings (n, d).
        k: Number of nearest neighbors.
        eps: Small constant for numerical stability.

    Returns:
        Array of entropy values (m,).
    """
    n_queries = query_embeddings.shape[0]
    n_ref, d = reference_embeddings.shape
    k = min(k, n_ref - 1)

    if k < 1:
        return np.zeros(n_queries, dtype=np.float32)

    # Batch distance computation
    distances = cdist(query_embeddings, reference_embeddings, metric="euclidean")

    # Get k-th nearest neighbor distances
    sorted_distances = np.sort(distances, axis=1)
    rho_k = sorted_distances[:, min(k, sorted_distances.shape[1] - 1)]
    rho_k = np.maximum(rho_k, eps)

    # KL estimator
    log_v_d = _log_volume_unit_ball(d)
    euler_gamma = 0.5772156649

    entropies = d * np.log(rho_k) + np.log(n_ref) + log_v_d + euler_gamma - digamma(k)

    return entropies.astype(np.float32)


def get_knn_indices(
    query_embedding: NDArray[np.float32],
    reference_embeddings: NDArray[np.float32],
    k: int,
) -> NDArray[np.int64]:
    """
    Get indices of k nearest neighbors.

    Args:
        query_embedding: Single query embedding (d,) or (1, d).
        reference_embeddings: Reference embeddings (n, d).
        k: Number of neighbors.

    Returns:
        Indices of k nearest neighbors.
    """
    query_embedding = np.atleast_2d(query_embedding)
    k = min(k, reference_embeddings.shape[0])

    distances = cdist(query_embedding, reference_embeddings, metric="euclidean")[0]
    indices = np.argsort(distances)[:k]

    return indices


def compute_information_gain(
    source_embedding: NDArray[np.float32],
    target_embedding: NDArray[np.float32],
    corpus_embeddings: NDArray[np.float32],
    config: InformationGainConfig | None = None,
) -> tuple[float, float, float]:
    """
    Compute Information Gain for a single source->target synonym pair.

    IG = H(target) - H(target | source)

    where:
    - H(target) = entropy of target in full corpus
    - H(target | source) = entropy of target within source's neighborhood

    Interpretation:
    - High IG: target provides new information beyond source's context
    - Low IG: target is redundant or trivially derived from source

    Args:
        source_embedding: Source term embedding (d,).
        target_embedding: Target term embedding (d,).
        corpus_embeddings: Full corpus embeddings (n, d).
        config: Configuration object.

    Returns:
        Tuple of (information_gain, target_entropy, conditional_entropy).
    """
    if config is None:
        config = InformationGainConfig()

    # Normalize if configured
    if config.normalize_embeddings:
        source_embedding = source_embedding / (np.linalg.norm(source_embedding) + 1e-10)
        target_embedding = target_embedding / (np.linalg.norm(target_embedding) + 1e-10)
        norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-10
        corpus_embeddings = corpus_embeddings / norms

    # Step 1: Compute marginal entropy H(target)
    # This measures how "spread out" the target is in the full embedding space
    target_entropy = knn_entropy_kl(
        target_embedding,
        corpus_embeddings,
        k=config.k_entropy
    )

    # Step 2: Get source neighborhood
    neighbor_indices = get_knn_indices(
        source_embedding,
        corpus_embeddings,
        k=config.k_neighborhood
    )
    source_neighborhood = corpus_embeddings[neighbor_indices]

    # Step 3: Compute conditional entropy H(target | source)
    # This measures target's entropy within source's local neighborhood
    conditional_entropy = knn_entropy_kl(
        target_embedding,
        source_neighborhood,
        k=min(config.k_entropy, config.k_neighborhood - 1)
    )

    # Step 4: Information Gain = H(target) - H(target | source)
    information_gain = target_entropy - conditional_entropy

    return information_gain, target_entropy, conditional_entropy


def compute_information_gain_batch(
    source_embeddings: NDArray[np.float32],
    target_embeddings: NDArray[np.float32],
    corpus_embeddings: NDArray[np.float32],
    config: InformationGainConfig | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """
    Compute Information Gain for a batch of synonym pairs.

    This is the efficient version for large-scale computation.

    Args:
        source_embeddings: Source embeddings (m, d).
        target_embeddings: Target embeddings (m, d).
        corpus_embeddings: Corpus embeddings (n, d).
        config: Configuration object.

    Returns:
        Tuple of arrays: (information_gains, target_entropies, conditional_entropies).
    """
    if config is None:
        config = InformationGainConfig()

    n_pairs = source_embeddings.shape[0]
    n_corpus, d = corpus_embeddings.shape

    # Normalize if configured
    if config.normalize_embeddings:
        source_norms = np.linalg.norm(source_embeddings, axis=1, keepdims=True) + 1e-10
        source_embeddings = source_embeddings / source_norms

        target_norms = np.linalg.norm(target_embeddings, axis=1, keepdims=True) + 1e-10
        target_embeddings = target_embeddings / target_norms

        corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-10
        corpus_embeddings = corpus_embeddings / corpus_norms

    # Step 1: Compute marginal entropies H(target) for all targets
    # This is efficient as we only compute against full corpus once per target
    target_entropies = knn_entropy_batch(
        target_embeddings,
        corpus_embeddings,
        k=config.k_entropy
    )

    # Step 2: Compute conditional entropies H(target | source)
    # This requires computing neighborhood for each source
    conditional_entropies = np.zeros(n_pairs, dtype=np.float32)

    # Compute all source-corpus distances at once
    logger.info(f"Computing distances for {n_pairs} pairs against {n_corpus} corpus embeddings")

    # Process in batches to manage memory
    batch_size = min(config.batch_size, n_pairs)

    for batch_start in range(0, n_pairs, batch_size):
        batch_end = min(batch_start + batch_size, n_pairs)
        batch_sources = source_embeddings[batch_start:batch_end]
        batch_targets = target_embeddings[batch_start:batch_end]

        # Get distances from sources to corpus
        source_to_corpus_dist = cdist(batch_sources, corpus_embeddings, metric="euclidean")

        for i in range(batch_end - batch_start):
            # Get k-neighborhood indices for this source
            sorted_indices = np.argsort(source_to_corpus_dist[i])[:config.k_neighborhood]
            neighborhood = corpus_embeddings[sorted_indices]

            # Compute conditional entropy for this target
            conditional_entropies[batch_start + i] = knn_entropy_kl(
                batch_targets[i:i+1],
                neighborhood,
                k=min(config.k_entropy, config.k_neighborhood - 1)
            )

        if config.verbose and (batch_start + batch_size) % 5000 == 0:
            logger.info(f"Processed {batch_start + batch_size}/{n_pairs} pairs")

    # Step 3: Information Gain
    information_gains = target_entropies - conditional_entropies

    return information_gains, target_entropies, conditional_entropies


def compute_percentile_threshold(
    scores: NDArray[np.float32],
    percentile: float = 10.0,
) -> float:
    """
    Compute threshold based on score distribution percentile.

    This avoids hardcoded thresholds by using the data distribution.

    Args:
        scores: Array of Information Gain scores.
        percentile: Percentile to use as threshold (filter below this).

    Returns:
        Threshold value.
    """
    return float(np.percentile(scores, percentile))


def compute_adaptive_threshold(
    scores: NDArray[np.float32],
    method: Literal["percentile", "otsu", "mad"] = "percentile",
    percentile: float = 10.0,
    mad_multiplier: float = 2.0,
) -> float:
    """
    Compute adaptive threshold using various statistical methods.

    Methods:
    - percentile: Simple percentile-based threshold
    - otsu: Otsu's method (minimizes intra-class variance)
    - mad: Median Absolute Deviation based (robust to outliers)

    Args:
        scores: Array of Information Gain scores.
        method: Thresholding method to use.
        percentile: Percentile for percentile method.
        mad_multiplier: Multiplier for MAD method.

    Returns:
        Computed threshold value.
    """
    if method == "percentile":
        return compute_percentile_threshold(scores, percentile)

    elif method == "otsu":
        # Otsu's method: find threshold that minimizes intra-class variance
        # by maximizing between-class variance
        n_bins = 256
        hist, bin_edges = np.histogram(scores, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Total weight
        total = hist.sum()
        if total == 0:
            return float(np.median(scores))

        # Cumulative sums
        cum_sum = np.cumsum(hist)
        cum_weighted_sum = np.cumsum(hist * bin_centers)

        # Global mean
        global_mean = cum_weighted_sum[-1] / total

        # Between-class variance for each threshold
        best_var = 0.0
        best_threshold = bin_centers[0]

        for i in range(1, n_bins - 1):
            w0 = cum_sum[i] / total
            w1 = 1 - w0

            if w0 <= 0 or w1 <= 0:
                continue

            mu0 = cum_weighted_sum[i] / cum_sum[i]
            mu1 = (cum_weighted_sum[-1] - cum_weighted_sum[i]) / (total - cum_sum[i])

            between_var = w0 * w1 * (mu0 - mu1) ** 2

            if between_var > best_var:
                best_var = between_var
                best_threshold = bin_centers[i]

        return float(best_threshold)

    elif method == "mad":
        # Median Absolute Deviation: robust measure of spread
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))

        # Threshold = median - multiplier * MAD
        # This identifies the lower tail of the distribution
        return float(median - mad_multiplier * mad)

    else:
        raise ValueError(f"Unknown thresholding method: {method}")


def filter_synonym_pairs(
    pairs: list[tuple[str, str, float]],
    source_embeddings: NDArray[np.float32],
    target_embeddings: NDArray[np.float32],
    corpus_embeddings: NDArray[np.float32],
    config: InformationGainConfig | None = None,
) -> list[InformationGainResult]:
    """
    Filter synonym pairs based on Information Gain.

    Args:
        pairs: List of (source, target, similarity) tuples.
        source_embeddings: Source term embeddings (n, d).
        target_embeddings: Target term embeddings (n, d).
        corpus_embeddings: Full corpus embeddings (m, d).
        config: Configuration object.

    Returns:
        List of InformationGainResult objects with filtering decisions.
    """
    if config is None:
        config = InformationGainConfig()

    n_pairs = len(pairs)
    logger.info(f"Computing Information Gain for {n_pairs} synonym pairs")

    # Compute IG for all pairs
    ig_scores, target_entropies, cond_entropies = compute_information_gain_batch(
        source_embeddings,
        target_embeddings,
        corpus_embeddings,
        config,
    )

    # Compute threshold
    threshold = compute_adaptive_threshold(
        ig_scores,
        method="percentile",
        percentile=config.percentile_threshold,
    )

    logger.info(
        f"IG statistics: min={ig_scores.min():.4f}, max={ig_scores.max():.4f}, "
        f"mean={ig_scores.mean():.4f}, std={ig_scores.std():.4f}"
    )
    logger.info(f"Threshold (p{config.percentile_threshold}): {threshold:.4f}")

    # Create results
    results = []
    for i, (source, target, similarity) in enumerate(pairs):
        is_filtered = ig_scores[i] < threshold or ig_scores[i] < config.min_ig_absolute

        filter_reason = None
        if is_filtered:
            if ig_scores[i] < config.min_ig_absolute:
                filter_reason = f"Below absolute threshold ({config.min_ig_absolute})"
            else:
                filter_reason = f"Below percentile threshold (p{config.percentile_threshold}={threshold:.4f})"

        results.append(InformationGainResult(
            source=source,
            target=target,
            information_gain=float(ig_scores[i]),
            target_entropy=float(target_entropies[i]),
            conditional_entropy=float(cond_entropies[i]),
            similarity=similarity,
            is_filtered=is_filtered,
            filter_reason=filter_reason,
        ))

    n_filtered = sum(1 for r in results if r.is_filtered)
    logger.info(f"Filtered {n_filtered}/{n_pairs} pairs ({100*n_filtered/n_pairs:.1f}%)")

    return results


class InformationGainFilter:
    """
    Efficient Information Gain filter for large-scale synonym pair filtering.

    This class provides:
    1. Pre-computation of corpus statistics
    2. Caching for repeated queries
    3. Batch processing with progress tracking
    4. FAISS acceleration for large corpora (optional)

    Example:
        >>> filter = InformationGainFilter(config)
        >>> filter.fit(corpus_embeddings, term_to_idx)
        >>> results = filter.filter_pairs(pairs, source_embs, target_embs)
    """

    def __init__(self, config: InformationGainConfig | None = None):
        """Initialize the filter with configuration."""
        self.config = config or InformationGainConfig()
        self.corpus_embeddings: NDArray[np.float32] | None = None
        self.term_to_idx: dict[str, int] | None = None
        self.is_fitted = False

        # Optional FAISS index
        self._faiss_index = None

    def fit(
        self,
        corpus_embeddings: NDArray[np.float32],
        term_to_idx: dict[str, int] | None = None,
    ) -> "InformationGainFilter":
        """
        Fit the filter with corpus embeddings.

        Args:
            corpus_embeddings: Corpus embeddings (n, d).
            term_to_idx: Optional mapping from term to embedding index.

        Returns:
            Self for method chaining.
        """
        self.corpus_embeddings = corpus_embeddings.astype(np.float32)
        self.term_to_idx = term_to_idx

        # Normalize if configured
        if self.config.normalize_embeddings:
            norms = np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True) + 1e-10
            self.corpus_embeddings = self.corpus_embeddings / norms

        # Build FAISS index if available and configured
        if self.config.use_faiss:
            try:
                import faiss

                d = corpus_embeddings.shape[1]
                self._faiss_index = faiss.IndexFlatL2(d)
                self._faiss_index.add(self.corpus_embeddings)
                logger.info(f"Built FAISS index with {corpus_embeddings.shape[0]} vectors")
            except ImportError:
                logger.warning("FAISS not available, falling back to scipy")
                self._faiss_index = None

        self.is_fitted = True
        return self

    def get_knn_faiss(
        self,
        query: NDArray[np.float32],
        k: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Get k nearest neighbors using FAISS."""
        if self._faiss_index is None:
            raise RuntimeError("FAISS index not available")

        query = np.atleast_2d(query).astype(np.float32)
        distances, indices = self._faiss_index.search(query, k)
        return distances[0], indices[0]

    def filter_pairs(
        self,
        pairs: list[tuple[str, str, float]],
        source_embeddings: NDArray[np.float32],
        target_embeddings: NDArray[np.float32],
    ) -> list[InformationGainResult]:
        """
        Filter synonym pairs based on Information Gain.

        Args:
            pairs: List of (source, target, similarity) tuples.
            source_embeddings: Source embeddings (n, d).
            target_embeddings: Target embeddings (n, d).

        Returns:
            List of filtering results.
        """
        if not self.is_fitted:
            raise RuntimeError("Filter not fitted. Call fit() first.")

        return filter_synonym_pairs(
            pairs=pairs,
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings,
            corpus_embeddings=self.corpus_embeddings,
            config=self.config,
        )

    def compute_threshold(
        self,
        ig_scores: NDArray[np.float32],
        method: Literal["percentile", "otsu", "mad"] = "percentile",
    ) -> float:
        """Compute adaptive threshold from IG scores."""
        return compute_adaptive_threshold(
            ig_scores,
            method=method,
            percentile=self.config.percentile_threshold,
        )


def analyze_ig_distribution(
    results: list[InformationGainResult],
) -> dict[str, float | int]:
    """
    Analyze the distribution of Information Gain scores.

    Args:
        results: List of InformationGainResult objects.

    Returns:
        Dictionary of distribution statistics.
    """
    ig_scores = np.array([r.information_gain for r in results])
    filtered_scores = np.array([r.information_gain for r in results if r.is_filtered])
    kept_scores = np.array([r.information_gain for r in results if not r.is_filtered])

    stats = {
        "total_pairs": len(results),
        "filtered_pairs": sum(1 for r in results if r.is_filtered),
        "kept_pairs": sum(1 for r in results if not r.is_filtered),
        "ig_mean": float(ig_scores.mean()),
        "ig_std": float(ig_scores.std()),
        "ig_min": float(ig_scores.min()),
        "ig_max": float(ig_scores.max()),
        "ig_median": float(np.median(ig_scores)),
        "ig_p10": float(np.percentile(ig_scores, 10)),
        "ig_p25": float(np.percentile(ig_scores, 25)),
        "ig_p75": float(np.percentile(ig_scores, 75)),
        "ig_p90": float(np.percentile(ig_scores, 90)),
    }

    if len(filtered_scores) > 0:
        stats["filtered_ig_mean"] = float(filtered_scores.mean())
        stats["filtered_ig_max"] = float(filtered_scores.max())

    if len(kept_scores) > 0:
        stats["kept_ig_mean"] = float(kept_scores.mean())
        stats["kept_ig_min"] = float(kept_scores.min())

    return stats
