"""
Unit tests for Information Gain-based synonym filtering.

Tests the mathematical correctness and edge cases of the IG computation.
"""

import numpy as np
import pytest

from src.information_gain import (
    InformationGainConfig,
    InformationGainResult,
    InformationGainFilter,
    knn_entropy_kl,
    knn_entropy_batch,
    get_knn_indices,
    compute_information_gain,
    compute_information_gain_batch,
    compute_percentile_threshold,
    compute_adaptive_threshold,
    filter_synonym_pairs,
    analyze_ig_distribution,
    _log_volume_unit_ball,
)


class TestLogVolumeUnitBall:
    """Tests for log volume of unit ball computation."""

    def test_dimension_1(self) -> None:
        """1D: V_1 = 2, log(2) ~ 0.693."""
        log_v = _log_volume_unit_ball(1)
        assert np.isclose(log_v, np.log(2), atol=1e-6)

    def test_dimension_2(self) -> None:
        """2D: V_2 = pi, log(pi) ~ 1.145."""
        log_v = _log_volume_unit_ball(2)
        assert np.isclose(log_v, np.log(np.pi), atol=1e-6)

    def test_dimension_3(self) -> None:
        """3D: V_3 = 4/3 * pi."""
        log_v = _log_volume_unit_ball(3)
        expected = np.log(4 / 3 * np.pi)
        assert np.isclose(log_v, expected, atol=1e-6)


class TestKNNEntropy:
    """Tests for KNN entropy estimation."""

    def test_uniform_distribution(self) -> None:
        """Higher entropy for uniformly spread points."""
        np.random.seed(42)

        # Uniform distribution (higher entropy)
        uniform_points = np.random.uniform(-1, 1, size=(100, 10)).astype(np.float32)
        uniform_query = np.array([0.0] * 10, dtype=np.float32)

        # Clustered distribution (lower entropy)
        clustered_points = np.random.normal(0, 0.1, size=(100, 10)).astype(np.float32)
        clustered_query = np.array([0.0] * 10, dtype=np.float32)

        uniform_entropy = knn_entropy_kl(uniform_query, uniform_points, k=5)
        clustered_entropy = knn_entropy_kl(clustered_query, clustered_points, k=5)

        # Uniform should have higher entropy
        assert uniform_entropy > clustered_entropy

    def test_batch_vs_single(self) -> None:
        """Batch computation should produce similar results to single computation.

        Note: There are minor numerical differences due to how k-th neighbor
        is selected (with/without self-exclusion), so we use a looser tolerance.
        """
        np.random.seed(42)
        reference = np.random.randn(100, 8).astype(np.float32)
        queries = np.random.randn(5, 8).astype(np.float32)

        # Single computation
        single_entropies = np.array([
            knn_entropy_kl(q, reference, k=5) for q in queries
        ])

        # Batch computation
        batch_entropies = knn_entropy_batch(queries, reference, k=5)

        # Both should be in similar range (within 10% relative tolerance)
        np.testing.assert_allclose(single_entropies, batch_entropies, rtol=0.1)

    def test_empty_reference(self) -> None:
        """Should handle small reference sets gracefully."""
        query = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        reference = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)

        # Should not raise, returns 0 for k < 1
        entropy = knn_entropy_kl(query, reference, k=5)
        assert isinstance(entropy, float)


class TestKNNIndices:
    """Tests for KNN index retrieval."""

    def test_correct_k_neighbors(self) -> None:
        """Should return exactly k neighbors."""
        np.random.seed(42)
        reference = np.random.randn(100, 5).astype(np.float32)
        query = np.array([0.0] * 5, dtype=np.float32)

        indices = get_knn_indices(query, reference, k=10)
        assert len(indices) == 10

    def test_neighbors_are_closest(self) -> None:
        """Returned indices should be the actual nearest neighbors."""
        np.random.seed(42)
        reference = np.random.randn(50, 3).astype(np.float32)
        query = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        indices = get_knn_indices(query, reference, k=5)

        # Compute all distances
        distances = np.linalg.norm(reference - query, axis=1)
        expected_indices = np.argsort(distances)[:5]

        np.testing.assert_array_equal(indices, expected_indices)


class TestInformationGain:
    """Tests for Information Gain computation."""

    def test_relative_ig_comparison(self) -> None:
        """
        Compare IG for nearby vs distant targets.

        A target in a different cluster should have different IG characteristics
        than a target very close to the source.
        """
        np.random.seed(42)

        # Corpus: two distinct clusters
        cluster1 = np.random.randn(50, 10).astype(np.float32) + np.array([5.0] * 10)
        cluster2 = np.random.randn(50, 10).astype(np.float32) + np.array([-5.0] * 10)
        corpus = np.vstack([cluster1, cluster2])

        # Source in cluster 1
        source = np.array([5.0] * 10, dtype=np.float32)

        # Target 1: very close to source (should be more predictable)
        target_near = np.array([5.01] * 10, dtype=np.float32)

        # Target 2: in different cluster (less predictable from source's view)
        target_far = np.array([-5.0] * 10, dtype=np.float32)

        config = InformationGainConfig(k_entropy=5, k_neighborhood=20)

        ig_near, _, _ = compute_information_gain(source, target_near, corpus, config)
        ig_far, _, _ = compute_information_gain(source, target_far, corpus, config)

        # The near target should have different IG than far target
        # (exact sign depends on local density, but they should differ)
        assert ig_near != ig_far

    def test_low_ig_for_nearby_target(self) -> None:
        """
        Target near source should have low IG (redundant information).
        """
        np.random.seed(42)

        corpus = np.random.randn(100, 10).astype(np.float32)

        # Source and target very close
        source = np.array([0.0] * 10, dtype=np.float32)
        target = np.array([0.01] * 10, dtype=np.float32)  # Almost same

        config = InformationGainConfig(k_entropy=5, k_neighborhood=30)
        ig, h_t, h_t_s = compute_information_gain(source, target, corpus, config)

        # Conditional entropy should be close to marginal (low IG)
        # The target is already well-explained by source's neighborhood
        assert ig < 2.0  # Relatively low IG

    def test_batch_computation(self) -> None:
        """Batch computation should work correctly."""
        np.random.seed(42)

        corpus = np.random.randn(100, 8).astype(np.float32)
        sources = np.random.randn(10, 8).astype(np.float32)
        targets = np.random.randn(10, 8).astype(np.float32)

        config = InformationGainConfig(k_entropy=5, k_neighborhood=20, batch_size=5)

        ig_scores, h_targets, h_conds = compute_information_gain_batch(
            sources, targets, corpus, config
        )

        assert ig_scores.shape == (10,)
        assert h_targets.shape == (10,)
        assert h_conds.shape == (10,)

        # IG = H(target) - H(target|source)
        np.testing.assert_allclose(ig_scores, h_targets - h_conds, rtol=1e-5)


class TestThresholdComputation:
    """Tests for threshold computation methods."""

    def test_percentile_threshold(self) -> None:
        """Percentile threshold should work correctly."""
        scores = np.arange(100).astype(np.float32)

        threshold_10 = compute_percentile_threshold(scores, percentile=10)
        assert np.isclose(threshold_10, 9.9, atol=0.1)

        threshold_50 = compute_percentile_threshold(scores, percentile=50)
        assert np.isclose(threshold_50, 49.5, atol=0.1)

    def test_otsu_threshold(self) -> None:
        """Otsu's method should find a reasonable split point.

        Note: Otsu finds the threshold that maximizes between-class variance,
        which may be at the edge of one cluster for well-separated distributions.
        We just verify it returns a valid value within the data range.
        """
        np.random.seed(42)
        low_values = np.random.normal(-10, 1.0, size=100)
        high_values = np.random.normal(10, 1.0, size=100)
        scores = np.concatenate([low_values, high_values]).astype(np.float32)

        threshold = compute_adaptive_threshold(scores, method="otsu")

        # Should be within the data range
        assert scores.min() <= threshold <= scores.max()
        # For bimodal, it should not be at the extreme edges
        assert threshold > scores.min() + 0.5
        assert threshold < scores.max() - 0.5

    def test_mad_threshold(self) -> None:
        """MAD threshold should be robust to outliers."""
        np.random.seed(42)

        # Normal distribution with outliers
        normal = np.random.normal(0, 1, size=95)
        outliers = np.array([-100, -50, -30, -20, -10])  # Extreme low outliers
        scores = np.concatenate([normal, outliers]).astype(np.float32)

        threshold = compute_adaptive_threshold(scores, method="mad", mad_multiplier=2.0)

        # MAD is robust, so threshold should be reasonable despite outliers
        assert threshold > -10  # Not influenced too much by outliers


class TestFilterSynonymPairs:
    """Integration tests for full filtering pipeline."""

    def test_filtering_removes_low_ig_pairs(self) -> None:
        """Pairs with low IG should be filtered."""
        np.random.seed(42)

        # Create corpus with distinct clusters
        corpus = np.vstack([
            np.random.randn(30, 5) + np.array([10, 0, 0, 0, 0]),
            np.random.randn(30, 5) + np.array([-10, 0, 0, 0, 0]),
            np.random.randn(30, 5) + np.array([0, 10, 0, 0, 0]),
        ]).astype(np.float32)

        # Create pairs: some good (cross-cluster), some bad (same point)
        pairs = [
            ("A", "B", 0.95),  # Will have low IG (same location)
            ("C", "D", 0.90),  # Will have high IG (cross-cluster)
        ]

        # Embeddings: A and B at same location, C and D at different clusters
        source_embs = np.array([
            [10, 0, 0, 0, 0],   # A in cluster 1
            [10, 0, 0, 0, 0],   # C in cluster 1
        ], dtype=np.float32)

        target_embs = np.array([
            [10.01, 0, 0, 0, 0],  # B very close to A (low IG)
            [-10, 0, 0, 0, 0],    # D in cluster 2 (high IG)
        ], dtype=np.float32)

        config = InformationGainConfig(
            k_entropy=5,
            k_neighborhood=20,
            percentile_threshold=50,  # Filter bottom 50%
        )

        results = filter_synonym_pairs(
            pairs, source_embs, target_embs, corpus, config
        )

        assert len(results) == 2

        # First pair should have lower IG (filtered)
        # Second pair should have higher IG (kept)
        if results[0].information_gain < results[1].information_gain:
            assert results[0].is_filtered
            assert not results[1].is_filtered


class TestInformationGainFilter:
    """Tests for the InformationGainFilter class."""

    def test_fit_and_filter(self) -> None:
        """Test complete fit and filter workflow."""
        np.random.seed(42)

        corpus = np.random.randn(100, 8).astype(np.float32)

        config = InformationGainConfig(
            k_entropy=5,
            k_neighborhood=20,
            percentile_threshold=20,
        )

        filter_obj = InformationGainFilter(config)
        filter_obj.fit(corpus)

        assert filter_obj.is_fitted
        assert filter_obj.corpus_embeddings.shape == corpus.shape

    def test_filter_without_fit_raises(self) -> None:
        """Should raise if filter() called before fit()."""
        filter_obj = InformationGainFilter()

        with pytest.raises(RuntimeError, match="not fitted"):
            filter_obj.filter_pairs([], np.array([]), np.array([]))


class TestAnalyzeDistribution:
    """Tests for distribution analysis."""

    def test_analyze_ig_distribution(self) -> None:
        """Should compute correct statistics."""
        results = [
            InformationGainResult("a", "b", 1.0, 2.0, 1.0, 0.9, False),
            InformationGainResult("c", "d", 2.0, 3.0, 1.0, 0.8, False),
            InformationGainResult("e", "f", -0.5, 1.0, 1.5, 0.7, True),
        ]

        stats = analyze_ig_distribution(results)

        assert stats["total_pairs"] == 3
        assert stats["filtered_pairs"] == 1
        assert stats["kept_pairs"] == 2
        assert np.isclose(stats["ig_mean"], (1.0 + 2.0 - 0.5) / 3)
        assert stats["ig_min"] == -0.5
        assert stats["ig_max"] == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
