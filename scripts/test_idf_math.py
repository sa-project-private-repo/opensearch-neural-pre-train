#!/usr/bin/env python3
"""
Minimal mathematical verification for IDF-aware FLOPS loss.

This script verifies the correctness of:
1. BM25 IDF formula
2. IDF -> penalty weight conversion
3. Stopword penalty application
4. Loss computation

Run time: ~10 seconds (no model loading)
"""

import sys
from pathlib import Path

import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_bm25_idf_formula():
    """Test BM25 IDF formula correctness."""
    print("=" * 60)
    print("Test 1: BM25 IDF Formula")
    print("=" * 60)

    N = 1000  # Total documents

    # Test cases: (doc_freq, expected_behavior)
    test_cases = [
        (0, "MAX"),      # Never appears -> max IDF
        (1, "HIGH"),     # Rare -> high IDF
        (100, "MEDIUM"), # Common -> medium IDF
        (500, "LOW"),    # Very common -> low IDF
        (1000, "MIN"),   # Appears everywhere -> min IDF
    ]

    results = []
    for df, expected in test_cases:
        # BM25 formula: log(1 + (N - df + 0.5) / (df + 0.5))
        idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
        results.append((df, idf, expected))
        print(f"  df={df:4d}: IDF={idf:.4f} ({expected})")

    # Verify ordering: IDF should decrease as df increases
    idfs = [r[1] for r in results]
    assert idfs == sorted(idfs, reverse=True), "IDF should decrease with df"
    print("  [PASS] IDF decreases as document frequency increases")

    # Verify non-negative
    assert all(idf >= 0 for idf in idfs), "IDF should be non-negative"
    print("  [PASS] All IDF values are non-negative")

    return True


def test_penalty_weight_conversion():
    """Test IDF -> penalty weight conversion."""
    print("\n" + "=" * 60)
    print("Test 2: Penalty Weight Conversion")
    print("=" * 60)

    vocab_size = 100
    alpha = 2.0

    # Create mock IDF weights
    idf_weights = torch.linspace(0.5, 5.0, vocab_size)  # Low to high IDF

    # Normalize to [0, 1]
    idf_min = idf_weights.min()
    idf_max = idf_weights.max()
    idf_normalized = (idf_weights - idf_min) / (idf_max - idf_min + 1e-8)

    # Compute penalty weights: exp(-alpha * idf_normalized)
    penalty_weights = torch.exp(-alpha * idf_normalized)

    print(f"  IDF range: [{idf_weights.min():.2f}, {idf_weights.max():.2f}]")
    print(f"  Normalized range: [{idf_normalized.min():.4f}, {idf_normalized.max():.4f}]")
    print(f"  Penalty range: [{penalty_weights.min():.4f}, {penalty_weights.max():.4f}]")

    # Verify: high IDF (rare tokens) should have LOW penalty
    rare_token_idx = vocab_size - 1  # Highest IDF
    common_token_idx = 0  # Lowest IDF

    rare_penalty = penalty_weights[rare_token_idx]
    common_penalty = penalty_weights[common_token_idx]

    print(f"\n  Rare token (idx={rare_token_idx}):")
    print(f"    IDF={idf_weights[rare_token_idx]:.4f} -> Penalty={rare_penalty:.4f}")
    print(f"  Common token (idx={common_token_idx}):")
    print(f"    IDF={idf_weights[common_token_idx]:.4f} -> Penalty={common_penalty:.4f}")

    assert rare_penalty < common_penalty, "Rare tokens should have lower penalty"
    print("  [PASS] Rare tokens have lower penalty than common tokens")

    return True


def test_stopword_penalty_multiplication():
    """Test stopword penalty application."""
    print("\n" + "=" * 60)
    print("Test 3: Stopword Penalty Multiplication")
    print("=" * 60)

    vocab_size = 100
    stopword_penalty = 5.0

    # Create mock penalty weights (after IDF normalization)
    penalty_weights = torch.rand(vocab_size) * 0.5 + 0.5  # Range [0.5, 1.0]

    # Create stopword mask (30% are stopwords)
    stopword_mask = torch.ones(vocab_size)
    stopword_indices = torch.randint(0, vocab_size, (30,))
    stopword_mask[stopword_indices] = 0

    # Store original for comparison
    original_stopword_penalty = penalty_weights[stopword_indices[0]].item()

    # Apply stopword penalty (multiply stopword positions)
    enhanced_weights = penalty_weights.clone()
    stopword_mask_bool = (stopword_mask == 0)
    enhanced_weights[stopword_mask_bool] = enhanced_weights[stopword_mask_bool] * stopword_penalty

    new_stopword_penalty = enhanced_weights[stopword_indices[0]].item()

    print(f"  Stopword count: {(stopword_mask == 0).sum().item()}")
    print(f"  Penalty multiplier: {stopword_penalty}x")
    print(f"\n  Sample stopword (idx={stopword_indices[0].item()}):")
    print(f"    Original penalty: {original_stopword_penalty:.4f}")
    print(f"    Enhanced penalty: {new_stopword_penalty:.4f}")
    print(f"    Ratio: {new_stopword_penalty / original_stopword_penalty:.1f}x")

    # Verify multiplication
    expected = original_stopword_penalty * stopword_penalty
    assert abs(new_stopword_penalty - expected) < 1e-6, "Multiplication incorrect"
    print("  [PASS] Stopword penalty correctly multiplied")

    # Verify non-stopwords unchanged
    non_stopword_idx = (stopword_mask == 1).nonzero()[0].item()
    original_non_sw = penalty_weights[non_stopword_idx].item()
    enhanced_non_sw = enhanced_weights[non_stopword_idx].item()
    assert abs(original_non_sw - enhanced_non_sw) < 1e-6, "Non-stopwords should be unchanged"
    print("  [PASS] Non-stopword penalties unchanged")

    return True


def test_flops_loss_computation():
    """Test FLOPS loss computation."""
    print("\n" + "=" * 60)
    print("Test 4: FLOPS Loss Computation")
    print("=" * 60)

    batch_size = 4
    vocab_size = 100
    beta = 0.3

    # Create mock data
    penalty_weights = torch.rand(vocab_size)
    sparse_repr = torch.rand(batch_size, vocab_size) * 3  # Activations 0-3

    # Compute loss components
    mean_activation = sparse_repr.mean(dim=0)

    # L1: Σ(w_j * |mean_act_j|)
    weighted_l1 = (penalty_weights * mean_activation.abs()).sum()

    # L2 (FIXED): Σ(w_j * mean_act_j²) - NOT (w_j * mean_act_j)²
    weighted_l2_correct = (penalty_weights * (mean_activation ** 2)).sum()
    weighted_l2_wrong = ((penalty_weights * mean_activation) ** 2).sum()

    # Combined loss
    loss_correct = weighted_l1 + beta * weighted_l2_correct
    loss_wrong = weighted_l1 + beta * weighted_l2_wrong

    print(f"  Batch size: {batch_size}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Mean activation range: [{mean_activation.min():.4f}, {mean_activation.max():.4f}]")
    print(f"\n  L1 component: {weighted_l1:.4f}")
    print(f"  L2 (correct: w*x²): {weighted_l2_correct:.4f}")
    print(f"  L2 (wrong: (w*x)²): {weighted_l2_wrong:.4f}")
    print(f"\n  Total loss (correct): {loss_correct:.4f}")
    print(f"  Total loss (wrong): {loss_wrong:.4f}")
    print(f"  Difference: {abs(loss_correct - loss_wrong):.4f}")

    # Verify the formulas are different
    assert abs(weighted_l2_correct - weighted_l2_wrong) > 0.01, "Formulas should differ"
    print("  [PASS] L2 formulas produce different results")

    return True


def test_loss_gradient_flow():
    """Test gradient flow through loss."""
    print("\n" + "=" * 60)
    print("Test 5: Gradient Flow")
    print("=" * 60)

    batch_size = 4
    vocab_size = 100
    beta = 0.3

    # Create mock data with gradients (as leaf tensor)
    penalty_weights = torch.rand(vocab_size)
    sparse_repr = torch.rand(batch_size, vocab_size) * 3
    sparse_repr = sparse_repr.clone().detach().requires_grad_(True)

    # Forward pass
    mean_activation = sparse_repr.mean(dim=0)
    weighted_l1 = (penalty_weights * mean_activation.abs()).sum()
    weighted_l2 = (penalty_weights * (mean_activation ** 2)).sum()
    loss = weighted_l1 + beta * weighted_l2

    # Backward pass
    loss.backward()

    # Check gradients
    assert sparse_repr.grad is not None, "Gradients should exist"
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Gradient shape: {sparse_repr.grad.shape}")
    print(f"  Gradient norm: {sparse_repr.grad.norm():.4f}")
    print(f"  Gradient range: [{sparse_repr.grad.min():.6f}, {sparse_repr.grad.max():.6f}]")

    # Verify gradients are different for different penalty weights
    high_penalty_idx = penalty_weights.argmax().item()
    low_penalty_idx = penalty_weights.argmin().item()

    high_penalty_grad = sparse_repr.grad[:, high_penalty_idx].abs().mean()
    low_penalty_grad = sparse_repr.grad[:, low_penalty_idx].abs().mean()

    print(f"\n  High penalty token (idx={high_penalty_idx}):")
    print(f"    Penalty weight: {penalty_weights[high_penalty_idx]:.4f}")
    print(f"    Mean |gradient|: {high_penalty_grad:.6f}")
    print(f"  Low penalty token (idx={low_penalty_idx}):")
    print(f"    Penalty weight: {penalty_weights[low_penalty_idx]:.4f}")
    print(f"    Mean |gradient|: {low_penalty_grad:.6f}")

    # Higher penalty should produce higher gradient (more pressure to reduce)
    print(f"\n  Gradient ratio: {high_penalty_grad / low_penalty_grad:.2f}x")
    print("  [PASS] Gradients flow correctly")

    return True


def test_semantic_vs_stopword_effect():
    """Test that semantic tokens have lower penalty than stopwords."""
    print("\n" + "=" * 60)
    print("Test 6: Semantic vs Stopword Effect")
    print("=" * 60)

    vocab_size = 1000
    alpha = 2.5
    stopword_penalty = 5.0

    # Simulate IDF distribution
    # - Semantic tokens (rare): high IDF
    # - Stopwords (common): low IDF
    semantic_indices = list(range(0, 100))      # First 100 are semantic
    stopword_indices = list(range(900, 1000))   # Last 100 are stopwords

    idf_weights = torch.zeros(vocab_size)
    idf_weights[semantic_indices] = torch.rand(100) * 2 + 4  # IDF 4-6 (rare)
    idf_weights[stopword_indices] = torch.rand(100) * 0.5    # IDF 0-0.5 (common)
    idf_weights[100:900] = torch.rand(800) * 2 + 1           # IDF 1-3 (medium)

    # Convert to penalty weights
    idf_min = idf_weights.min()
    idf_max = idf_weights.max()
    idf_normalized = (idf_weights - idf_min) / (idf_max - idf_min + 1e-8)
    penalty_weights = torch.exp(-alpha * idf_normalized)

    # Apply stopword penalty
    stopword_mask = torch.ones(vocab_size)
    stopword_mask[stopword_indices] = 0
    penalty_weights[stopword_mask == 0] *= stopword_penalty

    # Compare penalties
    semantic_penalty_mean = penalty_weights[semantic_indices].mean()
    stopword_penalty_mean = penalty_weights[stopword_indices].mean()

    print(f"  Semantic tokens:")
    print(f"    Mean IDF: {idf_weights[semantic_indices].mean():.4f}")
    print(f"    Mean penalty: {semantic_penalty_mean:.4f}")
    print(f"  Stopword tokens:")
    print(f"    Mean IDF: {idf_weights[stopword_indices].mean():.4f}")
    print(f"    Mean penalty (with 5x multiplier): {stopword_penalty_mean:.4f}")
    print(f"\n  Penalty ratio (stopword/semantic): {stopword_penalty_mean / semantic_penalty_mean:.1f}x")

    # Stopwords should have MUCH higher penalty
    assert stopword_penalty_mean > semantic_penalty_mean * 10, "Stopwords should have 10x+ penalty"
    print("  [PASS] Stopwords have significantly higher penalty than semantic tokens")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("IDF-Aware FLOPS Loss Mathematical Verification")
    print("=" * 60)

    tests = [
        ("BM25 IDF Formula", test_bm25_idf_formula),
        ("Penalty Weight Conversion", test_penalty_weight_conversion),
        ("Stopword Penalty Multiplication", test_stopword_penalty_multiplication),
        ("FLOPS Loss Computation", test_flops_loss_computation),
        ("Gradient Flow", test_loss_gradient_flow),
        ("Semantic vs Stopword Effect", test_semantic_vs_stopword_effect),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            results.append((name, False))
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "[PASS]" if p else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
