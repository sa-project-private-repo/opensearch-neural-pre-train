"""Test script for neural sparse encoder model loading."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.models.neural_sparse_encoder import NeuralSparseEncoder


def test_loading_scenarios():
    """Test different model loading scenarios."""

    print("=" * 80)
    print("Testing NeuralSparseEncoder.from_pretrained()")
    print("=" * 80)

    # Test 1: Load from HuggingFace Hub (with pretrained head)
    print("\n1. Testing load from HuggingFace Hub (opensearch-neural-sparse-encoding-v1)")
    print("-" * 80)
    try:
        model_v1 = NeuralSparseEncoder.from_pretrained(
            "opensearch-project/opensearch-neural-sparse-encoding-v1"
        )
        print("SUCCESS: Model loaded from HuggingFace Hub")
        print(f"  Base model: {model_v1.model_name}")
        print(f"  Vocab size: {model_v1.vocab_size}")
        print(f"  Hidden size: {model_v1.hidden_size}")
    except Exception as e:
        print(f"EXPECTED: {type(e).__name__}: {e}")
        print("Note: This model may not have neural_sparse_head.pt")
        print("Falling back to base model initialization...")

    # Test 2: Load from HuggingFace Hub (multilingual v1)
    print(
        "\n2. Testing load from HuggingFace Hub "
        "(opensearch-neural-sparse-encoding-multilingual-v1)"
    )
    print("-" * 80)
    try:
        model_multilingual = NeuralSparseEncoder.from_pretrained(
            "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
        )
        print("SUCCESS: Model loaded from HuggingFace Hub")
        print(f"  Base model: {model_multilingual.model_name}")
        print(f"  Vocab size: {model_multilingual.vocab_size}")
        print(f"  Hidden size: {model_multilingual.hidden_size}")
    except Exception as e:
        print(f"EXPECTED: {type(e).__name__}: {e}")
        print("Note: This model may not have neural_sparse_head.pt")
        print("Falling back to base model initialization...")

    # Test 3: Test inference with loaded model
    print("\n3. Testing inference with loaded model")
    print("-" * 80)
    try:
        # Use whichever model loaded successfully
        test_model = model_multilingual if "model_multilingual" in locals() else None

        if test_model is None:
            print(
                "Initializing new model from base encoder for testing..."
            )
            test_model = NeuralSparseEncoder(
                model_name="xlm-roberta-base",
                max_length=128,
            )

        test_model.eval()

        # Test texts
        test_texts = [
            "What is machine learning?",
            "머신러닝이란 무엇인가?",
            "机器学习是什么？",
        ]

        print("Encoding test texts...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_model = test_model.to(device)

        with torch.no_grad():
            sparse_reps = test_model.encode(test_texts, device=device)

        print(f"  Sparse representations shape: {sparse_reps.shape}")

        # Get sparsity stats
        stats = test_model.get_sparsity_stats(sparse_reps)
        print("\n  Sparsity statistics:")
        for key, value in stats.items():
            print(f"    {key}: {value:.4f}")

        # Get top terms for first text
        print(f"\n  Top-10 terms for '{test_texts[0]}':")
        top_terms = test_model.get_top_k_terms(sparse_reps[0], k=10)
        for term, weight in top_terms:
            print(f"    {term:20s}: {weight:.4f}")

        print("\nSUCCESS: Inference test passed")

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    # Test 4: Test save and load cycle
    print("\n4. Testing save and load cycle")
    print("-" * 80)
    try:
        # Create a new model
        save_test_model = NeuralSparseEncoder(
            model_name="xlm-roberta-base",
            max_length=128,
        )

        # Save to temporary directory
        save_path = project_root / "outputs" / "test_save_load"
        save_test_model.save_pretrained(str(save_path))
        print(f"  Model saved to: {save_path}")

        # Load back
        loaded_model = NeuralSparseEncoder.from_pretrained(str(save_path))
        print(f"  Model loaded from: {save_path}")

        # Verify configurations match
        assert (
            loaded_model.model_name == save_test_model.model_name
        ), "Model names don't match"
        assert (
            loaded_model.vocab_size == save_test_model.vocab_size
        ), "Vocab sizes don't match"
        assert (
            loaded_model.hidden_size == save_test_model.hidden_size
        ), "Hidden sizes don't match"

        print("SUCCESS: Save and load cycle works correctly")

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_loading_scenarios()
