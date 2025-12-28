"""
Test script for korean-neural-sparse-encoder-v1 model.

This script tests the model loading and inference from the huggingface folder.
"""

import sys
from pathlib import Path

# Add huggingface folder to path for custom model
PROJECT_ROOT = Path(__file__).parent.parent
HUGGINGFACE_DIR = PROJECT_ROOT / "huggingface"
sys.path.insert(0, str(HUGGINGFACE_DIR))

import torch
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_file

from modeling_splade import SPLADEModel


def test_model_loading() -> bool:
    """Test model loading from huggingface folder."""
    print("=" * 60)
    print("Test 1: Model Loading")
    print("=" * 60)

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(HUGGINGFACE_DIR))
        print(f"[OK] Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

        # Load config
        config = AutoConfig.from_pretrained(str(HUGGINGFACE_DIR))
        print(f"[OK] Config loaded: model_type={config.model_type}")

        # Create model
        model = SPLADEModel(config)
        print(f"[OK] Model created: {sum(p.numel() for p in model.parameters()):,} params")

        # Load weights
        state_dict = load_file(str(HUGGINGFACE_DIR / "model.safetensors"))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if len(missing) == 0 and len(unexpected) == 0:
            print("[OK] All weights loaded successfully")
        else:
            print(f"[WARN] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        return True

    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        return False


def test_inference() -> bool:
    """Test model inference with sample queries."""
    print("\n" + "=" * 60)
    print("Test 2: Model Inference")
    print("=" * 60)

    try:
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(str(HUGGINGFACE_DIR))
        config = AutoConfig.from_pretrained(str(HUGGINGFACE_DIR))
        model = SPLADEModel(config)
        state_dict = load_file(str(HUGGINGFACE_DIR / "model.safetensors"))
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Test queries
        test_queries = [
            "손해배상",
            "진단",
            "검색",
            "인공지능",
            "계약",
            "치료",
        ]

        print("\nInference results:")
        for query in test_queries:
            inputs = tokenizer(
                query, return_tensors="pt", padding=True, truncation=True
            )

            with torch.no_grad():
                sparse_repr, token_weights = model(**inputs)

            # Get top tokens
            top_k = 5
            top_values, top_indices = sparse_repr[0].topk(top_k)
            tokens = [
                tokenizer.decode([idx]).strip()
                for idx in top_indices.tolist()
            ]
            values = top_values.tolist()

            top_str = ", ".join(
                f"{t}({v:.2f})" for t, v in zip(tokens, values) if v > 0
            )
            print(f"  '{query}' -> {top_str}")

        print("\n[OK] Inference test passed")
        return True

    except Exception as e:
        print(f"[FAIL] Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sparse_representation() -> bool:
    """Test sparse representation properties."""
    print("\n" + "=" * 60)
    print("Test 3: Sparse Representation Properties")
    print("=" * 60)

    try:
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(str(HUGGINGFACE_DIR))
        config = AutoConfig.from_pretrained(str(HUGGINGFACE_DIR))
        model = SPLADEModel(config)
        state_dict = load_file(str(HUGGINGFACE_DIR / "model.safetensors"))
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Test query
        query = "손해배상 청구"
        inputs = tokenizer(
            query, return_tensors="pt", padding=True, truncation=True
        )

        with torch.no_grad():
            sparse_repr, token_weights = model(**inputs)

        # Check properties
        vocab_size = sparse_repr.shape[1]
        non_zero = (sparse_repr[0] > 0).sum().item()
        sparsity = 1 - (non_zero / vocab_size)
        max_val = sparse_repr.max().item()
        mean_val = sparse_repr[sparse_repr > 0].mean().item()

        print(f"  Vocab size: {vocab_size:,}")
        print(f"  Non-zero elements: {non_zero:,}")
        print(f"  Sparsity: {sparsity:.4f} ({sparsity * 100:.2f}%)")
        print(f"  Max value: {max_val:.4f}")
        print(f"  Mean (non-zero): {mean_val:.4f}")

        # Validate sparsity (should be high for SPLADE)
        if sparsity > 0.9:
            print(f"\n[OK] Sparsity is good (>90%)")
        else:
            print(f"\n[WARN] Sparsity is low ({sparsity:.2%})")

        return True

    except Exception as e:
        print(f"[FAIL] Sparse representation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synonym_expansion() -> bool:
    """Test synonym expansion capability."""
    print("\n" + "=" * 60)
    print("Test 4: Synonym Expansion")
    print("=" * 60)

    try:
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(str(HUGGINGFACE_DIR))
        config = AutoConfig.from_pretrained(str(HUGGINGFACE_DIR))
        model = SPLADEModel(config)
        state_dict = load_file(str(HUGGINGFACE_DIR / "model.safetensors"))
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Test cases with expected synonyms
        test_cases = [
            ("손해배상", ["손해", "배상", "보상", "피해"]),
            ("인공지능", ["AI", "지능", "인공"]),
            ("진단", ["검진", "검사", "진료"]),
            ("계약", ["약정", "합의", "협정"]),
            ("치료", ["처치", "시술", "진료"]),
        ]

        passed = 0
        total = len(test_cases)

        for query, expected_synonyms in test_cases:
            inputs = tokenizer(
                query, return_tensors="pt", padding=True, truncation=True
            )

            with torch.no_grad():
                sparse_repr, _ = model(**inputs)

            # Get top 20 tokens
            top_values, top_indices = sparse_repr[0].topk(20)
            top_tokens = [
                tokenizer.decode([idx]).strip().lower()
                for idx in top_indices.tolist()
            ]

            # Check if any expected synonym is in top tokens
            found = []
            for syn in expected_synonyms:
                if any(syn.lower() in t or t in syn.lower() for t in top_tokens):
                    found.append(syn)

            if len(found) > 0:
                passed += 1
                status = "[OK]"
            else:
                status = "[FAIL]"

            print(f"  {status} '{query}' -> found: {found}")

        print(f"\n  Result: {passed}/{total} passed")

        if passed >= total * 0.8:
            print("[OK] Synonym expansion test passed")
            return True
        else:
            print("[WARN] Some synonym expansion tests failed")
            return False

    except Exception as e:
        print(f"[FAIL] Synonym expansion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_inference() -> bool:
    """Test batch inference."""
    print("\n" + "=" * 60)
    print("Test 5: Batch Inference")
    print("=" * 60)

    try:
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(str(HUGGINGFACE_DIR))
        config = AutoConfig.from_pretrained(str(HUGGINGFACE_DIR))
        model = SPLADEModel(config)
        state_dict = load_file(str(HUGGINGFACE_DIR / "model.safetensors"))
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Batch of queries
        queries = ["손해배상", "인공지능", "진단", "계약"]

        inputs = tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )

        with torch.no_grad():
            sparse_repr, token_weights = model(**inputs)

        print(f"  Batch size: {len(queries)}")
        print(f"  Output shape: {sparse_repr.shape}")
        print(f"  Token weights shape: {token_weights.shape}")

        # Verify output shape
        assert sparse_repr.shape[0] == len(queries), "Batch size mismatch"
        # Model vocab_size (50000) may differ from tokenizer.vocab_size (49999)
        assert sparse_repr.shape[1] == config.vocab_size, "Vocab size mismatch"

        print("\n[OK] Batch inference test passed")
        return True

    except Exception as e:
        print(f"[FAIL] Batch inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Korean Neural Sparse Encoder v1 - Test Suite")
    print("=" * 60)
    print(f"Huggingface dir: {HUGGINGFACE_DIR}")

    results = []

    # Run tests
    results.append(("Model Loading", test_model_loading()))
    results.append(("Inference", test_inference()))
    results.append(("Sparse Representation", test_sparse_representation()))
    results.append(("Synonym Expansion", test_synonym_expansion()))
    results.append(("Batch Inference", test_batch_inference()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
