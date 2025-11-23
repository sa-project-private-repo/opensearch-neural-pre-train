"""Test KNN synonym expansion with trained SPLADE model.

This script tests whether the model learned to expand Korean terms to their
English synonyms discovered via KNN.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer

from src.model.splade_model import SPLADEDoc


def load_model_from_checkpoint(checkpoint_path: str, model_name: str = "bert-base-multilingual-cased"):
    """Load trained SPLADE model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    # Initialize model
    model = SPLADEDoc(model_name=model_name, dropout=0.1)

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        raise ValueError("No model weights found in checkpoint")

    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded on {device}")
    print(f"✓ Checkpoint epoch: {checkpoint.get('current_epoch', 'N/A')}")
    print(f"✓ Best validation loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer, device


def get_top_k_tokens(sparse_repr: torch.Tensor, tokenizer, k: int = 30):
    """Get top-k tokens from sparse representation."""
    # Get top-k indices and values
    topk_values, topk_indices = torch.topk(sparse_repr, k=k)

    # Convert to tokens
    results = []
    for value, idx in zip(topk_values.tolist(), topk_indices.tolist()):
        token = tokenizer.convert_ids_to_tokens([idx])[0]
        results.append((token, value))

    return results


def test_synonym_expansion():
    """Test key KNN-discovered synonym pairs."""

    # Load model
    checkpoint_path = "outputs/baseline_dgx/best_model/checkpoint.pt"
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("Please ensure training has completed and model is saved.")
        return

    model, tokenizer, device = load_model_from_checkpoint(checkpoint_path)

    # Test cases: Korean terms that should expand to English equivalents
    test_cases = [
        {
            'query': '컴퓨터',
            'expected_en': ['computer', 'computing', 'pc'],
            'description': '컴퓨터 (computer)',
        },
        {
            'query': '서버',
            'expected_en': ['server', 'servers'],
            'description': '서버 (server)',
        },
        {
            'query': '모델',
            'expected_en': ['model', 'models', 'modeling'],
            'description': '모델 (model)',
        },
        {
            'query': '시스템',
            'expected_en': ['system', 'systems'],
            'description': '시스템 (system)',
        },
        {
            'query': '인터넷',
            'expected_en': ['internet', 'web'],
            'description': '인터넷 (internet)',
        },
    ]

    print("\n" + "=" * 80)
    print("Testing KNN Synonym Expansion")
    print("=" * 80)

    results_summary = []

    for i, test in enumerate(test_cases, 1):
        query = test['query']
        expected = test['expected_en']
        description = test['description']

        print(f"\n📝 Test {i}: {description}")
        print(f"   Query: {query}")
        print(f"   Expected expansions: {', '.join(expected)}")

        # Tokenize query
        inputs = tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        # Encode to sparse representation
        with torch.no_grad():
            sparse_repr, _ = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
            )

        # Get top terms
        top_terms = get_top_k_tokens(sparse_repr[0], tokenizer, k=30)

        # Check for English synonyms
        found_synonyms = []
        for term, weight in top_terms:
            term_lower = term.lower().strip('##')
            for expected_term in expected:
                if expected_term in term_lower or term_lower in expected_term:
                    found_synonyms.append((term, weight))
                    break

        # Display results
        print(f"\n   Top weighted terms:")
        for j, (term, weight) in enumerate(top_terms[:15], 1):
            marker = "✅" if any(term.lower().strip('##') in syn[0].lower() for syn in found_synonyms) else "  "
            print(f"   {marker} {j:2d}. {term:20s}: {weight:.4f}")

        # Summary
        if found_synonyms:
            print(f"\n   ✅ SUCCESS: Found {len(found_synonyms)} synonym(s)")
            for term, weight in found_synonyms:
                print(f"      → {term} (weight: {weight:.4f})")
            results_summary.append((description, True, found_synonyms))
        else:
            print(f"\n   ❌ FAILED: No English synonyms found in top terms")
            results_summary.append((description, False, []))

    # Final summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    success_count = sum(1 for _, success, _ in results_summary if success)
    total_count = len(results_summary)

    print(f"\nPassed: {success_count}/{total_count} tests")

    for desc, success, synonyms in results_summary:
        status = "✅" if success else "❌"
        synonym_str = ", ".join([f"{term} ({weight:.3f})" for term, weight in synonyms]) if synonyms else "None"
        print(f"{status} {desc}: {synonym_str}")

    if success_count == total_count:
        print("\n🎉 All tests passed! KNN synonym expansion is working correctly.")
    elif success_count > 0:
        print(f"\n⚠️  Partial success: {success_count}/{total_count} tests passed.")
        print("   Some synonym pairs may need more training data or higher quality co-occurrence samples.")
    else:
        print("\n❌ All tests failed. Model may need more training or better synonym data.")


if __name__ == "__main__":
    test_synonym_expansion()
