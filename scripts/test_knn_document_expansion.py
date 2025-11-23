"""Test KNN synonym expansion in document encoding (SPLADE-doc architecture).

SPLADE-doc is an inference-free model:
- Query: bag-of-words (no model encoding)
- Document: model encoding with term expansion

This script tests whether the model expands Korean terms to their English
synonyms when encoding documents.
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


def get_top_k_tokens(sparse_repr: torch.Tensor, tokenizer, k: int = 50):
    """Get top-k tokens from sparse representation."""
    # Get top-k indices and values
    topk_values, topk_indices = torch.topk(sparse_repr, k=k)

    # Convert to tokens
    results = []
    for value, idx in zip(topk_values.tolist(), topk_indices.tolist()):
        token = tokenizer.convert_ids_to_tokens([idx])[0]
        if not token.startswith('[') or token in ['[MASK]']:
            results.append((token, value))

    return results


def test_document_synonym_expansion():
    """Test synonym expansion in document encoding."""

    # Load model
    checkpoint_path = "outputs/baseline_dgx/best_model/checkpoint.pt"
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("Please ensure training has completed and model is saved.")
        return

    model, tokenizer, device = load_model_from_checkpoint(checkpoint_path)

    # Test documents with Korean-English synonym pairs
    test_documents = [
        {
            'document': '컴퓨터는 현대 사회에서 필수적인 도구입니다. Computer technology has advanced rapidly.',
            'korean_term': '컴퓨터',
            'english_term': 'computer',
            'description': '컴퓨터 ↔ computer',
        },
        {
            'document': '서버는 네트워크의 핵심입니다. Server infrastructure is critical for modern applications.',
            'korean_term': '서버',
            'english_term': 'server',
            'description': '서버 ↔ server',
        },
        {
            'document': '기계학습 모델은 데이터를 분석합니다. Machine learning model processes information efficiently.',
            'korean_term': '모델',
            'english_term': 'model',
            'description': '모델 ↔ model',
        },
    ]

    print("\n" + "=" * 80)
    print("Testing Document-Side Synonym Expansion (SPLADE-doc)")
    print("=" * 80)
    print("\nNote: SPLADE-doc expands terms during DOCUMENT encoding (not query)")
    print("Testing if Korean terms in documents are expanded to English synonyms\n")

    results_summary = []

    for i, test in enumerate(test_documents, 1):
        document = test['document']
        korean_term = test['korean_term']
        english_term = test['english_term']
        description = test['description']

        print(f"\n📝 Test {i}: {description}")
        print(f"   Document: {document[:100]}...")
        print(f"   Looking for: {korean_term} → {english_term}")

        # Tokenize document
        inputs = tokenizer(
            document,
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
        top_terms = get_top_k_tokens(sparse_repr[0], tokenizer, k=50)

        # Find Korean and English terms
        korean_found = []
        english_found = []

        for term, weight in top_terms:
            term_lower = term.lower().strip('##')
            if korean_term in term or any(char in term for char in korean_term):
                korean_found.append((term, weight))
            if english_term in term_lower or term_lower in english_term:
                english_found.append((term, weight))

        # Display results
        print(f"\n   Top weighted terms (non-special tokens):")
        for j, (term, weight) in enumerate(top_terms[:20], 1):
            marker = ""
            if any(korean_term in t[0] for t in korean_found) and any(term in t[0] for t in korean_found):
                marker = "🇰🇷"
            elif any(term.lower().strip('##') in t[0].lower() for t in english_found):
                marker = "🇺🇸"
            print(f"   {marker:3s} {j:2d}. {term:25s}: {weight:.4f}")

        # Summary
        print(f"\n   Korean term '{korean_term}':")
        if korean_found:
            for term, weight in korean_found[:3]:
                print(f"      ✅ {term}: {weight:.4f}")
        else:
            print(f"      ❌ Not found in top terms")

        print(f"\n   English synonym '{english_term}':")
        if english_found:
            for term, weight in english_found[:3]:
                print(f"      ✅ {term}: {weight:.4f}")
            success = True
        else:
            print(f"      ❌ Not found in top terms")
            success = False

        results_summary.append((description, success, english_found))

    # Final summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    success_count = sum(1 for _, success, _ in results_summary if success)
    total_count = len(results_summary)

    print(f"\nPassed: {success_count}/{total_count} tests")

    for desc, success, synonyms in results_summary:
        status = "✅" if success else "❌"
        synonym_str = ", ".join([f"{term} ({weight:.3f})" for term, weight in synonyms[:2]]) if synonyms else "None"
        print(f"{status} {desc}: {synonym_str}")

    if success_count == total_count:
        print("\n🎉 All tests passed! KNN synonym expansion is working correctly.")
        print("   Documents with Korean terms are being expanded to include English synonyms.")
    elif success_count > 0:
        print(f"\n⚠️  Partial success: {success_count}/{total_count} tests passed.")
        print("   Some synonym pairs are working, but others may need more training.")
    else:
        print("\n❌ All tests failed.")
        print("   Possible causes:")
        print("   1. Training data may not have sufficient synonym co-occurrence")
        print("   2. Model may need more epochs or better hyperparameters")
        print("   3. Loss function may need adjustment for cross-lingual learning")


if __name__ == "__main__":
    test_document_synonym_expansion()
