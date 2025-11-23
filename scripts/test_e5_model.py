"""Test multilingual-e5-large model locally.

This script tests:
1. Model loading
2. Embedding generation
3. Cosine similarity calculation
4. Cross-lingual semantic matching
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Average pooling for E5 models.

    Args:
        last_hidden_states: Model outputs [batch_size, seq_len, hidden_dim]
        attention_mask: Attention mask [batch_size, seq_len]

    Returns:
        Pooled embeddings [batch_size, hidden_dim]
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_embeddings(texts: List[str], model, tokenizer, device: str) -> np.ndarray:
    """
    Generate embeddings for input texts.

    Args:
        texts: List of input texts
        model: E5 model
        tokenizer: E5 tokenizer
        device: Device to use

    Returns:
        Embeddings array [len(texts), hidden_dim]
    """
    # Add E5 instruction prefix
    texts_with_prefix = ["query: " + text for text in texts]

    # Tokenize
    inputs = tokenizer(
        texts_with_prefix,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])

    # Normalize for cosine similarity
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    """Test E5 model with Korean-English pairs."""

    print("=" * 80)
    print("Testing multilingual-e5-large Model")
    print("=" * 80)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Using device: {device}")

    # Load model
    print("\nLoading model: intfloat/multilingual-e5-large...")
    model_name = "intfloat/multilingual-e5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    print("✓ Model loaded successfully")

    # Test cases: Korean-English synonym pairs
    test_cases = [
        # AI/ML terms
        ("인공지능", "artificial intelligence"),
        ("인공지능", "AI"),
        ("기계학습", "machine learning"),
        ("딥러닝", "deep learning"),
        ("신경망", "neural network"),

        # Computer science terms
        ("컴퓨터", "computer"),
        ("인터넷", "internet"),
        ("서버", "server"),
        ("데이터베이스", "database"),

        # Unrelated pairs (should have low similarity)
        ("기계학습", "restaurant"),
        ("인공지능", "apple"),
    ]

    print("\n" + "=" * 80)
    print("Korean-English Semantic Similarity Tests")
    print("=" * 80)

    # Process all texts
    korean_texts = [pair[0] for pair in test_cases]
    english_texts = [pair[1] for pair in test_cases]

    print("\nGenerating embeddings...")
    korean_embeddings = get_embeddings(korean_texts, model, tokenizer, device)
    english_embeddings = get_embeddings(english_texts, model, tokenizer, device)
    print(f"✓ Generated embeddings: Korean {korean_embeddings.shape}, English {english_embeddings.shape}")

    # Calculate similarities
    print("\nResults:")
    print("-" * 80)
    print(f"{'Korean':<20} {'English':<25} {'Similarity':>10} {'Status':>10}")
    print("-" * 80)

    for i, (korean, english) in enumerate(test_cases):
        similarity = cosine_similarity(korean_embeddings[i], english_embeddings[i])

        # Status based on similarity
        if similarity >= 0.8:
            status = "✅ HIGH"
        elif similarity >= 0.6:
            status = "⚠️  MEDIUM"
        else:
            status = "❌ LOW"

        print(f"{korean:<20} {english:<25} {similarity:>10.4f} {status:>10}")

    print("-" * 80)

    # Additional test: Find nearest Korean term for English query
    print("\n" + "=" * 80)
    print("Cross-lingual Search Test")
    print("=" * 80)

    query = "machine learning"
    korean_candidates = ["기계학습", "인공지능", "컴퓨터", "딥러닝", "신경망", "인터넷"]

    print(f"\nEnglish Query: '{query}'")
    print(f"Korean Candidates: {korean_candidates}")

    query_embedding = get_embeddings([query], model, tokenizer, device)[0]
    candidate_embeddings = get_embeddings(korean_candidates, model, tokenizer, device)

    similarities = [cosine_similarity(query_embedding, emb) for emb in candidate_embeddings]
    ranked = sorted(zip(korean_candidates, similarities), key=lambda x: x[1], reverse=True)

    print("\nRanked Results:")
    print("-" * 80)
    for rank, (term, sim) in enumerate(ranked, 1):
        print(f"{rank}. {term:<20} (similarity: {sim:.4f})")
    print("-" * 80)

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"✓ Model: {model_name}")
    print(f"✓ Embedding dimension: {korean_embeddings.shape[1]}")
    print(f"✓ Device: {device}")
    print(f"✓ All tests completed successfully")
    print("=" * 80)


if __name__ == "__main__":
    main()
