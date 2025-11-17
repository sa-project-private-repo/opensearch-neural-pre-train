"""Test training pipeline components."""

import sys
sys.path.append('.')

import torch
from torch.utils.data import DataLoader

from src.models.neural_sparse_encoder import NeuralSparseEncoder
from src.training.losses import CombinedLoss
from src.training.data_collator import NeuralSparseDataCollator
from src.data.training_data_builder import TrainingDataBuilder


def test_model_initialization():
    """Test model initialization."""
    print("\n" + "=" * 80)
    print("Test 1: Model Initialization")
    print("=" * 80)

    model = NeuralSparseEncoder(
        model_name="klue/bert-base",
        max_length=128,
    )

    print(f"✓ Model initialized successfully")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Vocab size: {model.vocab_size}")

    return model


def test_loss_functions():
    """Test loss functions."""
    print("\n" + "=" * 80)
    print("Test 2: Loss Functions")
    print("=" * 80)

    # Create dummy representations
    batch_size = 4
    vocab_size = 100
    num_neg = 5

    query_rep = torch.randn(batch_size, vocab_size).relu()
    pos_rep = torch.randn(batch_size, vocab_size).relu()
    neg_reps = torch.randn(batch_size, num_neg, vocab_size).relu()

    # Test combined loss
    loss_fn = CombinedLoss(
        alpha_ranking=1.0,
        beta_cross_lingual=0.3,
        gamma_sparsity=0.001,
    )

    losses = loss_fn(query_rep, pos_rep, neg_reps)

    print(f"✓ Loss computation successful")
    print(f"  Total loss: {losses['total_loss'].item():.4f}")
    print(f"  Ranking loss: {losses['ranking_loss'].item():.4f}")
    print(f"  Sparsity loss: {losses['sparsity_loss'].item():.4f}")

    return loss_fn


def test_data_collator(model):
    """Test data collator."""
    print("\n" + "=" * 80)
    print("Test 3: Data Collator")
    print("=" * 80)

    collator = NeuralSparseDataCollator(
        tokenizer=model.tokenizer,
        query_max_length=64,
        doc_max_length=128,
        num_negatives=3,
    )

    # Create dummy features
    features = [
        {
            "query": "인공지능 모델 학습",
            "positive_doc": "인공지능 모델을 학습하는 방법입니다.",
            "negative_docs": [
                "날씨가 좋습니다.",
                "음식 레시피입니다.",
                "여행 계획입니다.",
            ],
            "korean_term": "모델",
            "english_term": "model",
        },
        {
            "query": "검색 시스템",
            "positive_doc": "검색 시스템 개발 방법입니다.",
            "negative_docs": [
                "스포츠 뉴스입니다.",
                "영화 리뷰입니다.",
                "책 추천입니다.",
            ],
        },
    ]

    batch = collator(features)

    print(f"✓ Data collation successful")
    print(f"  Query shape: {batch['query_input_ids'].shape}")
    print(f"  Positive doc shape: {batch['pos_doc_input_ids'].shape}")
    print(f"  Negative docs shape: {batch['neg_doc_input_ids'].shape}")

    return collator


def test_forward_pass(model, collator):
    """Test forward pass through model."""
    print("\n" + "=" * 80)
    print("Test 4: Forward Pass")
    print("=" * 80)

    # Create dummy batch
    features = [
        {
            "query": "machine learning model",
            "positive_doc": "This is a document about machine learning models.",
            "negative_docs": [
                "Cooking recipes are fun.",
                "Travel destinations guide.",
            ],
        }
    ]

    batch = collator(features)

    # Forward pass
    with torch.no_grad():
        query_outputs = model(
            input_ids=batch["query_input_ids"],
            attention_mask=batch["query_attention_mask"],
        )
        query_rep = query_outputs["sparse_rep"]

        pos_outputs = model(
            input_ids=batch["pos_doc_input_ids"],
            attention_mask=batch["pos_doc_attention_mask"],
        )
        pos_rep = pos_outputs["sparse_rep"]

    # Get sparsity stats
    stats = model.get_sparsity_stats(query_rep)

    print(f"✓ Forward pass successful")
    print(f"  Query representation shape: {query_rep.shape}")
    print(f"  Avg non-zero terms: {stats['avg_nonzero_terms']:.0f}")
    print(f"  Sparsity ratio: {stats['sparsity_ratio']:.3f}")

    # Get top activated terms
    top_terms = model.get_top_k_terms(query_rep[0], k=5)
    print(f"  Top 5 terms:")
    for term, weight in top_terms:
        print(f"    {term:20s}: {weight:.4f}")


def test_training_step(model, collator, loss_fn):
    """Test a single training step."""
    print("\n" + "=" * 80)
    print("Test 5: Training Step")
    print("=" * 80)

    # Create dummy batch
    features = [
        {
            "query": "neural network training",
            "positive_doc": "How to train neural networks effectively.",
            "negative_docs": [
                "Weather forecast for tomorrow.",
                "Restaurant menu options.",
            ],
            "korean_term": "학습",
            "english_term": "training",
        }
    ]

    batch = collator(features)

    # Move to device
    device = torch.device("cpu")
    model.to(device)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    # Forward pass
    query_outputs = model(
        input_ids=batch["query_input_ids"],
        attention_mask=batch["query_attention_mask"],
    )
    query_rep = query_outputs["sparse_rep"]

    pos_outputs = model(
        input_ids=batch["pos_doc_input_ids"],
        attention_mask=batch["pos_doc_attention_mask"],
    )
    pos_rep = pos_outputs["sparse_rep"]

    # Encode negatives
    batch_size, num_neg, seq_len = batch["neg_doc_input_ids"].shape
    neg_input_ids = batch["neg_doc_input_ids"].view(batch_size * num_neg, seq_len)
    neg_attention_mask = batch["neg_doc_attention_mask"].view(
        batch_size * num_neg, seq_len
    )

    neg_outputs = model(
        input_ids=neg_input_ids,
        attention_mask=neg_attention_mask,
    )
    neg_rep = neg_outputs["sparse_rep"].view(batch_size, num_neg, -1)

    # Compute loss
    losses = loss_fn(query_rep, pos_rep, neg_rep)

    print(f"✓ Training step successful")
    print(f"  Total loss: {losses['total_loss'].item():.4f}")
    print(f"  Ranking loss: {losses['ranking_loss'].item():.4f}")
    print(f"  Sparsity loss: {losses['sparsity_loss'].item():.4f}")

    # Backward pass (test gradient computation)
    losses['total_loss'].backward()

    print(f"✓ Backward pass successful")

    # Check gradients
    has_gradients = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    print(f"  Gradients computed: {has_gradients}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Testing Neural Sparse Training Pipeline")
    print("=" * 80)

    # Test 1: Model initialization
    model = test_model_initialization()

    # Test 2: Loss functions
    loss_fn = test_loss_functions()

    # Test 3: Data collator
    collator = test_data_collator(model)

    # Test 4: Forward pass
    test_forward_pass(model, collator)

    # Test 5: Training step
    test_training_step(model, collator, loss_fn)

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    print("\nTraining pipeline is ready to use.")


if __name__ == "__main__":
    main()
