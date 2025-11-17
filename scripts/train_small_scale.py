"""Small-scale training experiment script."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from src.models.neural_sparse_encoder import NeuralSparseEncoder
from src.training.losses import CombinedLoss
from src.training.data_collator import NeuralSparseDataCollator
from src.data.training_data_builder import TrainingDataBuilder
from src.training.trainer import NeuralSparseTrainer


def main():
    """Run small-scale training experiment."""
    print("=" * 80)
    print("Small-Scale Neural Sparse Training Experiment")
    print("=" * 80)

    # Load config
    with open('config/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("\n1. Loading configuration...")
    print(f"   Model: {config['model']['name']}")
    print(f"   Batch size: {config['training']['batch_size']}")

    # Build datasets
    print("\n2. Building datasets...")
    builder = TrainingDataBuilder()

    train_dataset, val_dataset = builder.build_training_dataset(
        qd_pairs_path=config['data']['qd_pairs_path'],
        documents_path=config['data']['documents_path'],
        synonyms_path=config['data']['synonyms_path'],
        num_negatives=config['data']['num_negatives'],
        train_split=config['data']['train_split'],
    )

    # Use small subset for testing (1000 samples)
    print("\n3. Creating small subset for testing...")
    small_size = min(1000, len(train_dataset))
    val_size = min(100, len(val_dataset))

    train_subset = Subset(train_dataset, range(small_size))
    val_subset = Subset(val_dataset, range(val_size))

    print(f"   Train subset: {len(train_subset)} samples")
    print(f"   Val subset: {len(val_subset)} samples")

    # Initialize model
    print("\n4. Initializing model...")
    model = NeuralSparseEncoder(
        model_name=config['model']['name'],
        max_length=config['model']['max_length'],
        use_relu=config['model']['use_relu'],
    )

    # Collator
    collator = NeuralSparseDataCollator(
        tokenizer=model.tokenizer,
        query_max_length=config['data']['query_max_length'],
        doc_max_length=config['data']['doc_max_length'],
        num_negatives=config['data']['num_negatives'],
    )

    # Data loaders
    print("\n5. Creating data loaders...")
    train_dataloader = DataLoader(
        train_subset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collator,
        num_workers=2,  # Reduce workers for small test
    )

    val_dataloader = DataLoader(
        val_subset,
        batch_size=config['evaluation']['eval_batch_size'],
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    print(f"   Train batches: {len(train_dataloader)}")
    print(f"   Val batches: {len(val_dataloader)}")

    # Loss and optimizer
    print("\n6. Setting up training...")
    loss_fn = CombinedLoss(
        alpha_ranking=config['loss']['alpha_ranking'],
        beta_cross_lingual=config['loss']['beta_cross_lingual'],
        gamma_sparsity=config['loss']['gamma_sparsity'],
        ranking_margin=config['loss']['ranking_margin'],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Trainer
    trainer = NeuralSparseTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        use_amp=config['training']['use_amp'] and torch.cuda.is_available(),
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        output_dir='outputs/small_scale_test',
        save_steps=500,
        eval_steps=100,
        logging_steps=20,
    )

    # Train for 1 epoch
    print("\n7. Starting training (1 epoch)...")
    print("=" * 80)
    trainer.train(num_epochs=1)

    # Evaluate
    print("\n8. Final evaluation...")
    val_losses = trainer.evaluate()
    print(f"\nValidation Results:")
    print(f"  Total loss: {val_losses['total_loss']:.4f}")
    print(f"  Ranking loss: {val_losses['ranking_loss']:.4f}")
    print(f"  Cross-lingual loss: {val_losses['cross_lingual_loss']:.4f}")
    print(f"  Sparsity loss: {val_losses['sparsity_loss']:.4f}")

    # Test sparse representations
    print("\n9. Testing sparse representations...")
    test_queries = [
        "인공지능 모델 학습",
        "machine learning training",
        "검색 시스템",
    ]

    model.eval()
    with torch.no_grad():
        for query in test_queries:
            sparse_rep = model.encode([query], device=device)
            stats = model.get_sparsity_stats(sparse_rep)
            top_terms = model.get_top_k_terms(sparse_rep[0], k=5)

            print(f"\nQuery: {query}")
            print(f"  Non-zero terms: {stats['avg_nonzero_terms']:.0f}")
            print(f"  Sparsity: {stats['sparsity_ratio']:.3f}")
            print(f"  Top 5 terms: {', '.join(t for t, w in top_terms)}")

    print("\n" + "=" * 80)
    print("Small-scale experiment completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
