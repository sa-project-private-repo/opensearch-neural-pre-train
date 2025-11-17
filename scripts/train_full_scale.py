"""Full-scale Neural Sparse pre-training script."""

import torch
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
import argparse

from src.models.neural_sparse_encoder import NeuralSparseEncoder
from src.training.losses import CombinedLoss
from src.training.data_collator import NeuralSparseDataCollator
from src.data.training_data_builder import TrainingDataBuilder
from src.training.trainer import NeuralSparseTrainer


def main(args):
    """Run full-scale training."""
    print("=" * 80)
    print("Full-Scale Neural Sparse Pre-training")
    print("=" * 80)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("\n1. Configuration:")
    print(f"   Model: {config['model']['name']}")
    print(f"   Epochs: {config['training']['num_epochs']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Learning rate: {config['training']['learning_rate']}")
    print(f"   Output dir: {config['logging']['output_dir']}")

    # Build datasets
    print("\n2. Loading datasets...")
    builder = TrainingDataBuilder()

    train_dataset, val_dataset = builder.build_training_dataset(
        qd_pairs_path=config['data']['qd_pairs_path'],
        documents_path=config['data']['documents_path'],
        synonyms_path=config['data']['synonyms_path'],
        num_negatives=config['data']['num_negatives'],
        train_split=config['data']['train_split'],
    )

    print(f"\n   Dataset summary:")
    print(f"   - Train: {len(train_dataset):,} pairs")
    print(f"   - Val: {len(val_dataset):,} pairs")
    print(f"   - Synonyms: {len(train_dataset.synonyms)} pairs")

    # Initialize model
    print("\n3. Initializing model...")
    model = NeuralSparseEncoder(
        model_name=config['model']['name'],
        max_length=config['model']['max_length'],
        use_relu=config['model']['use_relu'],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")

    # Collator
    collator = NeuralSparseDataCollator(
        tokenizer=model.tokenizer,
        query_max_length=config['data']['query_max_length'],
        doc_max_length=config['data']['doc_max_length'],
        num_negatives=config['data']['num_negatives'],
    )

    # Data loaders
    print("\n4. Creating data loaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collator,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['eval_batch_size'],
        shuffle=False,
        collate_fn=collator,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
    )

    print(f"   - Train batches: {len(train_dataloader):,}")
    print(f"   - Val batches: {len(val_dataloader):,}")

    steps_per_epoch = len(train_dataloader) // config['training']['gradient_accumulation_steps']
    total_steps = steps_per_epoch * config['training']['num_epochs']
    print(f"   - Steps per epoch: {steps_per_epoch:,}")
    print(f"   - Total steps: {total_steps:,}")

    # Loss and optimizer
    print("\n5. Setting up training...")
    loss_fn = CombinedLoss(
        alpha_ranking=config['loss']['alpha_ranking'],
        beta_cross_lingual=config['loss']['beta_cross_lingual'],
        gamma_sparsity=config['loss']['gamma_sparsity'],
        ranking_margin=config['loss']['ranking_margin'],
        use_contrastive=config['loss']['use_contrastive'],
    )

    print(f"   - Loss weights:")
    print(f"     α (ranking): {config['loss']['alpha_ranking']}")
    print(f"     β (cross-lingual): {config['loss']['beta_cross_lingual']}")
    print(f"     γ (sparsity): {config['loss']['gamma_sparsity']}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    # Learning rate scheduler
    from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

    warmup_steps = config['training']['warmup_steps']

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-7,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n6. Hardware:")
    print(f"   - Device: {device}")

    if torch.cuda.is_available():
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   - Memory: {total_memory:.1f} GB")

    # Trainer
    print("\n7. Initializing trainer...")
    trainer = NeuralSparseTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=config['training']['use_amp'] and torch.cuda.is_available(),
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        output_dir=config['logging']['output_dir'],
        save_steps=config['logging']['save_steps'],
        eval_steps=config['logging']['eval_steps'],
        logging_steps=config['logging']['logging_steps'],
    )

    # Start training
    print("\n" + "=" * 80)
    print(f"Starting full training: {config['training']['num_epochs']} epochs")
    print("=" * 80 + "\n")

    trainer.train(num_epochs=config['training']['num_epochs'])

    # Final evaluation
    print("\n" + "=" * 80)
    print("Final Evaluation")
    print("=" * 80)

    model.eval()
    val_losses = trainer.evaluate()

    print(f"\nValidation Results:")
    print(f"  Total loss: {val_losses['total_loss']:.4f}")
    print(f"  Ranking loss: {val_losses['ranking_loss']:.4f}")
    print(f"  Cross-lingual loss: {val_losses['cross_lingual_loss']:.4f}")
    print(f"  Sparsity loss: {val_losses['sparsity_loss']:.4f}")

    # Test representations
    print("\n" + "=" * 80)
    print("Testing Sparse Representations")
    print("=" * 80)

    test_queries = [
        "인공지능 모델 학습",
        "machine learning training",
        "검색 시스템 개발",
        "search system development",
        "데이터 분석",
        "data analysis",
    ]

    model.eval()
    with torch.no_grad():
        for query in test_queries:
            sparse_rep = model.encode([query], device=device)
            stats = model.get_sparsity_stats(sparse_rep)
            top_terms = model.get_top_k_terms(sparse_rep[0], k=10)

            print(f"\nQuery: {query}")
            print(f"  Non-zero terms: {stats['avg_nonzero_terms']:.0f} / {model.vocab_size}")
            print(f"  Sparsity ratio: {stats['sparsity_ratio']:.3f}")
            print(f"  L1 norm: {stats['avg_l1_norm']:.2f}")
            print(f"  Top 5 terms:")
            for i, (term, weight) in enumerate(top_terms[:5], 1):
                print(f"    {i}. {term:20s}: {weight:.4f}")

    # Save final model
    print("\n" + "=" * 80)
    print("Saving Final Model")
    print("=" * 80)

    final_model_path = Path(config['logging']['output_dir']) / 'final_model'
    model.save_pretrained(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")

    print("\n" + "=" * 80)
    print("Training Completed Successfully!")
    print("=" * 80)
    print(f"\nBest validation loss: {trainer.best_val_loss:.4f}")
    print(f"Best model: {config['logging']['output_dir']}/best_model")
    print(f"Final model: {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Full-scale Neural Sparse training')
    parser.add_argument(
        '--config',
        type=str,
        default='config/training_config.yaml',
        help='Path to training config file',
    )
    args = parser.parse_args()

    main(args)
