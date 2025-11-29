"""
Large-scale Cross-lingual Training Script (v5)

Trains SPLADEDocExpansion model with 3.2M+ KO-EN term pairs.
Uses ExplicitNoiseTokenLoss for noise suppression.
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.splade_model import SPLADEDocExpansion, create_splade_model
from src.data.synonym_dataset import SynonymDataset, SynonymCollator
from src.training.losses import (
    TokenExpansionLoss,
    CrossLingualKDLoss,
    ExplicitNoiseTokenLoss,
)


@dataclass
class TrainingConfig:
    """Training configuration for large-scale cross-lingual training."""
    # Model
    base_model: str = "bert-base-multilingual-cased"
    expansion_mode: str = "mlm"
    teacher_model: str = "intfloat/multilingual-e5-large"

    # Data
    synonym_data: str = "dataset/large_scale/ko_en_terms_merged.jsonl"

    # Training
    batch_size: int = 128
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.05
    max_length: int = 64
    gradient_accumulation_steps: int = 2

    # Loss weights
    lambda_expansion: float = 1.0
    lambda_kd: float = 0.3
    lambda_sparsity: float = 0.001
    lambda_noise: float = 0.3

    # Loss types
    expansion_loss_type: str = "additive"
    expansion_top_k: int = 10
    kd_loss_type: str = "relation"
    noise_penalty_type: str = "sum"

    # Noise tokens
    noise_tokens: tuple = (
        "function", "operator", "operation", "operations",
        "programming", "integration", "organization",
        "implementation", "configuration", "application",
        "system", "systems", "process", "processing",
        "method", "methods", "type", "types",
        "##ing", "##tion", "##ation", "##ment",
        "the", "and", "for", "with", "from",
    )

    # Output
    output_dir: str = "outputs/cross_lingual_expansion_v5_largescale"
    log_steps: int = 100
    save_steps: int = 5000
    eval_steps: int = 10000


def train_step(
    batch: Dict[str, torch.Tensor],
    model: SPLADEDocExpansion,
    teacher,
    expansion_loss_fn: TokenExpansionLoss,
    kd_loss_fn: Optional[CrossLingualKDLoss],
    noise_loss_fn: ExplicitNoiseTokenLoss,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Single training step."""
    ko_input_ids = batch['ko_input_ids'].to(device)
    ko_attention_mask = batch['ko_attention_mask'].to(device)
    en_input_ids = batch['en_input_ids'].to(device)
    en_attention_mask = batch['en_attention_mask'].to(device)

    # Forward pass
    ko_sparse, _ = model(ko_input_ids, ko_attention_mask)

    with torch.no_grad():
        en_sparse, _ = model(en_input_ids, en_attention_mask)

    losses = {}

    # 1. Token expansion loss
    expansion_loss = expansion_loss_fn(ko_sparse, en_sparse)
    losses['expansion'] = expansion_loss.item()

    # 2. KD loss
    if teacher is not None and kd_loss_fn is not None:
        with torch.no_grad():
            ko_terms = batch['ko_terms']
            en_terms = batch['en_terms']
            ko_teacher = teacher.encode(ko_terms, convert_to_tensor=True, normalize_embeddings=True)
            en_teacher = teacher.encode(en_terms, convert_to_tensor=True, normalize_embeddings=True)

        en_sparse_grad, _ = model(en_input_ids, en_attention_mask)
        kd_loss = kd_loss_fn(ko_sparse, en_sparse_grad, ko_teacher, en_teacher)
        losses['kd'] = kd_loss.item()
    else:
        kd_loss = torch.tensor(0.0, device=device)
        losses['kd'] = 0.0

    # 3. Sparsity loss
    sparsity_loss = ko_sparse.abs().mean()
    losses['sparsity'] = sparsity_loss.item()

    # 4. Noise loss
    noise_loss = noise_loss_fn(ko_sparse)
    losses['noise'] = noise_loss.item()

    # Combined loss
    total_loss = (
        config.lambda_expansion * expansion_loss
        + config.lambda_kd * kd_loss
        + config.lambda_sparsity * sparsity_loss
        + config.lambda_noise * noise_loss
    )
    losses['total'] = total_loss.item()

    return total_loss, losses


def evaluate(
    model: SPLADEDocExpansion,
    tokenizer,
    device: torch.device,
) -> Dict[str, float]:
    """Quick evaluation on test pairs."""
    TEST_PAIRS = [
        ("머신러닝", ["machine", "learning", "ML"]),
        ("딥러닝", ["deep", "learning", "DL"]),
        ("자연어처리", ["natural", "language", "processing", "NLP"]),
        ("학습", ["training", "learning"]),
        ("모델", ["model"]),
        ("데이터", ["data"]),
        ("알고리즘", ["algorithm"]),
        ("신경망", ["neural", "network"]),
    ]

    model.eval()
    total_activated = 0
    total_expected = 0

    for ko_term, en_synonyms in TEST_PAIRS:
        encoding = tokenizer(ko_term, max_length=64, padding='max_length', truncation=True, return_tensors='pt')

        with torch.no_grad():
            sparse_rep, _ = model(encoding['input_ids'].to(device), encoding['attention_mask'].to(device))

        sparse_rep = sparse_rep[0].cpu()
        top_k_indices = torch.topk(sparse_rep, k=50).indices.tolist()
        top_k_tokens = set(tokenizer.convert_ids_to_tokens(top_k_indices))

        for en_syn in en_synonyms:
            en_tokens = tokenizer.tokenize(en_syn.lower())
            for en_tok in en_tokens:
                total_expected += 1
                if en_tok in top_k_tokens or en_tok.lower() in [t.lower() for t in top_k_tokens]:
                    total_activated += 1

    model.train()
    return {'activation_rate': total_activated / total_expected if total_expected > 0 else 0}


def main():
    """Main training function."""
    config = TrainingConfig()

    # Paths
    config.synonym_data = str(PROJECT_ROOT / config.synonym_data)
    config.output_dir = str(PROJECT_ROOT / config.output_dir)
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Save config
    with open(output_path / 'config.json', 'w') as f:
        json.dump(asdict(config), f, indent=2)

    # Model
    print(f"\nCreating model: {config.base_model}")
    model = create_splade_model(
        model_name=config.base_model,
        use_idf=False,
        use_expansion=True,
        expansion_mode=config.expansion_mode,
        dropout=0.1,
    )
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    # Teacher model
    print(f"\nLoading teacher: {config.teacher_model}")
    try:
        from sentence_transformers import SentenceTransformer
        teacher = SentenceTransformer(config.teacher_model, device=str(device), trust_remote_code=True)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        print("Teacher loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load teacher: {e}")
        teacher = None

    # Dataset
    print(f"\nLoading dataset: {config.synonym_data}")
    dataset = SynonymDataset(config.synonym_data)
    collator = SynonymCollator(tokenizer, max_length=config.max_length)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Dataset size: {len(dataset):,}")
    print(f"Num batches per epoch: {len(train_loader):,}")

    # Loss functions
    expansion_loss_fn = TokenExpansionLoss(expansion_type=config.expansion_loss_type, top_k=config.expansion_top_k)
    kd_loss_fn = CrossLingualKDLoss(loss_type=config.kd_loss_type) if teacher else None
    noise_loss_fn = ExplicitNoiseTokenLoss(
        tokenizer=tokenizer,
        noise_tokens=list(config.noise_tokens),
        lambda_noise=1.0,
        penalty_type=config.noise_penalty_type,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda')

    # Training loop
    print(f"\n{'='*60}")
    print(f"STARTING LARGE-SCALE TRAINING (v5)")
    print(f"{'='*60}")
    print(f"Total steps: {total_steps:,}")
    print(f"Warmup steps: {warmup_steps:,}")

    global_step = 0
    best_activation_rate = 0.0
    history = []

    for epoch in range(config.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{config.num_epochs} ---")

        model.train()
        epoch_losses = {'total': 0, 'expansion': 0, 'kd': 0, 'sparsity': 0, 'noise': 0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            with autocast('cuda', dtype=torch.bfloat16):
                loss, losses = train_step(
                    batch, model, teacher,
                    expansion_loss_fn, kd_loss_fn, noise_loss_fn,
                    config, device
                )

            scaler.scale(loss / config.gradient_accumulation_steps).backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % config.log_steps == 0:
                    pbar.set_postfix({
                        'loss': f"{losses['total']:.4f}",
                        'exp': f"{losses['expansion']:.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                    })

                # Save checkpoint
                if global_step % config.save_steps == 0:
                    checkpoint_path = output_path / f'checkpoint-{global_step}'
                    checkpoint_path.mkdir(exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, checkpoint_path / 'checkpoint.pt')
                    tokenizer.save_pretrained(checkpoint_path)
                    print(f"\nSaved checkpoint to {checkpoint_path}")

                # Evaluation
                if global_step % config.eval_steps == 0:
                    eval_results = evaluate(model, tokenizer, device)
                    print(f"\n[Step {global_step}] Activation rate: {eval_results['activation_rate']:.1%}")

                    if eval_results['activation_rate'] > best_activation_rate:
                        best_activation_rate = eval_results['activation_rate']
                        best_path = output_path / 'best_model'
                        best_path.mkdir(exist_ok=True)
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'activation_rate': best_activation_rate,
                        }, best_path / 'checkpoint.pt')
                        tokenizer.save_pretrained(best_path)
                        print(f"  New best model saved! (rate: {best_activation_rate:.1%})")

            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses.get(key, 0)

        # Epoch summary
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        history.append(epoch_losses)

        print(f"\nEpoch {epoch+1} Summary:")
        for key, value in epoch_losses.items():
            print(f"  {key}: {value:.4f}")

        # End of epoch evaluation
        eval_results = evaluate(model, tokenizer, device)
        print(f"  Activation rate: {eval_results['activation_rate']:.1%}")

    # Save final model
    final_path = output_path / 'final_model'
    final_path.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': asdict(config),
    }, final_path / 'checkpoint.pt')
    tokenizer.save_pretrained(final_path)

    # Save history
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best activation rate: {best_activation_rate:.1%}")
    print(f"Output directory: {output_path}")


if __name__ == "__main__":
    main()
