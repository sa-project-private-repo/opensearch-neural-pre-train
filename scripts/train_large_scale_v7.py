#!/usr/bin/env python3
"""
v7: Large-scale KO-EN cross-lingual training with Direct Token Target Loss.

Key improvements over v6:
1. Direct Token Target Loss - supervise English synonym tokens directly
2. Margin Loss - ensure minimum activation for target tokens
3. Negative sampling - suppress unrelated token activations

Usage:
    TOKENIZERS_PARALLELISM=false python scripts/train_large_scale_v7.py
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.splade_model import create_splade_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration for v7."""

    model_name: str = "bert-base-multilingual-cased"
    max_length: int = 64
    synonym_data: str = "dataset/large_scale/ko_en_terms_cleaned_v2.jsonl"

    batch_size: int = 128
    num_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0

    # v7 Loss weights
    lambda_target: float = 1.0
    lambda_margin: float = 0.5
    lambda_negative: float = 0.3
    lambda_sparsity: float = 0.01

    target_margin: float = 1.0

    noise_tokens: tuple = (
        "x", "nan", "1960", "[UNK]", "##rol",
        "function", "operator", "operation",
        "-", "/", "\\", "|", "_", ".", ",",
        "the", "a", "an", "is", "are", "to", "of", "and",
        "Д", "Т", "Н", "Г", "П", "В", "Ч", "С", "М", "К",
    )

    output_dir: str = "outputs/cross_lingual_expansion_v7_largescale"
    save_steps: int = 5000
    log_steps: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DirectTargetDataset(Dataset):
    """Dataset providing Korean text and English target token IDs."""

    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        logger.info(f"Loading dataset: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                ko_term = entry.get("ko_term", entry.get("ko", ""))
                en_term = entry.get("en_term", "")

                if not en_term:
                    en_terms = entry.get("en_terms", [entry.get("en_primary", "")])
                    en_term = en_terms[0] if en_terms else ""

                if ko_term and en_term:
                    en_tokens = tokenizer.tokenize(en_term.lower())
                    en_token_ids = tokenizer.convert_tokens_to_ids(en_tokens)
                    unk_id = tokenizer.unk_token_id
                    en_token_ids = [tid for tid in en_token_ids if tid != unk_id]

                    if en_token_ids:
                        self.data.append({
                            "ko_term": ko_term,
                            "en_term": en_term,
                            "en_token_ids": en_token_ids,
                        })

        logger.info(f"Loaded {len(self.data)} valid term pairs")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        ko_encoding = self.tokenizer(
            item["ko_term"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": ko_encoding["input_ids"].squeeze(0),
            "attention_mask": ko_encoding["attention_mask"].squeeze(0),
            "en_token_ids": item["en_token_ids"],
        }


def collate_fn(batch: list) -> dict:
    """Custom collate function."""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "en_token_ids": [item["en_token_ids"] for item in batch],
    }


class DirectTargetLoss(nn.Module):
    """Direct Token Target Loss for v7."""

    def __init__(self, tokenizer: AutoTokenizer, target_margin: float = 1.0, noise_tokens: tuple = ()):
        super().__init__()
        self.target_margin = target_margin
        self.noise_token_ids = set()
        for token in noise_tokens:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            self.noise_token_ids.update(token_ids)
        logger.info(f"DirectTargetLoss: {len(self.noise_token_ids)} noise tokens")

    def forward(self, sparse_rep: torch.Tensor, en_token_ids_list: list) -> tuple:
        """Compute direct target loss."""
        batch_size, vocab_size = sparse_rep.shape
        device = sparse_rep.device

        target_loss = torch.tensor(0.0, device=device)
        margin_loss = torch.tensor(0.0, device=device)
        negative_loss = torch.tensor(0.0, device=device)
        n_valid = 0

        for i, en_token_ids in enumerate(en_token_ids_list):
            if not en_token_ids:
                continue

            n_valid += 1
            rep = sparse_rep[i]
            target_ids = torch.tensor(en_token_ids, device=device)
            target_activations = rep[target_ids]

            # Target loss: maximize activation at target positions
            target_loss = target_loss - torch.log(target_activations + 1e-8).mean()

            # Margin loss: ensure minimum activation
            margin_loss = margin_loss + F.relu(self.target_margin - target_activations).mean()

            # Negative loss: suppress top non-target activations
            target_set = set(en_token_ids)
            mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
            mask[target_ids] = False
            non_target_rep = rep[mask]
            top_k = min(100, len(non_target_rep))
            top_vals, _ = torch.topk(non_target_rep, top_k)
            negative_loss = negative_loss + top_vals.mean()

        if n_valid > 0:
            target_loss = target_loss / n_valid
            margin_loss = margin_loss / n_valid
            negative_loss = negative_loss / n_valid

        return target_loss, margin_loss, negative_loss


def compute_activation_rate(model: nn.Module, tokenizer: AutoTokenizer, device: torch.device) -> float:
    """Compute English token activation rate."""
    TEST_PAIRS = [
        ("머신러닝", ["machine", "learning"]),
        ("딥러닝", ["deep", "learning"]),
        ("자연어처리", ["natural", "language", "processing"]),
        ("데이터베이스", ["database", "data"]),
        ("알고리즘", ["algorithm"]),
        ("인공지능", ["artificial", "intelligence"]),
        ("프로그래밍", ["programming"]),
        ("네트워크", ["network"]),
        ("소프트웨어", ["software"]),
        ("하드웨어", ["hardware"]),
    ]

    model.eval()
    total_activated = 0
    total_expected = 0

    with torch.no_grad():
        for ko_term, en_synonyms in TEST_PAIRS:
            encoding = tokenizer(ko_term, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
            sparse_rep, _ = model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device))
            sparse_rep = sparse_rep[0].cpu()

            top_k_indices = torch.topk(sparse_rep, k=50).indices.tolist()
            top_k_tokens = set(tokenizer.convert_ids_to_tokens(top_k_indices))

            for en_syn in en_synonyms:
                for en_tok in tokenizer.tokenize(en_syn.lower()):
                    total_expected += 1
                    if en_tok in top_k_tokens:
                        total_activated += 1

    model.train()
    return total_activated / total_expected * 100 if total_expected > 0 else 0.0


def train_v7(config: TrainingConfig):
    """Main training function for v7."""
    logger.info("=" * 60)
    logger.info("STARTING LARGE-SCALE TRAINING (v7)")
    logger.info("Direct Token Target Loss")
    logger.info("=" * 60)

    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2, default=str)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    logger.info(f"\nCreating model: {config.model_name}")

    model = create_splade_model(
        model_name=config.model_name,
        use_idf=False,
        use_expansion=True,
        expansion_mode="mlm",
    )
    model = model.to(device)

    dataset = DirectTargetDataset(config.synonym_data, tokenizer, config.max_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    logger.info(f"Dataset size: {len(dataset):,}")
    logger.info(f"Num batches per epoch: {len(dataloader):,}")

    target_loss_fn = DirectTargetLoss(tokenizer, config.target_margin, config.noise_tokens)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    total_steps = len(dataloader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    logger.info(f"\nTotal steps: {total_steps:,}")
    logger.info(f"Warmup steps: {warmup_steps:,}")

    initial_rate = compute_activation_rate(model, tokenizer, device)
    logger.info(f"\nInitial activation rate: {initial_rate:.1f}%")

    global_step = 0
    training_history = []

    for epoch in range(config.num_epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{config.num_epochs} ---")
        model.train()

        epoch_losses = {"total": 0.0, "target": 0.0, "margin": 0.0, "negative": 0.0, "sparsity": 0.0}
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            en_token_ids = batch["en_token_ids"]

            sparse_rep, _ = model(input_ids, attention_mask)

            target_loss, margin_loss, negative_loss = target_loss_fn(sparse_rep, en_token_ids)
            sparsity_loss = sparse_rep.mean()

            total_loss = (
                config.lambda_target * target_loss +
                config.lambda_margin * margin_loss +
                config.lambda_negative * negative_loss +
                config.lambda_sparsity * sparsity_loss
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_losses["total"] += total_loss.item()
            epoch_losses["target"] += target_loss.item()
            epoch_losses["margin"] += margin_loss.item()
            epoch_losses["negative"] += negative_loss.item()
            epoch_losses["sparsity"] += sparsity_loss.item()

            global_step += 1

            if global_step % config.log_steps == 0:
                progress_bar.set_postfix({
                    "loss": f"{epoch_losses['total'] / (batch_idx + 1):.4f}",
                    "tgt": f"{epoch_losses['target'] / (batch_idx + 1):.4f}",
                    "mrg": f"{epoch_losses['margin'] / (batch_idx + 1):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

            if global_step % config.save_steps == 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                checkpoint_dir.mkdir(exist_ok=True)
                act_rate = compute_activation_rate(model, tokenizer, device)
                logger.info(f"\nStep {global_step}: Activation rate = {act_rate:.1f}%")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "global_step": global_step,
                    "activation_rate": act_rate,
                }, checkpoint_dir / "checkpoint.pt")

        n_batches = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches

        training_history.append(epoch_losses)

        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Total Loss: {epoch_losses['total']:.6f}")
        logger.info(f"  Target Loss: {epoch_losses['target']:.6f}")
        logger.info(f"  Margin Loss: {epoch_losses['margin']:.6f}")

        act_rate = compute_activation_rate(model, tokenizer, device)
        logger.info(f"  Activation Rate: {act_rate:.1f}%")

    final_dir = output_dir / "final_model"
    final_dir.mkdir(exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": vars(config)}, final_dir / "checkpoint.pt")

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Final model saved to: {final_dir}")
    logger.info("=" * 60)

    final_rate = compute_activation_rate(model, tokenizer, device)
    logger.info(f"\nFinal Activation Rate: {final_rate:.1f}%")

    return model


if __name__ == "__main__":
    config = TrainingConfig()
    train_v7(config)
