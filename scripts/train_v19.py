#!/usr/bin/env python3
"""
v19 Training Script: XLM-RoBERTa-large with High-Quality Data Only

Based on v17/v18 learnings:
- Use only high-quality MUSE data (exclude wikidata noise)
- Same conservative hyperparameters as v17 (proven to work)
- Dataset: 18K pairs (quality over quantity)
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.model.splade_model import create_splade_model


# Configuration - based on v17 success
CONFIG = {
    # Model
    "model_name": "xlm-roberta-large",
    "max_length": 64,

    # Data - high-quality only (no wikidata)
    "data_path": PROJECT_ROOT / "dataset" / "v19_high_quality" / "term_pairs.jsonl",

    # Training - same as v17
    "batch_size": 32,
    "gradient_accumulation_steps": 4,
    "num_epochs": 10,  # Same as v17
    "learning_rate": 2e-6,
    "warmup_ratio": 0.2,
    "max_grad_norm": 1.0,

    # Loss weights - same as v17
    "lambda_self": 2.0,
    "lambda_target": 5.0,
    "lambda_margin": 3.0,
    "lambda_negative": 0.5,
    "lambda_sparsity": 0.005,
    "target_margin": 2.0,

    # Mixed precision
    "use_fp16": True,

    # Output
    "output_dir": PROJECT_ROOT / "outputs" / "v19_xlm_large",
}


def is_korean_char(c: str) -> bool:
    """Check if character is Korean."""
    return (
        "\uac00" <= c <= "\ud7a3"
        or "\u1100" <= c <= "\u11ff"
        or "\u3130" <= c <= "\u318f"
    )


def is_english_char(c: str) -> bool:
    """Check if character is English."""
    return c.isalpha() and c.isascii()


def is_non_target_token(token: str) -> bool:
    """Check if token is from non-target language."""
    clean = token.replace("▁", "").replace("##", "")
    if not clean:
        return False

    has_korean = any(is_korean_char(c) for c in clean)
    has_english = any(is_english_char(c) for c in clean)

    if has_korean or has_english:
        return False

    has_japanese = any(
        "\u3040" <= c <= "\u309f" or "\u30a0" <= c <= "\u30ff" for c in clean
    )
    has_cjk = any("\u4e00" <= c <= "\u9fff" for c in clean)
    has_cyrillic = any("\u0400" <= c <= "\u04ff" for c in clean)
    has_arabic = any("\u0600" <= c <= "\u06ff" for c in clean)
    has_thai = any("\u0e00" <= c <= "\u0e7f" for c in clean)
    has_greek = any("\u0370" <= c <= "\u03ff" for c in clean)

    return (
        has_japanese or has_cjk or has_cyrillic or has_arabic or has_thai or has_greek
    )


class TermPairDataset(Dataset):
    """Dataset for Korean-English term pairs."""

    def __init__(self, data_path: Path, tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        print(f"Loading dataset from {data_path}...")

        with open(data_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading data"):
                item = json.loads(line.strip())

                ko_term = item.get("ko", "")
                en_term = item.get("en", "")

                if not ko_term or not en_term:
                    continue

                ko_tokens = tokenizer.tokenize(ko_term)
                ko_token_ids = tokenizer.convert_tokens_to_ids(ko_tokens)
                ko_token_ids = [
                    tid for tid in ko_token_ids if tid != tokenizer.unk_token_id
                ]

                en_tokens = tokenizer.tokenize(en_term.lower())
                en_token_ids = tokenizer.convert_tokens_to_ids(en_tokens)
                en_token_ids = [
                    tid for tid in en_token_ids if tid != tokenizer.unk_token_id
                ]

                if ko_token_ids and en_token_ids:
                    self.data.append(
                        {
                            "ko_term": ko_term,
                            "en_term": en_term,
                            "ko_token_ids": ko_token_ids,
                            "en_token_ids": en_token_ids,
                        }
                    )

        print(f"Loaded {len(self.data):,} valid term pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        encoding = self.tokenizer(
            item["ko_term"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "ko_token_ids": item["ko_token_ids"],
            "en_token_ids": item["en_token_ids"],
        }


def collate_fn(batch):
    """Custom collate function."""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "ko_token_ids": [item["ko_token_ids"] for item in batch],
        "en_token_ids": [item["en_token_ids"] for item in batch],
    }


class TermLevelLoss(nn.Module):
    """Loss function for term-level training."""

    def __init__(self, target_margin: float = 2.0, non_target_ids: torch.Tensor = None):
        super().__init__()
        self.target_margin = target_margin
        self.non_target_ids = non_target_ids

    def forward(self, sparse_rep, ko_token_ids, en_token_ids):
        batch_size = sparse_rep.shape[0]
        device = sparse_rep.device

        self_loss = torch.tensor(0.0, device=device)
        target_loss = torch.tensor(0.0, device=device)
        margin_loss = torch.tensor(0.0, device=device)
        negative_loss = torch.tensor(0.0, device=device)

        n_valid = 0

        for i in range(batch_size):
            rep = sparse_rep[i]

            if ko_token_ids[i]:
                ko_ids = torch.tensor(ko_token_ids[i], device=device)
                ko_activations = rep[ko_ids]
                self_loss = self_loss - torch.log(ko_activations + 1e-8).mean()

            if en_token_ids[i]:
                en_ids = torch.tensor(en_token_ids[i], device=device)
                en_activations = rep[en_ids]
                target_loss = target_loss - torch.log(en_activations + 1e-8).mean()
                margin_loss = margin_loss + F.relu(
                    self.target_margin - en_activations
                ).mean()

            if self.non_target_ids is not None:
                non_target_ids_device = self.non_target_ids.to(device)
                non_target_activations = rep[non_target_ids_device]
                negative_loss = negative_loss + F.relu(
                    non_target_activations - 0.1
                ).mean()

            n_valid += 1

        if n_valid > 0:
            self_loss = self_loss / n_valid
            target_loss = target_loss / n_valid
            margin_loss = margin_loss / n_valid
            negative_loss = negative_loss / n_valid

        return {
            "self": self_loss,
            "target": target_loss,
            "margin": margin_loss,
            "negative": negative_loss,
        }


TEST_PAIRS = [
    ("머신러닝", ["machine", "learning"], ["머신", "러닝"]),
    ("딥러닝", ["deep", "learning"], ["딥", "러닝"]),
    ("자연어처리", ["natural", "language", "processing"], ["자연어", "처리"]),
    ("인공지능", ["artificial", "intelligence"], ["인공", "지능"]),
    ("검색엔진", ["search", "engine"], ["검색", "엔진"]),
    ("데이터베이스", ["database"], ["데이터베이스"]),
    ("클라우드", ["cloud"], ["클라우드"]),
    ("서버", ["server"], ["서버"]),
    ("네트워크", ["network"], ["네트워크"]),
    ("추천시스템", ["recommend", "system"], ["추천", "시스템"]),
    ("추천", ["recommend", "recommendation"], ["추천"]),
    ("신경망", ["neural", "network"], ["신경망"]),
    ("강화학습", ["reinforcement", "learning"], ["강화", "학습"]),
    ("컴퓨터비전", ["computer", "vision"], ["컴퓨터", "비전"]),
    ("음성인식", ["speech", "recognition"], ["음성", "인식"]),
]


def evaluate_model(model, tokenizer, device, top_k=50):
    """Evaluate model on test pairs."""
    model.eval()

    ko_activated_total = 0
    en_activated_total = 0
    ko_expected_total = 0
    en_expected_total = 0

    with torch.no_grad():
        for ko_term, en_expected, ko_expected in TEST_PAIRS:
            encoding = tokenizer(
                ko_term,
                max_length=64,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            with autocast("cuda", enabled=CONFIG["use_fp16"]):
                sparse_rep, _ = model(
                    encoding["input_ids"].to(device),
                    encoding["attention_mask"].to(device),
                )

            sparse_rep = sparse_rep[0].float().cpu()
            top_indices = torch.topk(sparse_rep, k=top_k).indices.tolist()
            top_tokens = tokenizer.convert_ids_to_tokens(top_indices)
            top_tokens_set = set(top_tokens)

            for ko in ko_expected:
                ko_toks = tokenizer.tokenize(ko)
                for tok in ko_toks:
                    ko_expected_total += 1
                    if tok in top_tokens_set:
                        ko_activated_total += 1

            for en in en_expected:
                en_toks = tokenizer.tokenize(en.lower())
                for tok in en_toks:
                    en_expected_total += 1
                    if tok in top_tokens_set:
                        en_activated_total += 1

    model.train()

    ko_rate = (
        ko_activated_total / ko_expected_total * 100 if ko_expected_total > 0 else 0
    )
    en_rate = (
        en_activated_total / en_expected_total * 100 if en_expected_total > 0 else 0
    )

    return ko_rate, en_rate


def main():
    print("=" * 70)
    print("v19 TRAINING: XLM-RoBERTa-large with High-Quality Data")
    print("=" * 70)
    print("\nKey features:")
    print(f"  - Dataset: v19_high_quality (~18K pairs, MUSE only, no wikidata)")
    print(f"  - Learning rate: {CONFIG['learning_rate']} (same as v17)")
    print(f"  - Epochs: {CONFIG['num_epochs']} (same as v17)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_mem:.1f} GB")

    print(f"\nLoading tokenizer: {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    print(f"Vocab size: {tokenizer.vocab_size:,}")

    print("\nBuilding non-target language token ID list...")
    non_target_ids = []
    for token_id in tqdm(range(tokenizer.vocab_size)):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if is_non_target_token(token):
            non_target_ids.append(token_id)
    non_target_ids_tensor = torch.tensor(non_target_ids, dtype=torch.long)
    print(f"Found {len(non_target_ids):,} non-target language tokens")

    dataset = TermPairDataset(CONFIG["data_path"], tokenizer, CONFIG["max_length"])

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"\nDataset size: {len(dataset):,}")
    print(f"Batches per epoch: {len(dataloader):,}")
    print(
        f"Effective batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}"
    )

    print(f"\nCreating model: {CONFIG['model_name']}...")
    model = create_splade_model(
        model_name=CONFIG["model_name"],
        use_idf=False,
        use_expansion=True,
        expansion_mode="mlm",
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    loss_fn = TermLevelLoss(
        target_margin=CONFIG["target_margin"], non_target_ids=non_target_ids_tensor
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01
    )

    total_steps = (
        len(dataloader) * CONFIG["num_epochs"] // CONFIG["gradient_accumulation_steps"]
    )
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"\nTotal optimization steps: {total_steps:,}")
    print(f"Warmup steps: {warmup_steps:,}")

    scaler = GradScaler("cuda", enabled=CONFIG["use_fp16"])

    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

    ko_rate, en_rate = evaluate_model(model, tokenizer, device)
    print(f"\nInitial - KO: {ko_rate:.1f}%, EN: {en_rate:.1f}%")

    history = []
    best_score = 0
    global_step = 0

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['num_epochs']} ---")
        model.train()

        epoch_losses = defaultdict(float)
        optimizer.zero_grad()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast("cuda", enabled=CONFIG["use_fp16"]):
                sparse_rep, _ = model(input_ids, attention_mask)

                losses = loss_fn(
                    sparse_rep,
                    batch["ko_token_ids"],
                    batch["en_token_ids"],
                )

                sparsity_loss = sparse_rep.mean()

                total_loss = (
                    CONFIG["lambda_self"] * losses["self"]
                    + CONFIG["lambda_target"] * losses["target"]
                    + CONFIG["lambda_margin"] * losses["margin"]
                    + CONFIG["lambda_negative"] * losses["negative"]
                    + CONFIG["lambda_sparsity"] * sparsity_loss
                )

                total_loss = total_loss / CONFIG["gradient_accumulation_steps"]

            scaler.scale(total_loss).backward()

            epoch_losses["total"] += total_loss.item() * CONFIG["gradient_accumulation_steps"]
            epoch_losses["self"] += losses["self"].item()
            epoch_losses["target"] += losses["target"].item()
            epoch_losses["margin"] += losses["margin"].item()
            epoch_losses["negative"] += losses["negative"].item()

            if (batch_idx + 1) % CONFIG["gradient_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG["max_grad_norm"]
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if (batch_idx + 1) % 100 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{epoch_losses['total'] / (batch_idx + 1):.4f}",
                        "tgt": f"{epoch_losses['target'] / (batch_idx + 1):.4f}",
                        "step": global_step,
                    }
                )

        n_batches = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches

        history.append(dict(epoch_losses))

        ko_rate, en_rate = evaluate_model(model, tokenizer, device)
        combined_score = ko_rate + en_rate

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Total Loss: {epoch_losses['total']:.4f}")
        print(f"  Self Loss: {epoch_losses['self']:.4f}")
        print(f"  Target Loss: {epoch_losses['target']:.4f}")
        print(f"  Korean Preservation: {ko_rate:.1f}%")
        print(f"  English Activation: {en_rate:.1f}%")
        print(f"  Combined Score: {combined_score:.1f}")

        checkpoint_path = CONFIG["output_dir"] / f"checkpoint_epoch{epoch + 1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": dict(epoch_losses),
                "ko_rate": ko_rate,
                "en_rate": en_rate,
                "config": {
                    k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()
                },
            },
            checkpoint_path,
        )
        print(f"  Saved: {checkpoint_path}")

        if combined_score > best_score:
            best_score = combined_score
            best_path = CONFIG["output_dir"] / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "ko_rate": ko_rate,
                    "en_rate": en_rate,
                    "combined_score": combined_score,
                    "config": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in CONFIG.items()
                    },
                },
                best_path,
            )
            print(f"  ★ New best model! Score: {combined_score:.1f} (KO:{ko_rate:.1f}% + EN:{en_rate:.1f}%)")

    final_path = CONFIG["output_dir"] / "final_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()
            },
            "history": history,
        },
        final_path,
    )

    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Final model saved: {final_path}")
    print(f"Best combined score: {best_score:.1f}")

    with open(CONFIG["output_dir"] / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
