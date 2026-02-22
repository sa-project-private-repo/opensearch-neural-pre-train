"""
Korean MLM Pre-training Script.

Continue pre-training XLM-R base on Korean text via masked language modeling,
so the encoder better understands Korean before SPLADE sparse retrieval training.

Usage:
    torchrun --nproc_per_node=8 -m src.train.cli.pretrain_mlm \
        --config configs/pretrain_mlm.yaml

    # Single GPU
    python -m src.train.cli.pretrain_mlm \
        --data-dir data/mlm_korean \
        --output-dir outputs/pretrain_mlm
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import yaml
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LineByLineTextDataset(Dataset):
    """Reads a line-delimited text file and tokenizes each line.

    Each non-empty line is treated as a single training example.
    Blank lines are skipped.

    Args:
        file_path: Path to the text file (one sentence/document per line).
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum token sequence length.
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Loading dataset from {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(lines):,} lines from {file_path}")

        self.encodings = tokenizer(
            lines,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=False,
        )

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single tokenized example."""
        return {
            "input_ids": torch.tensor(
                self.encodings["input_ids"][idx], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                self.encodings["attention_mask"][idx], dtype=torch.long
            ),
        }


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments, with optional YAML config override."""
    parser = argparse.ArgumentParser(
        description="Korean MLM Pre-training with XLM-R",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. CLI flags override config values.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/mlm_korean",
        help="Directory containing train.txt and val.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/pretrain_mlm",
        help="Output directory for checkpoints and final model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="xlm-roberta-base",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Per-GPU batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token sequence length",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Resume training from a checkpoint directory",
    )
    parser.add_argument(
        "--mlm-probability",
        type=float,
        default=0.15,
        help="Probability of masking each token for MLM",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Fraction of total steps used for LR warm-up",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=2000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=1000,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=100,
        help="Log metrics every N steps",
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Merge YAML config (CLI flags take priority)
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            yaml_cfg: dict = yaml.safe_load(f) or {}

        cli_overrides = {
            k: v
            for k, v in vars(args).items()
            if v is not None and k != "config"
        }
        # Apply yaml defaults, then overlay explicit CLI overrides
        merged = {**yaml_cfg, **cli_overrides}
        # Re-map snake_case yaml keys to args namespace (replace - with _)
        for key, value in merged.items():
            attr = key.replace("-", "_")
            if hasattr(args, attr):
                setattr(args, attr, value)

    return args


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    """Construct HuggingFace TrainingArguments from parsed CLI args."""
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        bf16_full_eval=True,
        gradient_accumulation_steps=args.grad_accum,
        # Evaluation and checkpointing
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        report_to="tensorboard",
        # Performance
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,
        # DDP handled automatically by torchrun / Trainer
        ddp_find_unused_parameters=False,
    )


def main() -> None:
    """Entry point for Korean MLM pre-training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    if is_main:
        logger.info("=" * 60)
        logger.info("Korean MLM Pre-training")
        logger.info("=" * 60)
        logger.info(f"  model      : {args.model_name}")
        logger.info(f"  data_dir   : {args.data_dir}")
        logger.info(f"  output_dir : {args.output_dir}")
        logger.info(f"  epochs     : {args.epochs}")
        logger.info(f"  batch_size : {args.batch_size} per GPU")
        logger.info(f"  lr         : {args.lr}")
        logger.info(f"  max_length : {args.max_length}")
        logger.info(f"  grad_accum : {args.grad_accum}")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Tokenizer & Model
    # ------------------------------------------------------------------
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {num_params:,}")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    data_dir = Path(args.data_dir)
    train_dataset = LineByLineTextDataset(
        file_path=data_dir / "train.txt",
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    eval_dataset: Optional[LineByLineTextDataset] = None
    val_path = data_dir / "val.txt"
    if val_path.exists():
        eval_dataset = LineByLineTextDataset(
            file_path=val_path,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
    else:
        logger.warning(
            f"Validation file not found at {val_path}. "
            "Skipping evaluation."
        )

    # ------------------------------------------------------------------
    # Data Collator
    # ------------------------------------------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    # ------------------------------------------------------------------
    # Training Arguments
    # ------------------------------------------------------------------
    training_args = build_training_arguments(args)

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    resume_from: Optional[str] = args.resume
    if resume_from is None:
        # Auto-detect latest checkpoint in output_dir
        output_path = Path(args.output_dir)
        if output_path.exists():
            checkpoints = sorted(output_path.glob("checkpoint-*"))
            if checkpoints:
                resume_from = str(checkpoints[-1])
                if is_main:
                    logger.info(f"Auto-resuming from {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    if is_main:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Saved final model to {args.output_dir}")


if __name__ == "__main__":
    main()
