"""CLI module for V33 SPLADE training."""

from src.train.cli.train_v33_ddp import main as train_v33

__all__ = [
    "train_v33",
]
