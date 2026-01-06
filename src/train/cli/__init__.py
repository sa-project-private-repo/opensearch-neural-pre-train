"""CLI module for SPLADE training commands."""

from src.train.cli.train_v22 import main as train_v22
from src.train.cli.resume import main as resume

__all__ = [
    "train_v22",
    "resume",
]
