"""
SPLADE Neural Sparse Training Module.

This module provides a production-ready training pipeline for SPLADE models
optimized for Korean neural sparse retrieval.

Usage:
    python -m train v22              # Start V22 curriculum training
    python -m train v22 --resume     # Resume from checkpoint
    python -m train v22 --config config.yaml  # Custom config

See `python -m train --help` for all options.
"""

__version__ = "22.0.0"
__author__ = "Neural Sparse Team"

from src.train.config import V22Config, load_config
from src.train.core import SPLADETrainer, CheckpointManager
from src.train.data import SPLADEDataset, DataCollator
from src.train.utils import setup_logging, TrainingMetrics

__all__ = [
    "V22Config",
    "load_config",
    "SPLADETrainer",
    "CheckpointManager",
    "SPLADEDataset",
    "DataCollator",
    "setup_logging",
    "TrainingMetrics",
]
