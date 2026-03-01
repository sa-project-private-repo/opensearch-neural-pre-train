"""
SPLADE V33 Neural Sparse Training Module.

Usage:
    python -m train v33              # Start V33 DDP training
    python -m train v33 --config configs/train_v33.yaml

See `python -m train --help` for all options.
"""

__version__ = "33.0.0"
__author__ = "Neural Sparse Team"

try:
    from src.train.config import V33Config, load_config
    from src.train.core import SPLADETrainer, CheckpointManager
    from src.train.utils import setup_logging, TrainingMetrics

    __all__ = [
        "V33Config",
        "load_config",
        "SPLADETrainer",
        "CheckpointManager",
        "setup_logging",
        "TrainingMetrics",
    ]
except ImportError:
    __all__ = []
