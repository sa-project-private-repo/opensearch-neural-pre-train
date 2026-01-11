"""Configuration module for SPLADE training."""

from src.train.config.base import (
    BaseConfig,
    DataConfig,
    LossConfig,
    ModelConfig,
    TrainingConfig,
)
from src.train.config.loader import load_config, save_config
from src.train.config.v22 import CurriculumPhase, V22Config
from src.train.config.v24 import (
    HardNegativeConfig,
    KDConfig,
    V24Config,
    V24LossConfig,
    V24ModelConfig,
)

__all__ = [
    # Base configs
    "BaseConfig",
    "ModelConfig",
    "DataConfig",
    "LossConfig",
    "TrainingConfig",
    # V22
    "V22Config",
    "CurriculumPhase",
    # V24
    "V24Config",
    "V24ModelConfig",
    "V24LossConfig",
    "KDConfig",
    "HardNegativeConfig",
    # Utils
    "load_config",
    "save_config",
]
