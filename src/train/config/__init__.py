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
from src.train.config.v25 import (
    V25Config,
    V25LossConfig,
    create_default_v25_config,
)
from src.train.config.v29 import (
    V29Config,
    V29LossConfig,
    V29ModelConfig,
    create_default_v29_config,
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
    # V25
    "V25Config",
    "V25LossConfig",
    "create_default_v25_config",
    # V29
    "V29Config",
    "V29LossConfig",
    "V29ModelConfig",
    "create_default_v29_config",
    # Utils
    "load_config",
    "save_config",
]
