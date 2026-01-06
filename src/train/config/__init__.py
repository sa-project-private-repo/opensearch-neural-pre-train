"""Configuration module for SPLADE training."""

from src.train.config.base import BaseConfig, ModelConfig, DataConfig, LossConfig, TrainingConfig
from src.train.config.v22 import V22Config, CurriculumPhase
from src.train.config.loader import load_config, save_config

__all__ = [
    "BaseConfig",
    "ModelConfig",
    "DataConfig",
    "LossConfig",
    "TrainingConfig",
    "V22Config",
    "CurriculumPhase",
    "load_config",
    "save_config",
]
