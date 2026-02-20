"""Core training module for SPLADE."""

from src.train.core.trainer import SPLADETrainer
from src.train.core.checkpoint import CheckpointManager
from src.train.core.collapse_detector import CollapseDetectionHook
from src.train.core.hooks import (
    TrainingHook,
    LoggingHook,
    CheckpointHook,
    CurriculumHook,
    GradientMonitorHook,
)

__all__ = [
    "SPLADETrainer",
    "CheckpointManager",
    "CollapseDetectionHook",
    "TrainingHook",
    "LoggingHook",
    "CheckpointHook",
    "CurriculumHook",
    "GradientMonitorHook",
]
