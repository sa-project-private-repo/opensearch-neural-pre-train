"""
V22 Configuration with Curriculum Learning.

Implements 3-phase curriculum learning:
- Phase 1 (epochs 1-10): Focus on single-term pairs, temperature 0.07
- Phase 2 (epochs 11-20): Balanced data, temperature 0.05
- Phase 3 (epochs 21-30): Full data with hard negatives, temperature 0.03
"""

from dataclasses import dataclass, field
from typing import List, Optional
from src.train.config.base import BaseConfig, ModelConfig, DataConfig, LossConfig, TrainingConfig


@dataclass
class CurriculumPhase:
    """Configuration for a single curriculum learning phase."""

    start_epoch: int
    """Starting epoch (1-indexed)."""

    end_epoch: int
    """Ending epoch (inclusive, 1-indexed)."""

    temperature: float
    """InfoNCE temperature for this phase."""

    lambda_infonce: float
    """InfoNCE loss weight for this phase."""

    lr_multiplier: float = 1.0
    """Learning rate multiplier for this phase."""

    lambda_kd: Optional[float] = None
    """Knowledge distillation loss weight for this phase (v24+)."""

    data_weights: Optional[dict] = None
    """Weights for different data types (e.g., {'single_term': 0.5})."""

    description: str = ""
    """Human-readable description of this phase."""


@dataclass
class V22Config(BaseConfig):
    """
    V22 Configuration with Curriculum Learning.

    Extends BaseConfig with curriculum learning phases and
    temperature annealing for InfoNCE loss.
    """

    curriculum_phases: List[CurriculumPhase] = field(default_factory=list)
    """List of curriculum learning phases."""

    enable_curriculum: bool = True
    """Whether to enable curriculum learning."""

    # V22-specific defaults
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="skt/kobert-base-v1",
        dropout=0.1,
        use_expansion=True,
        expansion_mode="mlm",
    ))

    loss: LossConfig = field(default_factory=lambda: LossConfig(
        lambda_infonce=2.0,
        lambda_self=4.0,
        lambda_positive=10.0,
        lambda_margin=2.5,
        lambda_flops=5e-3,
        lambda_min_act=1.0,
        temperature=0.07,  # Start with higher temp
        margin=1.5,
    ))

    training: TrainingConfig = field(default_factory=lambda: TrainingConfig(
        num_epochs=30,
        learning_rate=3e-6,
        warmup_ratio=0.1,
        gradient_clip=1.0,
        mixed_precision="bf16",
        output_dir="outputs/train_v22",
        experiment_name="splade_v22_curriculum",
    ))

    def __post_init__(self) -> None:
        """Initialize default curriculum phases if not provided."""
        super().__post_init__()

        if not self.curriculum_phases and self.enable_curriculum:
            self.curriculum_phases = [
                CurriculumPhase(
                    start_epoch=1,
                    end_epoch=10,
                    temperature=0.07,
                    lambda_infonce=1.0,
                    lr_multiplier=1.0,
                    data_weights={"single_term": 0.5},
                    description="Phase 1: Focus on single-term pairs",
                ),
                CurriculumPhase(
                    start_epoch=11,
                    end_epoch=20,
                    temperature=0.05,
                    lambda_infonce=1.5,
                    lr_multiplier=0.5,
                    data_weights={"single_term": 0.33, "multi_term": 0.33, "hard_neg": 0.34},
                    description="Phase 2: Balanced training",
                ),
                CurriculumPhase(
                    start_epoch=21,
                    end_epoch=30,
                    temperature=0.03,
                    lambda_infonce=2.0,
                    lr_multiplier=0.25,
                    data_weights={"hard_neg": 0.5},
                    description="Phase 3: Hard negative focus",
                ),
            ]

    def get_phase_for_epoch(self, epoch: int) -> Optional[CurriculumPhase]:
        """
        Get the curriculum phase for a given epoch.

        Args:
            epoch: Current epoch (1-indexed)

        Returns:
            CurriculumPhase for the epoch, or None if no curriculum
        """
        if not self.enable_curriculum:
            return None

        for phase in self.curriculum_phases:
            if phase.start_epoch <= epoch <= phase.end_epoch:
                return phase
        return None

    def get_temperature_for_epoch(self, epoch: int) -> float:
        """Get temperature for a given epoch."""
        phase = self.get_phase_for_epoch(epoch)
        if phase:
            return phase.temperature
        return self.loss.temperature

    def get_lambda_infonce_for_epoch(self, epoch: int) -> float:
        """Get lambda_infonce for a given epoch."""
        phase = self.get_phase_for_epoch(epoch)
        if phase:
            return phase.lambda_infonce
        return self.loss.lambda_infonce

    def get_lr_multiplier_for_epoch(self, epoch: int) -> float:
        """Get learning rate multiplier for a given epoch."""
        phase = self.get_phase_for_epoch(epoch)
        if phase:
            return phase.lr_multiplier
        return 1.0


def create_default_v22_config(
    train_files: List[str],
    val_files: Optional[List[str]] = None,
    output_dir: str = "outputs/train_v22",
    model_name: str = "skt/kobert-base-v1",
    batch_size: int = 64,
    num_epochs: int = 30,
) -> V22Config:
    """
    Create a default V22 configuration with sensible defaults.

    Args:
        train_files: List of training data file paths
        val_files: Optional list of validation file paths
        output_dir: Output directory for checkpoints
        model_name: Pretrained model name
        batch_size: Training batch size
        num_epochs: Number of training epochs

    Returns:
        Configured V22Config instance
    """
    config = V22Config(
        model=ModelConfig(
            name=model_name,
            dropout=0.1,
            use_expansion=True,
        ),
        data=DataConfig(
            train_files=train_files,
            val_files=val_files or [],
            batch_size=batch_size,
            max_length=64,
            num_workers=4,
        ),
        training=TrainingConfig(
            num_epochs=num_epochs,
            learning_rate=3e-6,
            output_dir=output_dir,
        ),
    )

    config.validate()
    return config
