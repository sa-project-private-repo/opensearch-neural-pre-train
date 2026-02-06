"""
V29 Configuration for SPLADE with SPLADE v2 improvements + AI Hub data.

Key features:
- Max pooling (SPLADE v2)
- Separate FLOPS regularization (lambda_q, lambda_d)
- Quadratic lambda warmup
- AI Hub data integration
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from src.train.config.base import BaseConfig, DataConfig, TrainingConfig
from src.train.config.v22 import CurriculumPhase
from src.train.config.v24 import KDConfig, HardNegativeConfig
from src.train.config.v28 import V28LossConfig, V28ModelConfig


@dataclass
class V29LossConfig(V28LossConfig):
    """V29-specific loss configuration with SPLADE v2 improvements."""

    # SPLADE v2: Separate FLOPS regularization
    lambda_flops_q: float = 1e-4
    """FLOPS regularization weight for queries."""

    lambda_flops_d: float = 1e-3
    """FLOPS regularization weight for documents."""

    lambda_warmup_steps: int = 50000
    """Steps for quadratic lambda warmup."""

    # Distillation
    use_margin_mse: bool = False
    """Use margin-MSE distillation from cross-encoder."""

    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    """Cross-encoder model for margin-MSE distillation."""


@dataclass
class V29ModelConfig(V28ModelConfig):
    """V29-specific model configuration."""

    model_class: Literal[
        "SPLADEDocXLMR",
        "SPLADEDocContextGated",
        "SPLADEDocV29",
        "SPLADEDocV29ContextGated"
    ] = "SPLADEDocV29ContextGated"
    """Model class to use."""

    pooling: Literal["max", "sum"] = "max"
    """Pooling strategy - max (SPLADE v2) or sum (original)."""


@dataclass
class V29Config(BaseConfig):
    """
    V29 Configuration for SPLADE with SPLADE v2 improvements.

    Key improvements over V28:
    - Max pooling (SPLADE v2 Eq. 6) - ~2pt MRR improvement
    - Separate FLOPS regularization (lambda_q, lambda_d)
    - Quadratic lambda warmup schedule
    - AI Hub data integration (datasets 624, 86, 71828)

    Target metrics:
    - Recall@1 > 45% (vs V28 ~40%)
    - Korean token ratio > 90%
    - Average active tokens ~100 (more sparse)
    """

    model: V29ModelConfig = field(default_factory=V29ModelConfig)
    """Model configuration with pooling setting."""

    loss: V29LossConfig = field(default_factory=V29LossConfig)
    """Loss configuration with separate FLOPS weights."""

    data: DataConfig = field(default_factory=lambda: DataConfig(
        train_files=[
            "data/v24.0/train_*.jsonl",
            "data/aihub/processed/*.jsonl",
        ],
        val_files=["data/v24.0/val.jsonl"],
        batch_size=32,
        max_length=256,
        num_workers=4,
    ))
    """Data configuration with AI Hub data."""

    training: TrainingConfig = field(default_factory=lambda: TrainingConfig(
        num_epochs=15,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        gradient_clip=1.0,
        gradient_accumulation_steps=2,
        mixed_precision="bf16",
        output_dir="outputs/train_v29",
        experiment_name="splade_v29_xlmr_max_pooling",
        log_every_n_steps=50,
        save_every_n_epochs=3,
    ))
    """Training configuration."""

    knowledge_distillation: KDConfig = field(default_factory=KDConfig)
    """Knowledge distillation configuration."""

    hard_negative_mining: HardNegativeConfig = field(default_factory=HardNegativeConfig)
    """Hard negative mining configuration."""

    curriculum_phases: List[CurriculumPhase] = field(default_factory=list)
    """Curriculum learning phases."""

    enable_curriculum: bool = True
    """Whether to enable curriculum learning."""

    def __post_init__(self) -> None:
        """Initialize default curriculum phases if not provided."""
        if isinstance(self.model, dict):
            self.model = V29ModelConfig(**self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.loss, dict):
            self.loss = V29LossConfig(**self.loss)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.knowledge_distillation, dict):
            self.knowledge_distillation = KDConfig(**self.knowledge_distillation)
        if isinstance(self.hard_negative_mining, dict):
            self.hard_negative_mining = HardNegativeConfig(**self.hard_negative_mining)

        if not self.curriculum_phases and self.enable_curriculum:
            self.curriculum_phases = [
                CurriculumPhase(
                    start_epoch=1,
                    end_epoch=5,
                    temperature=0.08,
                    lambda_infonce=2.5,
                    lr_multiplier=1.0,
                    lambda_kd=2.5,
                    description="Phase 1: Foundation with teacher guidance",
                ),
                CurriculumPhase(
                    start_epoch=6,
                    end_epoch=10,
                    temperature=0.05,
                    lambda_infonce=3.0,
                    lr_multiplier=0.5,
                    lambda_kd=1.5,
                    description="Phase 2: Balanced training",
                ),
                CurriculumPhase(
                    start_epoch=11,
                    end_epoch=15,
                    temperature=0.04,
                    lambda_infonce=3.0,
                    lr_multiplier=0.25,
                    lambda_kd=0.8,
                    description="Phase 3: Hard negative refinement",
                ),
            ]

    def get_phase_for_epoch(self, epoch: int) -> Optional[CurriculumPhase]:
        """Get curriculum phase for given epoch."""
        if not self.enable_curriculum:
            return None

        for phase in self.curriculum_phases:
            if phase.start_epoch <= epoch <= phase.end_epoch:
                return phase
        return None

    def get_idf_cache_path(self) -> str:
        """Get default IDF cache path based on output dir."""
        if self.loss.idf_weights_path:
            return self.loss.idf_weights_path
        return f"{self.training.output_dir}/idf_weights/xlmr_v29_idf"


def create_default_v29_config(
    train_files: Optional[List[str]] = None,
    val_files: Optional[List[str]] = None,
    output_dir: str = "outputs/train_v29",
    model_name: str = "xlm-roberta-base",
    batch_size: int = 32,
    num_epochs: int = 15,
    pooling: str = "max",
    use_context_gate: bool = True,
    lambda_flops_q: float = 1e-4,
    lambda_flops_d: float = 1e-3,
) -> V29Config:
    """
    Create a default V29 configuration.

    Args:
        train_files: Training data files
        val_files: Validation data files
        output_dir: Output directory
        model_name: XLM-R model variant
        batch_size: Batch size per GPU
        num_epochs: Number of epochs
        pooling: Pooling strategy ("max" or "sum")
        use_context_gate: Whether to use context-gated model
        lambda_flops_q: Query FLOPS weight
        lambda_flops_d: Document FLOPS weight

    Returns:
        Configured V29Config instance
    """
    if train_files is None:
        train_files = [
            "data/v24.0/train_*.jsonl",
            "data/aihub/processed/*.jsonl",
        ]
    if val_files is None:
        val_files = ["data/v24.0/val.jsonl"]

    model_class = "SPLADEDocV29ContextGated" if use_context_gate else "SPLADEDocV29"

    config = V29Config(
        model=V29ModelConfig(
            name=model_name,
            dropout=0.1,
            use_expansion=True,
            expansion_mode="mlm",
            use_context_gate=use_context_gate,
            model_class=model_class,
            pooling=pooling,
        ),
        data=DataConfig(
            train_files=train_files,
            val_files=val_files,
            batch_size=batch_size,
            max_length=256,
            num_workers=4,
        ),
        loss=V29LossConfig(
            lambda_flops_q=lambda_flops_q,
            lambda_flops_d=lambda_flops_d,
            enable_language_filtering=True,
            use_context_gate=use_context_gate,
        ),
        training=TrainingConfig(
            num_epochs=num_epochs,
            learning_rate=2e-5,
            output_dir=output_dir,
            experiment_name="splade_v29_xlmr",
        ),
    )

    config.validate()
    return config
