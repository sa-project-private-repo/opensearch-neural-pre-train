"""
V24 Configuration for XLM-RoBERTa SPLADE with BGE-M3 Teacher.

Key differences from V22:
- XLM-RoBERTa base (250K vocab) instead of KoBERT (50K vocab)
- BGE-M3 as teacher model for knowledge distillation
- Rebalanced loss weights for stronger contrastive learning
- Longer max_length (192 vs 128)
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from src.train.config.base import BaseConfig, DataConfig, LossConfig, ModelConfig, TrainingConfig
from src.train.config.v22 import CurriculumPhase


@dataclass
class KDConfig:
    """Knowledge distillation configuration."""

    enabled: bool = True
    """Whether to enable knowledge distillation."""

    teacher_model: str = "BAAI/bge-m3"
    """Teacher model for distillation."""

    teacher_max_length: int = 512
    """Maximum sequence length for teacher encoding."""

    normalize_embeddings: bool = True
    """Whether to L2 normalize teacher embeddings."""

    warmup_epochs: int = 1
    """Number of epochs before enabling KD."""


@dataclass
class HardNegativeConfig:
    """Hard negative mining configuration."""

    enabled: bool = True
    """Whether to enable hard negative mining."""

    bm25_negatives: int = 5
    """Number of BM25-based negatives per sample."""

    dense_negatives: int = 5
    """Number of dense-based negatives per sample."""

    refresh_every_n_epochs: int = 5
    """Re-mine hard negatives every N epochs."""


@dataclass
class V24ModelConfig(ModelConfig):
    """V24-specific model configuration."""

    name: str = "xlm-roberta-base"
    """XLM-RoBERTa base model (250K vocab)."""

    model_class: Literal["SPLADEDocXLMR", "SPLADEDocXLMRWithIDF"] = "SPLADEDocXLMR"
    """Model class to use."""

    dropout: float = 0.1
    """Dropout rate."""

    use_expansion: bool = True
    """Use MLM head for vocabulary expansion."""

    expansion_mode: Literal["mlm", "projection"] = "mlm"
    """Expansion mode."""


@dataclass
class V24LossConfig(LossConfig):
    """V24-specific loss configuration."""

    # Rebalanced loss weights for XLM-R + BGE-M3
    lambda_infonce: float = 3.0
    """Stronger contrastive signal."""

    lambda_self: float = 0.5
    """Reduced self-reconstruction."""

    lambda_positive: float = 2.0
    """Balanced positive activation."""

    lambda_margin: float = 0.0
    """Disabled (redundant with InfoNCE)."""

    lambda_flops: float = 0.002
    """Allow more activations for 250K vocab."""

    lambda_min_act: float = 1.0
    """Activation floor."""

    lambda_kd: float = 2.0
    """Knowledge distillation weight."""

    kd_temperature: float = 3.0
    """KD temperature for soft labels."""

    # Loss hyperparameters
    temperature: float = 0.07
    """InfoNCE temperature."""

    margin: float = 0.3
    """Cosine similarity margin."""

    top_k: int = 5
    """Top-k for minimum activation."""

    min_activation: float = 0.5
    """Minimum activation threshold."""

    # IDF-aware FLOPS
    use_idf_weighting: bool = True
    """Enable IDF-aware penalty."""

    idf_alpha: float = 2.5
    """IDF exponential decay factor."""


@dataclass
class V24Config(BaseConfig):
    """
    V24 Configuration for XLM-RoBERTa SPLADE with BGE-M3 Teacher.

    Key improvements over V22:
    - XLM-RoBERTa backbone (250K vocab vs 50K)
    - BGE-M3 teacher for knowledge distillation
    - Stronger contrastive learning
    - Hard negative mining support
    """

    # Override with V24-specific configs
    model: V24ModelConfig = field(default_factory=V24ModelConfig)
    """Model configuration."""

    loss: V24LossConfig = field(default_factory=V24LossConfig)
    """Loss configuration."""

    data: DataConfig = field(default_factory=lambda: DataConfig(
        train_files=["data/v22.0/train_*.jsonl", "data/v22.0/augmented_*.jsonl"],
        val_files=["data/v22.0/val_*.jsonl"],
        batch_size=48,
        max_length=192,
        num_workers=4,
    ))
    """Data configuration."""

    training: TrainingConfig = field(default_factory=lambda: TrainingConfig(
        num_epochs=25,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        gradient_clip=1.0,
        gradient_accumulation_steps=2,
        mixed_precision="bf16",
        output_dir="outputs/train_v24",
        experiment_name="splade_v24_xlmr_bge",
    ))
    """Training configuration."""

    # Knowledge distillation
    knowledge_distillation: KDConfig = field(default_factory=KDConfig)
    """Knowledge distillation configuration."""

    # Hard negative mining
    hard_negative_mining: HardNegativeConfig = field(default_factory=HardNegativeConfig)
    """Hard negative mining configuration."""

    # Curriculum learning
    curriculum_phases: List[CurriculumPhase] = field(default_factory=list)
    """Curriculum learning phases."""

    enable_curriculum: bool = True
    """Whether to enable curriculum learning."""

    def __post_init__(self) -> None:
        """Initialize default curriculum phases if not provided."""
        # Convert nested dicts to V24-specific dataclasses
        if isinstance(self.model, dict):
            self.model = V24ModelConfig(**self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.loss, dict):
            self.loss = V24LossConfig(**self.loss)
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
                    end_epoch=8,
                    temperature=0.08,
                    lambda_infonce=2.5,
                    lr_multiplier=1.0,
                    lambda_kd=2.5,
                    data_weights={
                        "single_term": 0.4,
                        "multi_term": 0.35,
                        "original": 0.25,
                    },
                    description="Phase 1: Foundation with BGE-M3 teacher guidance",
                ),
                CurriculumPhase(
                    start_epoch=9,
                    end_epoch=17,
                    temperature=0.05,
                    lambda_infonce=3.0,
                    lr_multiplier=0.5,
                    lambda_kd=1.5,
                    data_weights={
                        "single_term": 0.3,
                        "multi_term": 0.35,
                        "hard_neg": 0.35,
                    },
                    description="Phase 2: Balanced training with hard negatives",
                ),
                CurriculumPhase(
                    start_epoch=18,
                    end_epoch=25,
                    temperature=0.04,
                    lambda_infonce=3.0,
                    lr_multiplier=0.25,
                    lambda_kd=0.8,
                    data_weights={
                        "hard_neg": 0.5,
                        "multi_term": 0.3,
                        "single_term": 0.2,
                    },
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

    def get_lambda_kd_for_epoch(self, epoch: int) -> float:
        """Get knowledge distillation weight for epoch."""
        phase = self.get_phase_for_epoch(epoch)
        if phase and phase.lambda_kd is not None:
            return phase.lambda_kd
        return self.loss.lambda_kd

    def get_temperature_for_epoch(self, epoch: int) -> float:
        """Get temperature for epoch."""
        phase = self.get_phase_for_epoch(epoch)
        if phase:
            return phase.temperature
        return self.loss.temperature

    def get_lambda_infonce_for_epoch(self, epoch: int) -> float:
        """Get lambda_infonce for epoch."""
        phase = self.get_phase_for_epoch(epoch)
        if phase:
            return phase.lambda_infonce
        return self.loss.lambda_infonce

    def get_lr_multiplier_for_epoch(self, epoch: int) -> float:
        """Get learning rate multiplier for epoch."""
        phase = self.get_phase_for_epoch(epoch)
        if phase:
            return phase.lr_multiplier
        return 1.0


def create_default_v24_config(
    train_files: Optional[List[str]] = None,
    val_files: Optional[List[str]] = None,
    output_dir: str = "outputs/train_v24",
    model_name: str = "xlm-roberta-base",
    batch_size: int = 48,
    num_epochs: int = 25,
) -> V24Config:
    """
    Create a default V24 configuration.

    Args:
        train_files: Training data files
        val_files: Validation data files
        output_dir: Output directory
        model_name: XLM-R model variant
        batch_size: Batch size
        num_epochs: Number of epochs

    Returns:
        Configured V24Config instance
    """
    if train_files is None:
        train_files = [
            "data/v22.0/train_*.jsonl",
            "data/v22.0/augmented_*.jsonl",
        ]
    if val_files is None:
        val_files = ["data/v22.0/val_*.jsonl"]

    config = V24Config(
        model=V24ModelConfig(
            name=model_name,
            dropout=0.1,
            use_expansion=True,
            expansion_mode="mlm",
        ),
        data=DataConfig(
            train_files=train_files,
            val_files=val_files,
            batch_size=batch_size,
            max_length=192,
            num_workers=4,
        ),
        training=TrainingConfig(
            num_epochs=num_epochs,
            learning_rate=3e-5,
            output_dir=output_dir,
            experiment_name="splade_v24_xlmr_bge",
        ),
    )

    config.validate()
    return config
