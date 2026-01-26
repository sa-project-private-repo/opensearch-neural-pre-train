"""
V26 Configuration for XLM-RoBERTa SPLADE with Enhanced IDF-Aware FLOPS.

Key differences from V25:
- Special token exclusion from IDF normalization
- Fixed high penalty for special tokens (100.0)
- Increased FLOPS weight (0.010 vs 0.002)
- Increased stopword penalty (15.0 vs 5.0)
- Sharper IDF curve (alpha 4.0 vs 2.5)
- Extended Korean stopword list
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from src.train.config.base import BaseConfig, DataConfig, ModelConfig, TrainingConfig
from src.train.config.v22 import CurriculumPhase
from src.train.config.v24 import KDConfig, HardNegativeConfig, V24ModelConfig


@dataclass
class V26LossConfig:
    """V26-specific loss configuration with enhanced IDF handling."""

    # Loss weights (V26 defaults - higher FLOPS weight)
    lambda_infonce: float = 3.0
    """Contrastive loss weight."""

    lambda_self: float = 0.5
    """Self-reconstruction loss weight."""

    lambda_positive: float = 2.0
    """Positive activation loss weight."""

    lambda_margin: float = 0.0
    """Triplet margin loss weight (disabled)."""

    lambda_flops: float = 0.010  # 5x increase from V25
    """IDF-aware FLOPS regularization weight (increased from 0.002)."""

    lambda_min_act: float = 1.0
    """Minimum activation loss weight."""

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

    # IDF configuration (V26 enhanced)
    use_idf_weighting: bool = True
    """Enable IDF-aware penalty (always True for V26)."""

    idf_alpha: float = 4.0  # Increased from V25's 2.5
    """IDF exponential decay factor (sharper curve in V26)."""

    idf_weights_path: Optional[str] = None
    """Path to pre-computed IDF weights. If None, will compute from corpus."""

    recompute_idf: bool = False
    """Force recomputation of IDF weights even if cache exists."""

    idf_smoothing: Literal["bm25", "standard"] = "bm25"
    """IDF smoothing method."""

    # Special token configuration (V26 new)
    special_token_penalty: float = 100.0
    """Fixed penalty for special tokens (<s>, </s>, etc.)."""

    # Stopword configuration (V26 enhanced)
    use_stopword_mask: bool = True
    """Enable stopword masking at inference time."""

    stopword_penalty: float = 15.0  # 3x increase from V25
    """Extra penalty multiplier for stopwords in FLOPS loss (increased from 5.0)."""

    use_extended_stopwords: bool = True
    """Use KOREAN_STOPWORDS_V26 extended list."""


@dataclass
class V26Config(BaseConfig):
    """
    V26 Configuration for XLM-RoBERTa SPLADE with Enhanced IDF-Aware FLOPS.

    Key improvements over V25:
    - Special tokens excluded from IDF normalization range
    - Fixed high penalty (100.0) for special tokens
    - 5x FLOPS weight increase (0.002 -> 0.010)
    - 3x stopword penalty increase (5.0 -> 15.0)
    - Sharper IDF curve (alpha 2.5 -> 4.0)
    - Extended Korean stopword list with V25 observed terms
    """

    # Override with V26-specific configs
    model: V24ModelConfig = field(default_factory=V24ModelConfig)
    """Model configuration (same as V24/V25)."""

    loss: V26LossConfig = field(default_factory=V26LossConfig)
    """Loss configuration with enhanced IDF settings."""

    data: DataConfig = field(default_factory=lambda: DataConfig(
        train_files=["data/v24.0/train_*.jsonl"],
        val_files=["data/v24.0/val.jsonl"],
        batch_size=24,
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
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        output_dir="outputs/train_v26",
        experiment_name="splade_v26_xlmr_enhanced_idf",
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
        # Convert nested dicts to V26-specific dataclasses
        if isinstance(self.model, dict):
            self.model = V24ModelConfig(**self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.loss, dict):
            self.loss = V26LossConfig(**self.loss)
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

    def get_idf_cache_path(self) -> str:
        """Get default IDF cache path based on output dir."""
        if self.loss.idf_weights_path:
            return self.loss.idf_weights_path
        return f"{self.training.output_dir}/idf_weights/xlmr_v26_idf"


def create_default_v26_config(
    train_files: Optional[List[str]] = None,
    val_files: Optional[List[str]] = None,
    output_dir: str = "outputs/train_v26",
    model_name: str = "xlm-roberta-base",
    batch_size: int = 24,
    num_epochs: int = 25,
    idf_weights_path: Optional[str] = None,
) -> V26Config:
    """
    Create a default V26 configuration.

    Args:
        train_files: Training data files
        val_files: Validation data files
        output_dir: Output directory
        model_name: XLM-R model variant
        batch_size: Batch size
        num_epochs: Number of epochs
        idf_weights_path: Path to pre-computed IDF weights

    Returns:
        Configured V26Config instance
    """
    if train_files is None:
        train_files = ["data/v24.0/train_*.jsonl"]
    if val_files is None:
        val_files = ["data/v24.0/val.jsonl"]

    config = V26Config(
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
        loss=V26LossConfig(
            idf_weights_path=idf_weights_path,
        ),
        training=TrainingConfig(
            num_epochs=num_epochs,
            learning_rate=3e-5,
            output_dir=output_dir,
            experiment_name="splade_v26_xlmr_enhanced_idf",
        ),
    )

    config.validate()
    return config
