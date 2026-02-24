"""
V28 Configuration for XLM-RoBERTa SPLADE with Language Filtering and Context Gate.

V28a: Korean Language Filtering
- Suppress non-Korean tokens (multilingual noise removal)
- Fixed high penalty for non-Korean tokens (100.0)

V28b: Context-Gated Sparse Expansion (CGSE)
- Document context determines token activation
- Same keyword in different contexts activates different tokens
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from src.train.config.base import BaseConfig, DataConfig, TrainingConfig
from src.train.config.v22 import CurriculumPhase
from src.train.config.v24 import KDConfig, HardNegativeConfig, V24ModelConfig
from src.train.config.v26 import V26LossConfig


@dataclass
class V28LossConfig(V26LossConfig):
    """V28-specific loss configuration with language filtering and context gate."""

    # ===== V28a: Language Filtering =====
    enable_language_filtering: bool = True
    """Enable Korean language token filtering."""

    korean_token_penalty: float = 0.0
    """Penalty for Korean tokens (0.0 = no penalty, preserve Korean)."""

    non_korean_penalty: float = 1.0
    """Penalty for non-Korean tokens (soft nudge, not death sentence)."""

    lambda_language: float = 0.1
    """Weight for language filtering penalty in total loss."""

    lambda_min_act: float = 5.0
    """Must dominate language penalty to prevent collapse."""

    top_k: int = 10
    """Monitor more activation tokens."""

    min_activation: float = 1.0
    """Higher floor prevents collapse."""

    # Language penalty warmup
    language_warmup_steps: int = 5000
    """Steps to linearly warmup language penalty from 0."""

    language_penalty_max: float = 0.1
    """Maximum lambda_language after warmup."""

    # Collapse detection
    collapse_flops_threshold: float = 0.01
    """FLOPS below this for N checks triggers collapse guard."""

    collapse_check_window: int = 3
    """Consecutive low-flops checks before halving penalty."""

    # FLOPS warmup
    flops_warmup_steps: int = 0
    """Steps to linearly warmup FLOPS penalty from 0 to lambda_flops."""

    # V29: Separate FLOPS regularization (SPLADE v2 style)
    lambda_flops_q: float = 0.0
    """FLOPS regularization weight for queries (SPLADE v2). 0=disabled."""

    lambda_flops_d: float = 0.0
    """FLOPS regularization weight for documents (SPLADE v2). 0=disabled."""

    # ===== V28b: Context-Gated Expansion =====
    use_context_gate: bool = True
    """Enable context-gated sparse expansion."""

    context_gate_hidden: int = 256
    """Hidden dimension for context gate MLP."""

    context_attention_heads: int = 4
    """Number of attention heads for context pooling."""

    context_gate_dropout: float = 0.1
    """Dropout rate for context gate layers."""

    # Context-aware KD
    use_context_aware_kd: bool = True
    """Enable context-aware knowledge distillation."""

    context_kd_weight: float = 1.0
    """Weight for context-aware KD loss."""


@dataclass
class V28ModelConfig(V24ModelConfig):
    """V28-specific model configuration."""

    model_class: Literal["SPLADEDocXLMR", "SPLADEDocContextGated"] = "SPLADEDocContextGated"
    """Model class to use (V28 uses context-gated variant)."""

    use_context_gate: bool = True
    """Whether to use context-gated model."""

    context_gate_hidden: int = 256
    """Hidden dimension for context gate."""

    context_attention_heads: int = 4
    """Number of attention heads in context gate."""

    top_k_sparse: int = 0
    """Top-k activation constraint during training (0=disabled)."""


@dataclass
class V28Config(BaseConfig):
    """
    V28 Configuration for XLM-RoBERTa SPLADE with Language Filtering and Context Gate.

    Key improvements over V26/V27:
    - V28a: Korean language filtering (suppress multilingual token leakage)
    - V28b: Context-gated sparse expansion (context-dependent activation)

    Architecture:
        Document -> Transformer -> Hidden States
                        |
                 Context Pooling -> Context Vector [batch, hidden]
                        |
                   Context Gate (Linear + Sigmoid) -> [batch, vocab_size]
                        |
                 MLM Logits * Context Gate -> Gated Logits
                        |
                 ReLU + log(1+x) -> Max Pooling -> Sparse Vector
    """

    # Override with V28-specific configs
    model: V28ModelConfig = field(default_factory=V28ModelConfig)
    """Model configuration with context gate settings."""

    loss: V28LossConfig = field(default_factory=V28LossConfig)
    """Loss configuration with language filtering."""

    data: DataConfig = field(default_factory=lambda: DataConfig(
        train_files=["data/v24.0/train_*.jsonl"],
        val_files=["data/v24.0/val.jsonl"],
        batch_size=24,
        max_length=192,
        query_max_length=64,
        doc_max_length=256,
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
        output_dir="outputs/train_v28",
        experiment_name="splade_v28_xlmr_context_gated",
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
        # Convert nested dicts to V28-specific dataclasses
        if isinstance(self.model, dict):
            self.model = V28ModelConfig(**self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.loss, dict):
            self.loss = V28LossConfig(**self.loss)
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
                    description="Phase 3: Hard negative refinement with context gate",
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
        return f"{self.training.output_dir}/idf_weights/xlmr_v28_idf"


def create_default_v28_config(
    train_files: Optional[List[str]] = None,
    val_files: Optional[List[str]] = None,
    output_dir: str = "outputs/train_v28",
    model_name: str = "xlm-roberta-base",
    batch_size: int = 24,
    num_epochs: int = 25,
    idf_weights_path: Optional[str] = None,
    use_context_gate: bool = True,
    enable_language_filtering: bool = True,
) -> V28Config:
    """
    Create a default V28 configuration.

    Args:
        train_files: Training data files
        val_files: Validation data files
        output_dir: Output directory
        model_name: XLM-R model variant
        batch_size: Batch size
        num_epochs: Number of epochs
        idf_weights_path: Path to pre-computed IDF weights
        use_context_gate: Whether to use context-gated model
        enable_language_filtering: Whether to enable Korean filtering

    Returns:
        Configured V28Config instance
    """
    if train_files is None:
        train_files = ["data/v24.0/train_*.jsonl"]
    if val_files is None:
        val_files = ["data/v24.0/val.jsonl"]

    config = V28Config(
        model=V28ModelConfig(
            name=model_name,
            dropout=0.1,
            use_expansion=True,
            expansion_mode="mlm",
            use_context_gate=use_context_gate,
        ),
        data=DataConfig(
            train_files=train_files,
            val_files=val_files,
            batch_size=batch_size,
            max_length=192,
            query_max_length=64,
            doc_max_length=256,
            num_workers=4,
        ),
        loss=V28LossConfig(
            idf_weights_path=idf_weights_path,
            enable_language_filtering=enable_language_filtering,
            use_context_gate=use_context_gate,
        ),
        training=TrainingConfig(
            num_epochs=num_epochs,
            learning_rate=3e-5,
            output_dir=output_dir,
            experiment_name="splade_v28_xlmr_context_gated",
        ),
    )

    config.validate()
    return config
