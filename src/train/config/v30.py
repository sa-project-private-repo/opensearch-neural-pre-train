"""
V30 Configuration: Back to V26 baseline with simplified loss.

V30 returns to the proven V26 architecture:
- No context gate (plain SPLADEDocV29 with max pooling)
- 4 active loss components (InfoNCE + FLOPS + Language + KD)
- Mined hard negatives for AI Hub data
- Max pooling (SPLADE v2 standard, proven in V26)

Rationale:
- V28's context gate added complexity without clear gains
- V29 experiments showed max pooling improvement but context gate unclear
- V30 isolates the pooling gain by removing context gate
- Simpler model = easier to debug and optimize
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from src.train.config.base import BaseConfig, DataConfig, TrainingConfig
from src.train.config.v22 import CurriculumPhase
from src.train.config.v24 import KDConfig, HardNegativeConfig
from src.train.config.v28 import V28LossConfig, V28ModelConfig


@dataclass
class V30LossConfig(V28LossConfig):
    """V30-specific loss configuration: simplified back to V26 baseline."""

    # Core losses (4 active components)
    lambda_infonce: float = 3.0
    """InfoNCE contrastive loss weight."""

    lambda_self: float = 0.0
    """Self-reconstruction loss weight (DISABLED)."""

    lambda_positive: float = 0.0
    """Positive pair alignment loss weight (DISABLED)."""

    lambda_margin: float = 0.0
    """Triplet margin loss weight (DISABLED)."""

    lambda_flops: float = 0.010
    """IDF-weighted FLOPS regularization (V26-style)."""

    lambda_min_act: float = 0.0
    """Minimum activation loss weight (DISABLED)."""

    lambda_kd: float = 1.0
    """Knowledge distillation loss weight (reduced from V28)."""

    lambda_language: float = 0.3
    """Korean language filtering loss weight."""

    non_korean_penalty: float = 5.0
    """Penalty for non-Korean tokens (reduced from V28's 100.0)."""

    # Architecture flags
    use_context_gate: bool = False
    """Disable context gate (back to V26 baseline)."""

    use_stopword_mask: bool = True
    """Enable stopword masking (V26 standard)."""

    stopword_penalty: float = 15.0
    """Penalty for stopword activations."""


@dataclass
class V30ModelConfig(V28ModelConfig):
    """V30-specific model configuration: SPLADEDocV29 without context gate."""

    model_class: Literal["SPLADEDocXLMR", "SPLADEDocV29"] = "SPLADEDocV29"
    """Model class - defaults to SPLADEDocV29 (no context gate)."""

    pooling: Literal["max", "sum"] = "max"
    """Pooling strategy - max pooling (SPLADE v2 standard, proven in V26)."""

    use_context_gate: bool = False
    """Disable context gate (V30 simplification)."""


@dataclass
class V30Config(BaseConfig):
    """
    V30 Configuration: Back to V26 baseline with simplified loss.

    Key features:
    - Plain SPLADEDocV29 with max pooling (no context gate)
    - 4 active loss components (InfoNCE + FLOPS + Language + KD)
    - IDF-weighted FLOPS regularization (V26-style)
    - Mined hard negatives for AI Hub data

    Architecture simplification:
        Document -> XLM-R -> Hidden States
                        |
                   MLM Head -> Token Logits
                        |
                 ReLU + log(1+x) -> Max Pooling -> Sparse Vector

    Target metrics:
    - Recall@1 > 45% (baseline for V30)
    - Korean token ratio > 90%
    - Average active tokens ~100
    """

    model: V30ModelConfig = field(default_factory=V30ModelConfig)
    """Model configuration - SPLADEDocV29 with max pooling, no context gate."""

    loss: V30LossConfig = field(default_factory=V30LossConfig)
    """Loss configuration - simplified 4-component loss."""

    data: DataConfig = field(
        default_factory=lambda: DataConfig(
            train_files=[
                "data/v24.0/train_*.jsonl",
                "data/aihub/processed/aihub_*_mined.jsonl",
            ],
            val_files=["data/v24.0/val.jsonl"],
            batch_size=32,
            max_length=192,
            num_workers=4,
        )
    )
    """Data configuration with AI Hub mined negatives."""

    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            num_epochs=15,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            gradient_clip=1.0,
            gradient_accumulation_steps=2,
            mixed_precision="bf16",
            output_dir="outputs/train_v30_ddp",
            experiment_name="splade_v30_xlmr_baseline",
            log_every_n_steps=50,
            save_every_n_epochs=3,
        )
    )
    """Training configuration."""

    knowledge_distillation: KDConfig = field(default_factory=KDConfig)
    """Knowledge distillation configuration."""

    hard_negative_mining: HardNegativeConfig = field(
        default_factory=HardNegativeConfig
    )
    """Hard negative mining configuration."""

    curriculum_phases: List[CurriculumPhase] = field(default_factory=list)
    """Curriculum learning phases."""

    enable_curriculum: bool = True
    """Whether to enable curriculum learning."""

    def __post_init__(self) -> None:
        """Initialize default curriculum phases if not provided."""
        if isinstance(self.model, dict):
            self.model = V30ModelConfig(**self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.loss, dict):
            self.loss = V30LossConfig(**self.loss)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.knowledge_distillation, dict):
            self.knowledge_distillation = KDConfig(**self.knowledge_distillation)
        if isinstance(self.hard_negative_mining, dict):
            self.hard_negative_mining = HardNegativeConfig(
                **self.hard_negative_mining
            )

        if not self.curriculum_phases and self.enable_curriculum:
            self.curriculum_phases = [
                CurriculumPhase(
                    start_epoch=1,
                    end_epoch=5,
                    temperature=0.08,
                    lambda_infonce=2.5,
                    lr_multiplier=1.0,
                    lambda_kd=1.5,
                    description="Phase 1: Foundation with teacher guidance",
                ),
                CurriculumPhase(
                    start_epoch=6,
                    end_epoch=10,
                    temperature=0.05,
                    lambda_infonce=3.0,
                    lr_multiplier=0.5,
                    lambda_kd=1.0,
                    description="Phase 2: Balanced training",
                ),
                CurriculumPhase(
                    start_epoch=11,
                    end_epoch=15,
                    temperature=0.04,
                    lambda_infonce=3.0,
                    lr_multiplier=0.25,
                    lambda_kd=0.5,
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
        return f"{self.training.output_dir}/idf_weights/xlmr_v30_idf"


def create_default_v30_config(
    train_files: Optional[List[str]] = None,
    val_files: Optional[List[str]] = None,
    output_dir: str = "outputs/train_v30_ddp",
    model_name: str = "xlm-roberta-base",
    batch_size: int = 32,
    num_epochs: int = 15,
    lambda_flops: float = 0.010,
) -> V30Config:
    """
    Create a default V30 configuration.

    Args:
        train_files: Training data files
        val_files: Validation data files
        output_dir: Output directory
        model_name: XLM-R model variant
        batch_size: Batch size per GPU
        num_epochs: Number of epochs
        lambda_flops: IDF-weighted FLOPS regularization weight

    Returns:
        Configured V30Config instance
    """
    if train_files is None:
        train_files = [
            "data/v24.0/train_*.jsonl",
            "data/aihub/processed/aihub_*_mined.jsonl",
        ]
    if val_files is None:
        val_files = ["data/v24.0/val.jsonl"]

    config = V30Config(
        model=V30ModelConfig(
            name=model_name,
            dropout=0.1,
            use_expansion=True,
            expansion_mode="mlm",
            use_context_gate=False,
            model_class="SPLADEDocV29",
            pooling="max",
        ),
        data=DataConfig(
            train_files=train_files,
            val_files=val_files,
            batch_size=batch_size,
            max_length=192,
            num_workers=4,
        ),
        loss=V30LossConfig(
            lambda_flops=lambda_flops,
            enable_language_filtering=True,
            use_context_gate=False,
        ),
        training=TrainingConfig(
            num_epochs=num_epochs,
            learning_rate=2e-5,
            output_dir=output_dir,
            experiment_name="splade_v30_xlmr_baseline",
        ),
    )

    config.validate()
    return config
