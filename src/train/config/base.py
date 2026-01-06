"""
Base configuration dataclasses for SPLADE training.

Provides type-safe, validated configuration objects with sensible defaults.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = "skt/kobert-base-v1"
    """Pretrained transformer model name."""

    dropout: float = 0.1
    """Dropout rate for regularization."""

    use_expansion: bool = True
    """Whether to use MLM-based vocabulary expansion."""

    expansion_mode: Literal["mlm", "projection"] = "mlm"
    """Expansion mode: 'mlm' uses MLM head, 'projection' uses learned projection."""

    freeze_encoder_layers: int = 0
    """Number of encoder layers to freeze (0 = none)."""


@dataclass
class DataConfig:
    """Data configuration."""

    train_files: List[str] = field(default_factory=list)
    """List of training data file paths or glob patterns."""

    val_files: List[str] = field(default_factory=list)
    """List of validation data file paths or glob patterns."""

    batch_size: int = 64
    """Training batch size."""

    max_length: int = 64
    """Maximum sequence length for tokenization."""

    num_workers: int = 4
    """Number of data loading workers."""

    prefetch_factor: int = 2
    """Number of batches to prefetch per worker."""

    pin_memory: bool = True
    """Whether to pin memory for faster GPU transfer."""


@dataclass
class LossConfig:
    """Loss function configuration."""

    # Loss weights
    lambda_infonce: float = 2.0
    """Weight for InfoNCE contrastive loss."""

    lambda_self: float = 4.0
    """Weight for self-reconstruction loss."""

    lambda_positive: float = 10.0
    """Weight for positive activation loss."""

    lambda_margin: float = 2.5
    """Weight for triplet margin loss."""

    lambda_flops: float = 5e-3
    """Weight for FLOPS regularization."""

    lambda_min_act: float = 1.0
    """Weight for minimum activation loss."""

    # Loss hyperparameters
    temperature: float = 0.05
    """Temperature for InfoNCE softmax scaling."""

    margin: float = 1.5
    """Margin for triplet loss."""

    top_k: int = 5
    """Top-k for minimum activation loss."""

    min_activation: float = 0.5
    """Minimum activation threshold."""


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Basic training
    num_epochs: int = 30
    """Total number of training epochs."""

    learning_rate: float = 3e-6
    """Peak learning rate."""

    weight_decay: float = 0.01
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.1
    """Warmup ratio (fraction of total steps)."""

    gradient_clip: float = 1.0
    """Maximum gradient norm for clipping."""

    gradient_accumulation_steps: int = 1
    """Number of gradient accumulation steps."""

    # Mixed precision
    mixed_precision: Optional[Literal["fp16", "bf16"]] = "bf16"
    """Mixed precision mode: 'fp16', 'bf16', or None."""

    # Checkpointing
    save_every_n_epochs: int = 5
    """Save checkpoint every N epochs."""

    save_every_n_steps: Optional[int] = None
    """Save checkpoint every N steps (overrides epochs if set)."""

    keep_last_n_checkpoints: int = 3
    """Number of recent checkpoints to keep."""

    # Logging
    log_every_n_steps: int = 50
    """Log metrics every N steps."""

    eval_every_n_steps: Optional[int] = None
    """Evaluate on validation set every N steps."""

    # Output
    output_dir: str = "outputs/train_v22"
    """Directory for checkpoints and logs."""

    experiment_name: str = "splade_v22"
    """Name for TensorBoard experiment."""


@dataclass
class BaseConfig:
    """Base configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    """Model configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    """Data configuration."""

    loss: LossConfig = field(default_factory=LossConfig)
    """Loss function configuration."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    """Training configuration."""

    device: str = "cuda"
    """Device to use for training."""

    seed: int = 42
    """Random seed for reproducibility."""

    def validate(self) -> None:
        """Validate configuration values."""
        if self.data.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.data.max_length < 1:
            raise ValueError("max_length must be >= 1")
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.training.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if not self.data.train_files:
            raise ValueError("train_files must not be empty")

    @property
    def output_path(self) -> Path:
        """Get output directory as Path."""
        return Path(self.training.output_dir)

    def __post_init__(self) -> None:
        """Post-initialization validation."""
        # Convert nested dicts to dataclasses if needed
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.loss, dict):
            self.loss = LossConfig(**self.loss)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
