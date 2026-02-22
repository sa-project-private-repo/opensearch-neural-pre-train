"""
V31 Configuration for Unified Sparse + Dense Encoder (Issue #17).

Key features:
- UnifiedEncoder: shared XLM-RoBERTa backbone with sparse + dense heads
- Dense contrastive loss (InfoNCE) on CLS-pooled embeddings
- Dense knowledge distillation from teacher
- Freeze sparse head for initial epochs to warm up dense head
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from src.train.config.base import BaseConfig, DataConfig, TrainingConfig
from src.train.config.v22 import CurriculumPhase
from src.train.config.v24 import KDConfig, HardNegativeConfig
from src.train.config.v28 import V28Config, V28LossConfig, V28ModelConfig


@dataclass
class V31LossConfig(V28LossConfig):
    """V31-specific loss configuration with unified sparse + dense objectives."""

    # Dense contrastive loss (InfoNCE)
    alpha_dense_contrastive: float = 0.3
    """Weight for dense contrastive loss relative to sparse loss."""

    # Dense knowledge distillation
    beta_dense_kd: float = 0.1
    """Weight for dense knowledge distillation (MSE vs teacher scores)."""

    # Dense head temperature
    dense_temperature: float = 0.05
    """Temperature for dense InfoNCE contrastive loss."""

    # Dense head output dimension
    dense_output_size: int = 768
    """Output embedding dimension for dense head."""

    # Staged training: freeze sparse head initially to warm up dense head
    freeze_sparse_epochs: int = 2
    """Freeze sparse-specific parameters for first N epochs, warm up dense head."""


@dataclass
class V31ModelConfig(V28ModelConfig):
    """V31-specific model configuration for unified encoder."""

    model_class: Literal[
        "SPLADEDocXLMR",
        "SPLADEDocContextGated",
        "UnifiedEncoder",
    ] = "UnifiedEncoder"
    """Model class to use (V31 defaults to UnifiedEncoder)."""

    dense_output_size: int = 768
    """Dense head output embedding dimension."""


@dataclass
class V31Config(V28Config):
    """
    V31 Configuration for Unified Sparse + Dense Encoder.

    Extends V28Config with dual-head training:
    - Sparse path: context-gated SPLADE (same as V28)
    - Dense path: CLS-pooled, L2-normalized embedding

    Architecture:
        Input -> XLM-RoBERTa Backbone
                   |                   |
           Sparse Head             Dense Head
         (context-gated)         (CLS pooling)
               |                      |
         Sparse Repr            Dense Repr (L2-norm)
         [batch, vocab]         [batch, 768]

    Training:
        Phase 1 (freeze_sparse_epochs): warm up dense head only
        Phase 2 (remaining epochs): joint sparse + dense training
    """

    # Override with V31-specific configs
    model: V31ModelConfig = field(default_factory=V31ModelConfig)
    """Model configuration for unified encoder."""

    loss: V31LossConfig = field(default_factory=V31LossConfig)
    """Loss configuration with dense objectives."""

    def __post_init__(self) -> None:
        """Convert nested dicts to V31-specific dataclasses."""
        if isinstance(self.model, dict):
            self.model = V31ModelConfig(**self.model)
        if isinstance(self.loss, dict):
            self.loss = V31LossConfig(**self.loss)
        # Delegate remaining init to parent (handles data, training, etc.)
        super().__post_init__()

    def is_sparse_frozen(self, epoch: int) -> bool:
        """
        Return True if sparse head should be frozen at this epoch.

        Args:
            epoch: Current training epoch (1-indexed)

        Returns:
            True if epoch <= freeze_sparse_epochs
        """
        return epoch <= self.loss.freeze_sparse_epochs


def create_default_v31_config(
    train_files: Optional[List[str]] = None,
    val_files: Optional[List[str]] = None,
    output_dir: str = "outputs/train_v31",
    model_name: str = "xlm-roberta-base",
    batch_size: int = 32,
    num_epochs: int = 25,
    idf_weights_path: Optional[str] = None,
    dense_output_size: int = 768,
    alpha_dense_contrastive: float = 0.3,
    beta_dense_kd: float = 0.1,
    freeze_sparse_epochs: int = 2,
) -> "V31Config":
    """
    Create a default V31 unified encoder configuration.

    Args:
        train_files: Training data files
        val_files: Validation data files
        output_dir: Output directory
        model_name: XLM-R model variant
        batch_size: Batch size per GPU
        num_epochs: Number of training epochs
        idf_weights_path: Path to pre-computed IDF weights
        dense_output_size: Dense embedding output dimension
        alpha_dense_contrastive: Weight for dense contrastive loss
        beta_dense_kd: Weight for dense KD loss
        freeze_sparse_epochs: Epochs to freeze sparse head

    Returns:
        Configured V31Config instance
    """
    from src.train.config.base import DataConfig, TrainingConfig

    if train_files is None:
        train_files = ["data/v29.0/train_*.jsonl"]
    if val_files is None:
        val_files = ["data/v29.0/val.jsonl"]

    config = V31Config(
        model=V31ModelConfig(
            name=model_name,
            dropout=0.1,
            use_expansion=True,
            expansion_mode="mlm",
            use_context_gate=True,
            dense_output_size=dense_output_size,
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
        loss=V31LossConfig(
            idf_weights_path=idf_weights_path,
            enable_language_filtering=True,
            use_context_gate=True,
            alpha_dense_contrastive=alpha_dense_contrastive,
            beta_dense_kd=beta_dense_kd,
            dense_output_size=dense_output_size,
            freeze_sparse_epochs=freeze_sparse_epochs,
        ),
        training=TrainingConfig(
            num_epochs=num_epochs,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            gradient_clip=1.0,
            gradient_accumulation_steps=8,
            mixed_precision="bf16",
            output_dir=output_dir,
            experiment_name="splade_v31_unified_encoder",
        ),
    )

    return config
