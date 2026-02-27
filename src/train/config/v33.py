"""
V33 Configuration: Clean SPLADE-max with ModernBERT.

V33 is a clean break from V28-V32 (XLM-RoBERTa lineage):
- Base model: skt/A.X-Encoder-base (ModernBERT, 50K vocab, 768 hidden)
- Pure SPLADE-max architecture (MLM -> log(1+ReLU) -> max pool)
- SPLADE v2 loss: InfoNCE + FLOPS with quadratic lambda scheduler
- No context gate, no language filtering, no stopword masking

Rationale:
- 50K vocab (48.4% Korean) vs 250K XLM-R vocab eliminates need for
  language filtering; FLOPS alone can control sparsity
- ModernBERT: 22 layers, flash attention, 16K context length
- Following proven SPLADE v2 paper recipe exactly
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class V33ModelConfig:
    """V33 model configuration."""

    name: str = "skt/A.X-Encoder-base"
    dropout: float = 0.1


@dataclass
class V33LossConfig:
    """V33 loss configuration: SPLADE v2 style."""

    lambda_q: float = 1e-2
    """Query FLOPS regularization weight."""

    lambda_d: float = 3e-3
    """Document FLOPS regularization weight."""

    temperature: float = 1.0
    """InfoNCE temperature (1.0 for sparse dot-product, not cosine)."""

    flops_warmup_steps: int = 20000
    """Quadratic warmup steps for FLOPS lambda."""

    lambda_kd: float = 0.0
    """Knowledge distillation weight (0 = disabled)."""

    kd_temperature: float = 1.0
    """KD temperature for softmax."""


@dataclass
class V33DataConfig:
    """V33 data configuration."""

    train_files: List[str] = field(
        default_factory=lambda: [
            "data/v29.0/train_*.jsonl",
        ]
    )
    val_files: List[str] = field(
        default_factory=lambda: [
            "data/v29.0/val.jsonl",
        ]
    )
    batch_size: int = 64
    """Per-GPU batch size."""

    query_max_length: int = 64
    doc_max_length: int = 256
    num_workers: int = 4


@dataclass
class V33TrainingConfig:
    """V33 training configuration."""

    num_epochs: int = 25
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "bf16"
    output_dir: str = "outputs/train_v33"
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 5
    seed: int = 42


@dataclass
class V33Config:
    """
    V33 Configuration: Clean SPLADE-max with ModernBERT.

    Architecture:
        input -> A.X-Encoder-base -> MLM head -> logits [B, S, 50K]
                    -> log(1 + ReLU(logits)) -> max_pool -> sparse [B, 50K]

    Training: InfoNCE + FLOPS (quadratic warmup) on B200 x8

    Target metrics:
    - Ko-StrategyQA Recall@1 > 50%
    - Average active tokens < 200
    - Training stable (no collapse)
    """

    model: V33ModelConfig = field(default_factory=V33ModelConfig)
    loss: V33LossConfig = field(default_factory=V33LossConfig)
    data: V33DataConfig = field(default_factory=V33DataConfig)
    training: V33TrainingConfig = field(default_factory=V33TrainingConfig)

    def __post_init__(self) -> None:
        if isinstance(self.model, dict):
            self.model = V33ModelConfig(**self.model)
        if isinstance(self.loss, dict):
            self.loss = V33LossConfig(**self.loss)
        if isinstance(self.data, dict):
            self.data = V33DataConfig(**self.data)
        if isinstance(self.training, dict):
            self.training = V33TrainingConfig(**self.training)
