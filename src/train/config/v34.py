"""
V34 Configuration: V33 + expanded Korean data + extended training + relaxed query sparsity.

V34 builds directly on V33 (clean SPLADE-max with ModernBERT) with three changes:
- More training data: data/v34.0/ (expanded Korean corpus)
- More epochs: 30 (was 25) for better convergence on larger data
- Relaxed query sparsity: lambda_q=0.008 (was 0.01) to allow more query tokens

Architecture is identical to V33:
- Base model: skt/A.X-Encoder-base (ModernBERT, 50K vocab, 768 hidden)
- Pure SPLADE-max: MLM -> log(1+ReLU) -> max pool (SPLADE v2 paper)
- FLOPS regularization with quadratic lambda warmup (separate q/d)
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
class V34ModelConfig:
    """V34 model configuration."""

    name: str = "skt/A.X-Encoder-base"
    dropout: float = 0.1


@dataclass
class V34LossConfig:
    """V34 loss configuration: SPLADE v2 style."""

    lambda_q: float = 8e-3
    """Query FLOPS regularization weight (relaxed from 0.01 to allow more query tokens)."""

    lambda_d: float = 3e-3
    """Document FLOPS regularization weight."""

    temperature: float = 1.0
    """InfoNCE temperature (1.0 for sparse dot-product, not cosine)."""

    flops_warmup_steps: int = 25000
    """Quadratic warmup steps for FLOPS lambda (proportional to more data)."""

    lambda_kd: float = 0.0
    """Knowledge distillation weight (0 = disabled)."""

    kd_temperature: float = 1.0
    """KD temperature for softmax."""


@dataclass
class V34DataConfig:
    """V34 data configuration."""

    train_files: List[str] = field(
        default_factory=lambda: [
            "data/v34.0/train_shard_*.jsonl",
        ]
    )
    val_files: List[str] = field(
        default_factory=lambda: [
            "data/v34.0/val.jsonl",
        ]
    )
    batch_size: int = 64
    """Per-GPU batch size."""

    query_max_length: int = 64
    doc_max_length: int = 256
    num_workers: int = 4


@dataclass
class V34TrainingConfig:
    """V34 training configuration."""

    num_epochs: int = 30
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "bf16"
    output_dir: str = "outputs/train_v34"
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 5
    seed: int = 42


@dataclass
class V34Config:
    """
    V34 Configuration: V33 + expanded Korean data + extended training + relaxed query sparsity.

    Architecture:
        input -> A.X-Encoder-base -> MLM head -> logits [B, S, 50K]
                    -> log(1 + ReLU(logits)) -> max_pool -> sparse [B, 50K]

    Training: InfoNCE + FLOPS (quadratic warmup) on B200 x8

    Target metrics:
    - Ko-StrategyQA Recall@1 > 50%
    - Average active tokens < 200
    - Training stable (no collapse)
    """

    model: V34ModelConfig = field(default_factory=V34ModelConfig)
    loss: V34LossConfig = field(default_factory=V34LossConfig)
    data: V34DataConfig = field(default_factory=V34DataConfig)
    training: V34TrainingConfig = field(default_factory=V34TrainingConfig)

    def __post_init__(self) -> None:
        if isinstance(self.model, dict):
            self.model = V34ModelConfig(**self.model)
        if isinstance(self.loss, dict):
            self.loss = V34LossConfig(**self.loss)
        if isinstance(self.data, dict):
            self.data = V34DataConfig(**self.data)
        if isinstance(self.training, dict):
            self.training = V34TrainingConfig(**self.training)
