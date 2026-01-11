"""Pipeline configuration for Korean dataset preprocessing."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PipelineConfig:
    """Configuration for the preprocessing pipeline."""

    # Output settings
    output_dir: str = "data/v24.0"
    cache_dir: str = "~/.cache/huggingface"

    # Sequence length
    max_seq_length: int = 192

    # Mining settings
    use_bge_m3_mining: bool = True
    negatives_per_sample: int = 5
    mining_batch_size: int = 256
    mining_min_score: float = 0.3
    mining_max_score: float = 0.85

    # Deduplication
    dedup_threshold: float = 0.9
    use_minhash: bool = True

    # Output sharding
    shard_size: int = 100_000
    train_val_split: float = 0.95

    # Dataset selection (empty means all)
    datasets: List[str] = field(default_factory=list)

    # Processing limits (None means no limit)
    max_samples_per_dataset: Optional[int] = None

    # Device settings
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Expand paths and validate config."""
        self.output_dir = str(Path(self.output_dir).expanduser())
        self.cache_dir = str(Path(self.cache_dir).expanduser())

    @property
    def output_path(self) -> Path:
        """Return output directory as Path."""
        return Path(self.output_dir)


# Default datasets to process
DEFAULT_DATASETS = [
    "kor_nli",
    "klue_nli",
    "kor_sts",
    "korquad",
    "klue_mrc",
    "nsmc",
    "ynat",
    "question_pair",
    "korean_instructions",
    "persona_chat",
]
