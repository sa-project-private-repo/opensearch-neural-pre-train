"""
Benchmark configuration settings.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class BenchmarkConfig:
    """Configuration for OpenSearch retrieval benchmark."""

    # OpenSearch connection
    opensearch_host: str = "ltr-vector.awsbuddy.com"
    opensearch_port: int = 443
    opensearch_region: str = "us-east-1"
    use_ssl: bool = True
    verify_certs: bool = True

    # Index names
    bm25_index: str = "benchmark-bm25-v25"
    dense_index: str = "benchmark-dense-v25"
    sparse_index: str = "benchmark-sparse-v25"
    hybrid_index: str = "benchmark-hybrid-v25"

    # Model paths
    bge_m3_model: str = "BAAI/bge-m3"
    neural_sparse_path: Path = field(
        default_factory=lambda: Path("huggingface/v25")
    )

    # Data paths
    validation_triplets_path: Path = field(
        default_factory=lambda: Path("data/v24.0/val.jsonl")
    )

    # Neural sparse settings
    neural_sparse_max_length: int = 192  # V25 uses 192

    # Benchmark settings
    sample_size: int = 2000
    top_k: int = 10
    batch_size: int = 32

    # Device
    device: str = "cuda"

    @property
    def index_names(self) -> List[str]:
        """Return all index names."""
        return [
            self.bm25_index,
            self.dense_index,
            self.sparse_index,
            self.hybrid_index,
        ]

    def get_opensearch_url(self) -> str:
        """Return OpenSearch connection URL."""
        protocol = "https" if self.use_ssl else "http"
        return f"{protocol}://{self.opensearch_host}:{self.opensearch_port}"
