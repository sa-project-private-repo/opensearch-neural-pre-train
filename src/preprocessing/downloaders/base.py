"""Base classes for dataset downloaders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional


@dataclass
class RawSample:
    """Raw sample from any dataset.

    Attributes:
        text1: Primary text (query/premise/question)
        text2: Secondary text (positive/hypothesis/context)
        label: Task-specific label (int/float/str)
        source: Dataset source identifier
        metadata: Additional metadata
    """

    text1: str
    text2: str
    label: Any
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDownloader(ABC):
    """Abstract base class for HuggingFace dataset downloaders.

    Subclasses must implement:
        - download(): Download dataset from HuggingFace
        - iterate(): Yield RawSample objects
        - get_stats(): Return dataset statistics
    """

    dataset_name: str = ""
    hf_path: str = ""
    hf_subset: Optional[str] = None
    expected_size: int = 0

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize downloader.

        Args:
            cache_dir: HuggingFace cache directory
        """
        self.cache_dir = cache_dir
        self.dataset = None

    @abstractmethod
    def download(self) -> None:
        """Download dataset from HuggingFace."""
        pass

    @abstractmethod
    def iterate(self) -> Iterator[RawSample]:
        """Iterate over raw samples.

        Yields:
            RawSample objects for each data point
        """
        pass

    def get_stats(self) -> Dict[str, int]:
        """Return dataset statistics.

        Returns:
            Dict with keys like 'total', 'train', 'val', etc.
        """
        if self.dataset is None:
            return {"total": 0}

        stats = {}
        for split in self.dataset.keys():
            stats[split] = len(self.dataset[split])
        stats["total"] = sum(stats.values())
        return stats

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset_name})"
