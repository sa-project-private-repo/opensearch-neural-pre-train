"""Base classes for dataset converters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.preprocessing.downloaders.base import RawSample


@dataclass
class Triplet:
    """Standard triplet format for training.

    Attributes:
        query: Search query or question
        positive: Relevant/positive document
        negative: Non-relevant/negative document (may be None for mining)
        pair_type: Category label for the triplet
        difficulty: easy/medium/hard indicator
        source: Original dataset source
        metadata: Additional metadata
    """

    query: str
    positive: str
    negative: Optional[str] = None
    pair_type: str = "unknown"
    difficulty: str = "medium"
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL output."""
        return {
            "query": self.query,
            "positive": self.positive,
            "negative": self.negative,
            "pair_type": self.pair_type,
            "difficulty": self.difficulty,
            "source": self.source,
        }

    def is_complete(self) -> bool:
        """Check if triplet has all required fields."""
        return bool(self.query and self.positive and self.negative)


class BaseConverter(ABC):
    """Abstract base class for dataset-to-triplet converters.

    Subclasses must implement:
        - convert(): Convert RawSamples to Triplets
        - requires_mining(): Whether hard negative mining is needed
    """

    def __init__(
        self,
        min_length: int = 5,
        max_length: int = 512,
    ):
        """Initialize converter.

        Args:
            min_length: Minimum text length in characters
            max_length: Maximum text length in characters
        """
        self.min_length = min_length
        self.max_length = max_length

    @abstractmethod
    def convert(self, samples: List[RawSample]) -> List[Triplet]:
        """Convert raw samples to triplets.

        Args:
            samples: List of RawSample from downloader

        Returns:
            List of Triplet objects
        """
        pass

    @abstractmethod
    def requires_mining(self) -> bool:
        """Whether this converter needs hard negative mining.

        Returns:
            True if some triplets will have negative=None
        """
        pass

    def _validate_text(self, text: str) -> bool:
        """Check if text meets length requirements."""
        if not text:
            return False
        text_len = len(text.strip())
        return self.min_length <= text_len <= self.max_length

    def _truncate(self, text: str) -> str:
        """Truncate text to max_length."""
        if len(text) > self.max_length:
            return text[: self.max_length]
        return text
