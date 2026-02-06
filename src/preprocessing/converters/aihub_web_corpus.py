"""AI Hub Dataset 624: 대규모 웹데이터 기반 한국어 말뭉치 converter."""

import logging
from typing import List

from src.preprocessing.converters.base import BaseConverter, Triplet
from src.preprocessing.downloaders.base import RawSample

logger = logging.getLogger(__name__)


class AIHubWebCorpusConverter(BaseConverter):
    """Convert AI Hub web corpus (dataset 624) to triplets.

    Dataset: 대규모 웹데이터 기반 한국어 말뭉치
    Size: 9.6GB (news articles, 1B+ words)

    Conversion strategy:
    - Query: Article title or first sentence
    - Positive: Article content
    - Negative: None (hard negative mining via BM25)
    """

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 512,
        max_query_length: int = 100,
    ):
        """Initialize converter.

        Args:
            min_length: Minimum text length
            max_length: Maximum document length
            max_query_length: Maximum query length
        """
        super().__init__(min_length=min_length, max_length=max_length)
        self.max_query_length = max_query_length

    def convert(self, samples: List[RawSample]) -> List[Triplet]:
        """Convert raw web corpus samples to triplets.

        Args:
            samples: List of RawSample with title and content fields

        Returns:
            List of Triplet objects
        """
        triplets = []

        for sample in samples:
            # For AI Hub datasets, text1 contains title, text2 contains content
            title = sample.text1.strip() if sample.text1 else ""
            content = sample.text2.strip() if sample.text2 else ""

            # Skip if content is too short
            if not self._validate_text(content):
                continue

            # Use title as query, or first sentence if no title
            if title and len(title) >= self.min_length:
                query = title
            else:
                # Extract first sentence as query
                sentences = content.split(".")
                if not sentences:
                    continue
                first_sentence = sentences[0].strip()
                if len(first_sentence) < self.min_length:
                    continue
                query = first_sentence

            # Truncate query
            if len(query) > self.max_query_length:
                query = query[: self.max_query_length]

            # Truncate content
            positive = self._truncate(content)

            # Get category from metadata if available
            category = sample.metadata.get("category", "")

            triplet = Triplet(
                query=query,
                positive=positive,
                negative=None,  # Hard negative mining needed
                pair_type="news_qa",
                difficulty="medium",
                source="aihub_624",
                metadata={"category": category},
            )

            triplets.append(triplet)

        logger.info(
            f"Web corpus conversion: {len(samples)} samples -> {len(triplets)} triplets"
        )

        return triplets

    def requires_mining(self) -> bool:
        """This converter requires hard negative mining."""
        return True
