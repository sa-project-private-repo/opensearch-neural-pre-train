"""AI Hub Dataset 71828: AI 교관 데이터 converter."""

import logging
from typing import List

from src.preprocessing.converters.base import BaseConverter, Triplet
from src.preprocessing.downloaders.base import RawSample

logger = logging.getLogger(__name__)


class AIHubAIInstructorConverter(BaseConverter):
    """Convert AI Hub AI instructor (dataset 71828) to triplets.

    Dataset: AI 교관 데이터
    Size: 5.66GB (12K Q&A pairs)

    Conversion strategy:
    - Query: Question
    - Positive: Answer
    - Negative: None (hard negative mining)
    """

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 512,
    ):
        """Initialize converter.

        Args:
            min_length: Minimum text length
            max_length: Maximum text length
        """
        super().__init__(min_length=min_length, max_length=max_length)

    def convert(self, samples: List[RawSample]) -> List[Triplet]:
        """Convert raw AI instructor samples to triplets.

        Args:
            samples: List of RawSample with question/answer fields

        Returns:
            List of Triplet objects
        """
        triplets = []

        for sample in samples:
            # For AI instructor: text1 is question, text2 is answer
            question = sample.text1.strip() if sample.text1 else ""
            answer = sample.text2.strip() if sample.text2 else ""

            # Skip if either is too short
            if not question or len(question) < self.min_length:
                continue
            if not answer or len(answer) < self.min_length:
                continue

            # Truncate
            query = self._truncate(question)
            positive = self._truncate(answer)

            # Get category/domain from metadata if available
            category = sample.metadata.get("category", "general")

            triplet = Triplet(
                query=query,
                positive=positive,
                negative=None,
                pair_type="instruction_qa",
                difficulty="hard",
                source="aihub_71828",
                metadata={"category": category},
            )

            triplets.append(triplet)

        logger.info(
            f"AI instructor conversion: {len(samples)} samples -> {len(triplets)} triplets"
        )

        return triplets

    def requires_mining(self) -> bool:
        """This converter requires hard negative mining."""
        return True
