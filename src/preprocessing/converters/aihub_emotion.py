"""AI Hub Dataset 86: 감성 대화 말뭉치 converter."""

import logging
from typing import List

from src.preprocessing.converters.base import BaseConverter, Triplet
from src.preprocessing.downloaders.base import RawSample

logger = logging.getLogger(__name__)


class AIHubEmotionConverter(BaseConverter):
    """Convert AI Hub emotion dialog (dataset 86) to triplets.

    Dataset: 감성 대화 말뭉치
    Size: 20.3MB (270K emotional dialogues)

    Conversion strategy:
    - Query: User utterance
    - Positive: System response
    - Negative: None (in-batch negatives or mining)
    """

    def __init__(
        self,
        min_length: int = 5,
        max_length: int = 256,
    ):
        """Initialize converter.

        Args:
            min_length: Minimum text length
            max_length: Maximum text length
        """
        super().__init__(min_length=min_length, max_length=max_length)

    def convert(self, samples: List[RawSample]) -> List[Triplet]:
        """Convert raw emotion dialog samples to triplets.

        Args:
            samples: List of RawSample with user/system utterances

        Returns:
            List of Triplet objects
        """
        triplets = []

        for sample in samples:
            # For emotion dialog: text1 is user utterance, text2 is system response
            user_utterance = sample.text1.strip() if sample.text1 else ""
            system_response = sample.text2.strip() if sample.text2 else ""

            # Skip if either is too short
            if not user_utterance or len(user_utterance) < self.min_length:
                continue
            if not system_response or len(system_response) < self.min_length:
                continue

            # Truncate
            query = self._truncate(user_utterance)
            positive = self._truncate(system_response)

            # Get emotion label from label field or metadata
            emotion = "unknown"
            if isinstance(sample.label, str):
                emotion = sample.label
            elif "emotion" in sample.metadata:
                emotion = sample.metadata["emotion"]

            triplet = Triplet(
                query=query,
                positive=positive,
                negative=None,
                pair_type="dialog_qa",
                difficulty="easy",
                source="aihub_86",
                metadata={"emotion": emotion},
            )

            triplets.append(triplet)

        logger.info(
            f"Emotion dialog conversion: {len(samples)} samples -> {len(triplets)} triplets"
        )

        return triplets

    def requires_mining(self) -> bool:
        """This converter can use in-batch negatives but mining improves quality."""
        return True
