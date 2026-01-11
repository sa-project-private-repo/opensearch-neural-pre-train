"""Dialog dataset converter (Instructions, Persona Chat)."""

import logging
from typing import List

from src.preprocessing.converters.base import BaseConverter, Triplet
from src.preprocessing.downloaders.base import RawSample

logger = logging.getLogger(__name__)


class DialogConverter(BaseConverter):
    """Convert dialog pairs to triplets.

    Strategy:
    - User utterance -> query
    - Bot/assistant response -> positive
    - Negatives must be mined (responses to different queries)

    Works for both instruction-following and persona chat data.
    """

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 512,
        filter_short_responses: bool = True,
        min_response_length: int = 20,
    ):
        """Initialize dialog converter.

        Args:
            min_length: Minimum text length
            max_length: Maximum text length
            filter_short_responses: Filter very short responses
            min_response_length: Minimum response length if filtering
        """
        super().__init__(min_length, max_length)
        self.filter_short = filter_short_responses
        self.min_response = min_response_length

    def convert(self, samples: List[RawSample]) -> List[Triplet]:
        """Convert dialog samples to triplets.

        Creates triplets where:
        - query = user utterance / instruction
        - positive = assistant response
        - negative = None (must be mined)
        """
        triplets = []

        for sample in samples:
            user_text = sample.text1.strip()
            bot_text = sample.text2.strip()

            if not self._validate_text(user_text):
                continue
            if not self._validate_text(bot_text):
                continue

            # Filter short responses if enabled
            if self.filter_short and len(bot_text) < self.min_response:
                continue

            # Determine pair_type from label or source
            label = sample.label
            if isinstance(label, str):
                pair_type = label  # "instruction" or "dialog"
            else:
                pair_type = sample.source

            triplets.append(
                Triplet(
                    query=user_text,
                    positive=bot_text,
                    negative=None,  # Must be mined
                    pair_type=pair_type,
                    difficulty="medium",
                    source=sample.source,
                    metadata=sample.metadata,
                )
            )

        logger.info(
            f"Dialog conversion: {len(samples)} samples -> {len(triplets)} triplets"
        )

        return triplets

    def requires_mining(self) -> bool:
        """Dialog always requires mining for negatives."""
        return True
