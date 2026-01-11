"""STS dataset converter."""

import logging
import random
from typing import List, Tuple

from src.preprocessing.converters.base import BaseConverter, Triplet
from src.preprocessing.downloaders.base import RawSample

logger = logging.getLogger(__name__)


class STSConverter(BaseConverter):
    """Convert STS pairs based on similarity scores.

    Strategy:
    - High similarity (>0.7): Use as positive pairs
    - Low similarity (<0.3): Use as negative pairs
    - Create triplets by pairing high similarity with low similarity

    Note: STS scores are normalized to 0-1 range.
    """

    def __init__(
        self,
        positive_threshold: float = 0.7,
        negative_threshold: float = 0.3,
        min_length: int = 5,
        max_length: int = 256,
    ):
        """Initialize STS converter.

        Args:
            positive_threshold: Minimum similarity for positive pairs (0-1)
            negative_threshold: Maximum similarity for negative pairs (0-1)
            min_length: Minimum text length
            max_length: Maximum text length
        """
        super().__init__(min_length, max_length)
        self.pos_threshold = positive_threshold
        self.neg_threshold = negative_threshold

    def convert(self, samples: List[RawSample]) -> List[Triplet]:
        """Convert STS samples to triplets.

        Creates triplets where:
        - query = sentence1 from high similarity pair
        - positive = sentence2 from same pair
        - negative = sentence from low similarity pair with same query
        """
        # Collect positive and negative pairs
        positives: List[Tuple[str, str]] = []
        negatives: List[Tuple[str, str]] = []

        for sample in samples:
            if not self._validate_text(sample.text1):
                continue
            if not self._validate_text(sample.text2):
                continue

            sent1 = sample.text1.strip()
            sent2 = sample.text2.strip()
            score = float(sample.label)

            if score >= self.pos_threshold:
                positives.append((sent1, sent2))
            elif score <= self.neg_threshold:
                negatives.append((sent1, sent2))

        # Build negative pool for random sampling
        negative_pool = []
        for s1, s2 in negatives:
            negative_pool.extend([s1, s2])

        # Create triplets
        triplets = []
        for query, positive in positives:
            neg = None
            difficulty = "easy"

            # Find negative from low similarity pairs
            if negative_pool:
                # Try to find a negative that's not the same as positive
                candidates = [n for n in negative_pool if n != positive and n != query]
                if candidates:
                    neg = random.choice(candidates)
                    difficulty = "medium"

            triplets.append(
                Triplet(
                    query=query,
                    positive=positive,
                    negative=neg,
                    pair_type="sts_similarity",
                    difficulty=difficulty,
                    source="sts",
                )
            )

            # Also create reverse pair (sent2 as query)
            if random.random() < 0.5:  # 50% chance to avoid duplicates
                neg2 = None
                if negative_pool:
                    candidates = [
                        n for n in negative_pool if n != query and n != positive
                    ]
                    if candidates:
                        neg2 = random.choice(candidates)

                triplets.append(
                    Triplet(
                        query=positive,
                        positive=query,
                        negative=neg2,
                        pair_type="sts_similarity",
                        difficulty=difficulty,
                        source="sts",
                    )
                )

        logger.info(
            f"STS conversion: {len(samples)} samples -> {len(triplets)} triplets"
        )
        logger.info(f"  Positive pairs (>{self.pos_threshold}): {len(positives)}")
        logger.info(f"  Negative pairs (<{self.neg_threshold}): {len(negatives)}")

        return triplets

    def requires_mining(self) -> bool:
        """STS typically requires mining for hard negatives."""
        return True
