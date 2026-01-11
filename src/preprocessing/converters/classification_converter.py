"""Classification dataset converter (NSMC, YNAT)."""

import logging
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from src.preprocessing.converters.base import BaseConverter, Triplet
from src.preprocessing.downloaders.base import RawSample

logger = logging.getLogger(__name__)


class ClassificationConverter(BaseConverter):
    """Convert classification data to triplets.

    Strategy:
    - Same class items -> positive pairs (semantic similarity)
    - Different class items -> negative pairs
    - Random sampling within classes to create pairs

    Works for both binary (NSMC sentiment) and multi-class (YNAT topic).
    """

    def __init__(
        self,
        max_pairs_per_class: int = 50000,
        use_random_negatives: bool = True,
        min_length: int = 5,
        max_length: int = 256,
    ):
        """Initialize classification converter.

        Args:
            max_pairs_per_class: Maximum pairs to generate per class
            use_random_negatives: Use random cross-class negatives
            min_length: Minimum text length
            max_length: Maximum text length
        """
        super().__init__(min_length, max_length)
        self.max_per_class = max_pairs_per_class
        self.random_neg = use_random_negatives

    def convert(self, samples: List[RawSample]) -> List[Triplet]:
        """Convert classification samples to triplets.

        Creates triplets where:
        - query = text from class A
        - positive = different text from same class A
        - negative = text from different class B
        """
        # Group by class label
        by_class: Dict[int, List[str]] = defaultdict(list)

        for sample in samples:
            if not self._validate_text(sample.text1):
                continue

            text = sample.text1.strip()
            label = sample.label

            by_class[label].append(text)

        # Log class distribution
        for label, texts in by_class.items():
            logger.info(f"  Class {label}: {len(texts)} samples")

        # Generate pairs within each class
        triplets = []
        class_list = list(by_class.keys())

        for label, texts in by_class.items():
            if len(texts) < 2:
                continue

            # Sample pairs within class
            pairs = self._sample_pairs(texts, self.max_per_class)

            for text1, text2 in pairs:
                neg = None

                if self.random_neg and len(class_list) > 1:
                    # Get negative from different class
                    neg_classes = [c for c in class_list if c != label]
                    neg_class = random.choice(neg_classes)
                    neg = random.choice(by_class[neg_class])

                # Determine pair_type based on source
                source = samples[0].source if samples else "classification"
                pair_type = f"{source}_{label}"

                triplets.append(
                    Triplet(
                        query=text1,
                        positive=text2,
                        negative=neg,
                        pair_type=pair_type,
                        difficulty="easy" if neg else "unknown",
                        source=source,
                    )
                )

        logger.info(
            f"Classification conversion: {len(samples)} samples -> "
            f"{len(triplets)} triplets"
        )

        return triplets

    def _sample_pairs(
        self,
        texts: List[str],
        max_pairs: int,
    ) -> List[Tuple[str, str]]:
        """Sample pairs from a list of texts.

        Uses random sampling to avoid O(n^2) all-pairs.

        Args:
            texts: List of texts in same class
            max_pairs: Maximum pairs to generate

        Returns:
            List of (text1, text2) pairs
        """
        n = len(texts)
        if n < 2:
            return []

        # Calculate number of pairs possible
        total_pairs = n * (n - 1) // 2

        if total_pairs <= max_pairs:
            # Generate all pairs
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((texts[i], texts[j]))
            return pairs

        # Random sampling
        pairs = set()
        attempts = 0
        max_attempts = max_pairs * 3

        while len(pairs) < max_pairs and attempts < max_attempts:
            i, j = random.sample(range(n), 2)
            pair = (min(i, j), max(i, j))  # Canonical order
            if pair not in pairs:
                pairs.add(pair)
            attempts += 1

        return [(texts[i], texts[j]) for i, j in pairs]

    def requires_mining(self) -> bool:
        """Classification can provide negatives from different classes.

        But mining can improve quality with harder negatives.
        """
        return True
