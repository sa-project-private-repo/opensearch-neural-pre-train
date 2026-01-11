"""NLI dataset converter."""

import logging
import random
from collections import defaultdict
from typing import Dict, List

from src.preprocessing.converters.base import BaseConverter, Triplet
from src.preprocessing.downloaders.base import RawSample

logger = logging.getLogger(__name__)


class NLIConverter(BaseConverter):
    """Convert NLI triplets to training format.

    Strategy:
    - Entailment (label=0): premise->query, hypothesis->positive
    - Contradiction (label=2): Use as hard negative for same premise
    - Neutral (label=1): Skip or use as soft negative

    This converter can produce complete triplets when premise has both
    entailment and contradiction hypotheses.
    """

    # NLI label mapping
    LABEL_ENTAILMENT = 0
    LABEL_NEUTRAL = 1
    LABEL_CONTRADICTION = 2

    def __init__(
        self,
        use_contradiction_as_negative: bool = True,
        include_neutral: bool = False,
        min_length: int = 5,
        max_length: int = 256,
    ):
        """Initialize NLI converter.

        Args:
            use_contradiction_as_negative: Use contradiction pairs as negatives
            include_neutral: Include neutral pairs as soft negatives
            min_length: Minimum text length
            max_length: Maximum text length
        """
        super().__init__(min_length, max_length)
        self.use_contradiction = use_contradiction_as_negative
        self.include_neutral = include_neutral

    def convert(self, samples: List[RawSample]) -> List[Triplet]:
        """Convert NLI samples to triplets.

        Groups samples by premise, then creates triplets with:
        - entailment hypothesis as positive
        - contradiction hypothesis as negative (when available)
        """
        # Group by premise
        premise_groups: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: {"entailment": [], "contradiction": [], "neutral": []}
        )

        for sample in samples:
            if not self._validate_text(sample.text1):
                continue
            if not self._validate_text(sample.text2):
                continue

            premise = sample.text1.strip()
            hypothesis = sample.text2.strip()
            label = sample.label

            if label == self.LABEL_ENTAILMENT:
                premise_groups[premise]["entailment"].append(hypothesis)
            elif label == self.LABEL_CONTRADICTION:
                premise_groups[premise]["contradiction"].append(hypothesis)
            elif label == self.LABEL_NEUTRAL and self.include_neutral:
                premise_groups[premise]["neutral"].append(hypothesis)

        # Create triplets
        triplets = []
        for premise, hyps in premise_groups.items():
            for pos in hyps["entailment"]:
                neg = None
                difficulty = "medium"

                if self.use_contradiction and hyps["contradiction"]:
                    # Use contradiction as hard negative
                    neg = random.choice(hyps["contradiction"])
                    difficulty = "hard"
                elif self.include_neutral and hyps["neutral"]:
                    # Use neutral as soft negative
                    neg = random.choice(hyps["neutral"])
                    difficulty = "medium"

                triplets.append(
                    Triplet(
                        query=premise,
                        positive=pos,
                        negative=neg,
                        pair_type="nli_entailment",
                        difficulty=difficulty,
                        source="nli",
                    )
                )

        logger.info(
            f"NLI conversion: {len(samples)} samples -> {len(triplets)} triplets"
        )
        complete = sum(1 for t in triplets if t.is_complete())
        logger.info(f"  Complete triplets (with negative): {complete}")

        return triplets

    def requires_mining(self) -> bool:
        """NLI partially requires mining.

        Some triplets will have negatives from contradictions,
        but others may need mining.
        """
        return True
