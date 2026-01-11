"""QA/MRC dataset converter."""

import logging
from typing import List

from src.preprocessing.converters.base import BaseConverter, Triplet
from src.preprocessing.downloaders.base import RawSample

logger = logging.getLogger(__name__)


class QAConverter(BaseConverter):
    """Convert QA pairs to query-document triplets.

    Strategy:
    - Question -> query
    - Context paragraph -> positive document
    - Negatives must be mined (no natural negatives in QA data)

    For long contexts, optionally truncates around the answer span.
    """

    def __init__(
        self,
        context_window: int = 512,
        include_answer_in_context: bool = True,
        min_length: int = 10,
        max_length: int = 512,
    ):
        """Initialize QA converter.

        Args:
            context_window: Maximum context length in characters
            include_answer_in_context: Ensure answer span is in truncated context
            min_length: Minimum text length
            max_length: Maximum text length
        """
        super().__init__(min_length, max_length)
        self.context_window = context_window
        self.include_answer = include_answer_in_context

    def convert(self, samples: List[RawSample]) -> List[Triplet]:
        """Convert QA samples to triplets.

        Creates triplets where:
        - query = question
        - positive = context (possibly truncated around answer)
        - negative = None (must be mined)
        """
        triplets = []

        for sample in samples:
            question = sample.text1.strip()
            context = sample.text2.strip()

            if not self._validate_text(question):
                continue
            if not self._validate_text(context):
                continue

            # Truncate context if needed
            if len(context) > self.context_window:
                context = self._truncate_context(
                    context,
                    sample.label,  # answer text
                    sample.metadata.get("answer_start", -1),
                )

            triplets.append(
                Triplet(
                    query=question,
                    positive=context,
                    negative=None,  # Must be mined
                    pair_type="qa_mrc",
                    difficulty="medium",
                    source="qa",
                    metadata={"answer": sample.label},
                )
            )

        logger.info(
            f"QA conversion: {len(samples)} samples -> {len(triplets)} triplets"
        )

        return triplets

    def _truncate_context(
        self,
        context: str,
        answer: str,
        answer_start: int,
    ) -> str:
        """Truncate context while keeping answer visible.

        Args:
            context: Full context text
            answer: Answer text
            answer_start: Answer start position (-1 if unknown)

        Returns:
            Truncated context
        """
        if len(context) <= self.context_window:
            return context

        if not self.include_answer or answer_start < 0:
            # Simple truncation from start
            return context[: self.context_window]

        # Center window around answer
        answer_end = answer_start + len(answer) if answer else answer_start

        # Calculate window boundaries
        half_window = self.context_window // 2
        start = max(0, answer_start - half_window)
        end = min(len(context), start + self.context_window)

        # Adjust if near end
        if end == len(context):
            start = max(0, end - self.context_window)

        return context[start:end]

    def requires_mining(self) -> bool:
        """QA always requires mining for negatives."""
        return True
