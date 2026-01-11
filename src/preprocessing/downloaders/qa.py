"""QA/MRC dataset downloaders (KorQuAD, KLUE-MRC)."""

import logging
from typing import Iterator

from datasets import load_dataset

from src.preprocessing.downloaders.base import BaseDownloader, RawSample

logger = logging.getLogger(__name__)


class KorQuADDownloader(BaseDownloader):
    """KorQuAD v1 dataset downloader.

    Dataset: KorQuAD/squad_kor_v1
    Size: 60,407 question-context-answer triplets
    Format: SQuAD-style extractive QA
    """

    dataset_name = "korquad"
    hf_path = "squad_kor_v1"
    expected_size = 60_407

    def download(self) -> None:
        """Download KorQuAD from HuggingFace."""
        logger.info(f"Downloading {self.dataset_name}...")
        self.dataset = load_dataset(self.hf_path, cache_dir=self.cache_dir)
        logger.info(f"Downloaded {self.get_stats()}")

    def iterate(self) -> Iterator[RawSample]:
        """Iterate over QA samples.

        Yields:
            RawSample with question as text1, context as text2,
            answer text as label
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not downloaded. Call download() first.")

        for split in self.dataset.keys():
            for item in self.dataset[split]:
                # Extract answer text (first answer if multiple)
                answers = item.get("answers", {})
                answer_texts = answers.get("text", [])
                answer_starts = answers.get("answer_start", [])

                answer_text = answer_texts[0] if answer_texts else ""
                answer_start = answer_starts[0] if answer_starts else -1

                yield RawSample(
                    text1=item["question"],
                    text2=item["context"],
                    label=answer_text,
                    source=self.dataset_name,
                    metadata={
                        "split": split,
                        "id": item.get("id", ""),
                        "title": item.get("title", ""),
                        "answer_start": answer_start,
                    },
                )


class KLUEMRCDownloader(BaseDownloader):
    """KLUE-MRC benchmark downloader.

    Dataset: klue (mrc subset)
    Size: ~17,554 question-paragraph pairs
    Format: Extractive QA with impossible questions
    """

    dataset_name = "klue_mrc"
    hf_path = "klue"
    hf_subset = "mrc"
    expected_size = 17_554

    def download(self) -> None:
        """Download KLUE-MRC from HuggingFace."""
        logger.info(f"Downloading {self.dataset_name}...")
        self.dataset = load_dataset(
            self.hf_path,
            self.hf_subset,
            cache_dir=self.cache_dir,
        )
        logger.info(f"Downloaded {self.get_stats()}")

    def iterate(self) -> Iterator[RawSample]:
        """Iterate over KLUE-MRC samples.

        Skips impossible questions (is_impossible=True).
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not downloaded. Call download() first.")

        for split in self.dataset.keys():
            for item in self.dataset[split]:
                # Skip impossible questions
                if item.get("is_impossible", False):
                    continue

                # Extract answer
                answers = item.get("answers", {})
                answer_texts = answers.get("text", [])
                answer_starts = answers.get("answer_start", [])

                answer_text = answer_texts[0] if answer_texts else ""
                answer_start = answer_starts[0] if answer_starts else -1

                # Get context (may be in 'context' or 'paragraph')
                context = item.get("context") or item.get("paragraph", "")

                yield RawSample(
                    text1=item["question"],
                    text2=context,
                    label=answer_text,
                    source=self.dataset_name,
                    metadata={
                        "split": split,
                        "guid": item.get("guid", ""),
                        "answer_start": answer_start,
                        "plausible_answers": item.get("plausible_answers", []),
                    },
                )
