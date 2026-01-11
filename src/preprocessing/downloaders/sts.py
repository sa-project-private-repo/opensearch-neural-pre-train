"""STS dataset downloader (KorSTS)."""

import logging
from typing import Iterator

from datasets import load_dataset

from src.preprocessing.downloaders.base import BaseDownloader, RawSample

logger = logging.getLogger(__name__)


class KorSTSDownloader(BaseDownloader):
    """KorSTS dataset downloader.

    Dataset: dkoterwa/kor-sts (Kakaobrain translation)
    Size: 8,628 sentence pairs
    Labels: similarity score 0.0-5.0
    """

    dataset_name = "kor_sts"
    hf_path = "kor_nli"  # KorSTS is in kor_nli package
    hf_subset = "sts"
    expected_size = 8_628

    def download(self) -> None:
        """Download KorSTS from HuggingFace."""
        logger.info(f"Downloading {self.dataset_name}...")
        # KorSTS is part of kor_nli package
        try:
            self.dataset = load_dataset(
                "kor_nli",
                "sts",
                cache_dir=self.cache_dir,
            )
        except Exception:
            # Fallback to alternative path
            logger.info("Trying alternative path...")
            self.dataset = load_dataset(
                "dkoterwa/kor-sts",
                cache_dir=self.cache_dir,
            )
        logger.info(f"Downloaded {self.get_stats()}")

    def iterate(self) -> Iterator[RawSample]:
        """Iterate over STS samples.

        Yields:
            RawSample with sentence1 as text1, sentence2 as text2,
            similarity score as label
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not downloaded. Call download() first.")

        for split in self.dataset.keys():
            for item in self.dataset[split]:
                # Handle different field names
                sent1 = item.get("sentence1") or item.get("sent1", "")
                sent2 = item.get("sentence2") or item.get("sent2", "")
                score = item.get("score") or item.get("label", 0.0)

                # Normalize score to 0-1 range if needed
                if isinstance(score, (int, float)) and score > 1:
                    score = score / 5.0  # 0-5 -> 0-1

                yield RawSample(
                    text1=sent1,
                    text2=sent2,
                    label=float(score),
                    source=self.dataset_name,
                    metadata={"split": split},
                )
