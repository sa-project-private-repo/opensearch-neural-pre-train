"""Classification dataset downloaders (NSMC, YNAT)."""

import logging
from typing import Iterator

from datasets import load_dataset

from src.preprocessing.downloaders.base import BaseDownloader, RawSample

logger = logging.getLogger(__name__)


class NSMCDownloader(BaseDownloader):
    """NSMC (Naver Sentiment Movie Corpus) downloader.

    Dataset: e9t/nsmc
    Size: 200,000 movie reviews
    Labels: 0=negative, 1=positive
    """

    dataset_name = "nsmc"
    hf_path = "e9t/nsmc"
    expected_size = 200_000

    def download(self) -> None:
        """Download NSMC from HuggingFace."""
        logger.info(f"Downloading {self.dataset_name}...")
        self.dataset = load_dataset(
            self.hf_path, cache_dir=self.cache_dir, trust_remote_code=True
        )
        logger.info(f"Downloaded {self.get_stats()}")

    def iterate(self) -> Iterator[RawSample]:
        """Iterate over NSMC samples.

        Yields:
            RawSample with document as text1, empty text2,
            sentiment label (0/1)
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not downloaded. Call download() first.")

        for split in self.dataset.keys():
            for item in self.dataset[split]:
                # Skip samples with NaN or empty document
                doc = item.get("document", "")
                if not doc or doc != doc:  # NaN check
                    continue

                yield RawSample(
                    text1=str(doc),
                    text2="",  # Will be filled by converter with same-class pair
                    label=item.get("label", 0),
                    source=self.dataset_name,
                    metadata={
                        "split": split,
                        "id": item.get("id", ""),
                    },
                )


class YNATDownloader(BaseDownloader):
    """YNAT (KLUE Topic Classification) downloader.

    Dataset: klue (ynat subset)
    Size: 45,678 news headlines
    Labels: 7 categories (IT, economy, society, life, world, sports, politics)
    """

    dataset_name = "ynat"
    hf_path = "klue"
    hf_subset = "ynat"
    expected_size = 45_678

    # Label mapping for YNAT
    LABEL_NAMES = {
        0: "IT_science",
        1: "economy",
        2: "society",
        3: "life_culture",
        4: "world",
        5: "sports",
        6: "politics",
    }

    def download(self) -> None:
        """Download YNAT from HuggingFace."""
        logger.info(f"Downloading {self.dataset_name}...")
        self.dataset = load_dataset(
            self.hf_path,
            self.hf_subset,
            cache_dir=self.cache_dir,
        )
        logger.info(f"Downloaded {self.get_stats()}")

    def iterate(self) -> Iterator[RawSample]:
        """Iterate over YNAT samples.

        Yields:
            RawSample with title as text1, empty text2,
            topic label (0-6)
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not downloaded. Call download() first.")

        for split in self.dataset.keys():
            for item in self.dataset[split]:
                label = item.get("label", 0)
                label_name = self.LABEL_NAMES.get(label, "unknown")

                yield RawSample(
                    text1=item.get("title", ""),
                    text2="",
                    label=label,
                    source=self.dataset_name,
                    metadata={
                        "split": split,
                        "guid": item.get("guid", ""),
                        "label_name": label_name,
                        "url": item.get("url", ""),
                        "date": item.get("date", ""),
                    },
                )
