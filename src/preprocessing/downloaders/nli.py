"""NLI dataset downloaders (KorNLI, KLUE-NLI)."""

import logging
from typing import Iterator

from datasets import load_dataset

from src.preprocessing.downloaders.base import BaseDownloader, RawSample

logger = logging.getLogger(__name__)


class KorNLIDownloader(BaseDownloader):
    """KorNLI dataset downloader.

    Dataset: kor_nli (Kakaobrain)
    Configs: multi_nli, snli, xnli
    Size: ~942,854 training pairs total
    Labels: entailment(0), neutral(1), contradiction(2)
    """

    dataset_name = "kor_nli"
    hf_path = "kor_nli"
    hf_configs = ["multi_nli", "snli", "xnli"]
    expected_size = 942_854

    def download(self) -> None:
        """Download all KorNLI configs from HuggingFace."""
        logger.info(f"Downloading {self.dataset_name}...")
        self.datasets = {}
        for config in self.hf_configs:
            logger.info(f"  Loading config: {config}")
            self.datasets[config] = load_dataset(
                self.hf_path,
                config,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        # Set dataset to first for get_stats compatibility
        self.dataset = self.datasets.get("multi_nli")
        logger.info(f"Downloaded {len(self.hf_configs)} configs")

    def iterate(self) -> Iterator[RawSample]:
        """Iterate over NLI samples from all configs.

        Yields:
            RawSample with premise as text1, hypothesis as text2
        """
        if not hasattr(self, "datasets") or not self.datasets:
            raise RuntimeError("Dataset not downloaded. Call download() first.")

        # Process all configs and splits
        for config_name, dataset in self.datasets.items():
            for split in dataset.keys():
                for item in dataset[split]:
                    # Label mapping: entailment=0, neutral=1, contradiction=2
                    label = item["label"]
                    if isinstance(label, str):
                        label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
                        label = label_map.get(label, 1)

                    yield RawSample(
                        text1=item["premise"],
                        text2=item["hypothesis"],
                        label=label,
                        source=f"{self.dataset_name}_{config_name}",
                        metadata={"split": split, "config": config_name},
                    )


class KLUENLIDownloader(BaseDownloader):
    """KLUE-NLI benchmark downloader.

    Dataset: klue (nli subset)
    Size: ~25,000 pairs
    Labels: entailment(0), neutral(1), contradiction(2)
    """

    dataset_name = "klue_nli"
    hf_path = "klue"
    hf_subset = "nli"
    expected_size = 24_998

    def download(self) -> None:
        """Download KLUE-NLI from HuggingFace."""
        logger.info(f"Downloading {self.dataset_name}...")
        self.dataset = load_dataset(
            self.hf_path,
            self.hf_subset,
            cache_dir=self.cache_dir,
        )
        logger.info(f"Downloaded {self.get_stats()}")

    def iterate(self) -> Iterator[RawSample]:
        """Iterate over KLUE-NLI samples."""
        if self.dataset is None:
            raise RuntimeError("Dataset not downloaded. Call download() first.")

        for split in self.dataset.keys():
            for item in self.dataset[split]:
                yield RawSample(
                    text1=item["premise"],
                    text2=item["hypothesis"],
                    label=item["label"],
                    source=self.dataset_name,
                    metadata={"split": split, "guid": item.get("guid", "")},
                )
