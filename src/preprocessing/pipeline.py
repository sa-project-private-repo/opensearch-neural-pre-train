"""Main preprocessing pipeline orchestrator."""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Type

from src.preprocessing.cleaners.deduplicator import ExactDeduplicator
from src.preprocessing.cleaners.text_cleaner import KoreanTextCleaner
from src.preprocessing.config import PipelineConfig
from src.preprocessing.converters.base import BaseConverter, Triplet
from src.preprocessing.converters.classification_converter import ClassificationConverter
from src.preprocessing.converters.dialog_converter import DialogConverter
from src.preprocessing.converters.nli_converter import NLIConverter
from src.preprocessing.converters.qa_converter import QAConverter
from src.preprocessing.converters.sts_converter import STSConverter
from src.preprocessing.downloaders.base import BaseDownloader
from src.preprocessing.downloaders.classification import NSMCDownloader, YNATDownloader
from src.preprocessing.downloaders.dialog import (
    KoreanInstructionsDownloader,
    PersonaChatDownloader,
)
from src.preprocessing.downloaders.nli import KLUENLIDownloader, KorNLIDownloader
from src.preprocessing.downloaders.qa import KLUEMRCDownloader, KorQuADDownloader
from src.preprocessing.downloaders.sts import KorSTSDownloader

logger = logging.getLogger(__name__)


# Registry of datasets and their converters
DATASET_REGISTRY: Dict[str, Dict] = {
    # NLI datasets
    "kor_nli": {
        "downloader": KorNLIDownloader,
        "converter": NLIConverter,
        "description": "Korean NLI (942K samples)",
    },
    "klue_nli": {
        "downloader": KLUENLIDownloader,
        "converter": NLIConverter,
        "description": "KLUE NLI (25K samples)",
    },
    # STS datasets
    "kor_sts": {
        "downloader": KorSTSDownloader,
        "converter": STSConverter,
        "description": "Korean STS (8.6K samples)",
    },
    # QA datasets
    "korquad": {
        "downloader": KorQuADDownloader,
        "converter": QAConverter,
        "description": "KorQuAD v1 (60K samples)",
    },
    "klue_mrc": {
        "downloader": KLUEMRCDownloader,
        "converter": QAConverter,
        "description": "KLUE MRC (17K samples)",
    },
    # Classification datasets
    "nsmc": {
        "downloader": NSMCDownloader,
        "converter": ClassificationConverter,
        "description": "NSMC Sentiment (200K samples)",
    },
    "ynat": {
        "downloader": YNATDownloader,
        "converter": ClassificationConverter,
        "description": "YNAT Topic Classification (45K samples)",
    },
    # Dialog datasets
    "korean_instructions": {
        "downloader": KoreanInstructionsDownloader,
        "converter": DialogConverter,
        "description": "Korean Instructions (200K samples)",
    },
    "persona_chat": {
        "downloader": PersonaChatDownloader,
        "converter": DialogConverter,
        "description": "Korean Persona Chat (10K samples)",
    },
}


class PreprocessingPipeline:
    """Main preprocessing pipeline.

    Orchestrates:
    1. Dataset download from HuggingFace
    2. Conversion to triplet format
    3. Text cleaning and normalization
    4. Hard negative mining (optional)
    5. Deduplication
    6. Sharding and output
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()

        # Initialize components
        self.cleaner = KoreanTextCleaner(
            min_korean_ratio=0.05,  # Relaxed for instructions
            max_special_ratio=0.4,
        )
        self.deduplicator = ExactDeduplicator()

        # Lazy loaded miner
        self._miner = None

        # Statistics
        self.stats: Dict[str, Dict] = {}

    @property
    def miner(self):
        """Lazy load BGE-M3 miner."""
        if self._miner is None and self.config.use_bge_m3_mining:
            from src.preprocessing.miners.bge_m3_miner import BGEM3HardNegativeMiner

            self._miner = BGEM3HardNegativeMiner(
                batch_size=self.config.mining_batch_size,
                max_length=self.config.max_seq_length,
            )
        return self._miner

    def run(
        self,
        datasets: Optional[List[str]] = None,
        skip_download: bool = False,
        skip_mining: bool = False,
    ) -> Path:
        """Run the full preprocessing pipeline.

        Args:
            datasets: List of dataset names to process (None = all)
            skip_download: Skip download if data exists
            skip_mining: Skip BGE-M3 hard negative mining

        Returns:
            Path to output directory
        """
        # Determine which datasets to process
        if datasets is None:
            datasets = list(DATASET_REGISTRY.keys())
        else:
            # Validate dataset names
            for name in datasets:
                if name not in DATASET_REGISTRY:
                    raise ValueError(
                        f"Unknown dataset: {name}. "
                        f"Available: {list(DATASET_REGISTRY.keys())}"
                    )

        logger.info(f"Processing {len(datasets)} datasets: {datasets}")

        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each dataset
        all_triplets: List[Triplet] = []

        for dataset_name in datasets:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Processing: {dataset_name}")
            logger.info(f"{'=' * 50}")

            triplets = self._process_dataset(dataset_name, skip_download)
            all_triplets.extend(triplets)

            self.stats[dataset_name] = {
                "raw_samples": triplets[0].metadata.get("raw_count", 0)
                if triplets
                else 0,
                "triplets": len(triplets),
            }

        logger.info(f"\nTotal triplets before dedup: {len(all_triplets)}")

        # Clean all triplets
        logger.info("Cleaning triplets...")
        cleaned_triplets = self._clean_triplets(all_triplets)
        logger.info(f"After cleaning: {len(cleaned_triplets)}")

        # Deduplicate
        logger.info("Deduplicating...")
        unique_triplets = self.deduplicator.deduplicate(cleaned_triplets)
        logger.info(f"After dedup: {len(unique_triplets)}")

        # Hard negative mining (optional)
        if self.config.use_bge_m3_mining and not skip_mining:
            logger.info("Mining hard negatives with BGE-M3...")
            unique_triplets = self._mine_negatives(unique_triplets)

        # Shuffle
        logger.info("Shuffling triplets...")
        random.shuffle(unique_triplets)

        # Split train/val
        split_idx = int(len(unique_triplets) * self.config.train_val_split)
        train_triplets = unique_triplets[:split_idx]
        val_triplets = unique_triplets[split_idx:]

        logger.info(f"Train: {len(train_triplets)}, Val: {len(val_triplets)}")

        # Save to shards
        self._save_shards(train_triplets, output_dir, "train")
        self._save_shards(val_triplets, output_dir, "val")

        # Save metadata
        self._save_metadata(output_dir, train_triplets, val_triplets)

        logger.info(f"\nPipeline complete! Output: {output_dir}")
        return output_dir

    def _process_dataset(
        self,
        dataset_name: str,
        skip_download: bool = False,
    ) -> List[Triplet]:
        """Process a single dataset.

        Args:
            dataset_name: Name of dataset
            skip_download: Skip download if exists

        Returns:
            List of triplets
        """
        registry = DATASET_REGISTRY[dataset_name]
        downloader_cls: Type[BaseDownloader] = registry["downloader"]
        converter_cls: Type[BaseConverter] = registry["converter"]

        # Download
        cache_dir = Path(self.config.cache_dir) / dataset_name
        downloader = downloader_cls(cache_dir=str(cache_dir))

        if not skip_download:
            downloader.download()
        else:
            logger.info(f"Skipping download for {dataset_name}")

        # Convert
        samples = list(downloader.iterate())
        logger.info(f"Downloaded {len(samples)} raw samples")

        converter = converter_cls()
        triplets = converter.convert(samples)

        # Add raw count to metadata for stats
        if triplets:
            triplets[0].metadata["raw_count"] = len(samples)

        return triplets

    def _clean_triplets(self, triplets: List[Triplet]) -> List[Triplet]:
        """Clean all triplets.

        Args:
            triplets: List of triplets

        Returns:
            Cleaned triplets
        """
        cleaned = []
        for triplet in triplets:
            clean_triplet = self.cleaner.clean_triplet(triplet)
            if clean_triplet:
                cleaned.append(clean_triplet)

        return cleaned

    def _mine_negatives(self, triplets: List[Triplet]) -> List[Triplet]:
        """Mine hard negatives for triplets.

        Args:
            triplets: List of triplets

        Returns:
            Triplets with mined negatives
        """
        if not self.miner:
            logger.warning("Miner not available, skipping mining")
            return triplets

        # Build index from all positives
        all_positives = list(set(t.positive for t in triplets))
        self.miner.build_index(all_positives)

        # Mine negatives
        return self.miner.mine_for_triplets(
            triplets,
            num_negatives=1,
            min_score=self.config.mining_min_score,
            max_score=self.config.mining_max_score,
            skip_complete=True,
        )

    def _save_shards(
        self,
        triplets: List[Triplet],
        output_dir: Path,
        prefix: str,
    ) -> List[Path]:
        """Save triplets to sharded JSONL files.

        Args:
            triplets: List of triplets
            output_dir: Output directory
            prefix: File prefix (train/val)

        Returns:
            List of shard file paths
        """
        shard_paths = []
        shard_size = self.config.shard_size

        for i in range(0, len(triplets), shard_size):
            shard_idx = i // shard_size + 1
            shard_triplets = triplets[i : i + shard_size]

            if prefix == "val":
                shard_path = output_dir / f"{prefix}.jsonl"
            else:
                shard_path = output_dir / f"{prefix}_shard_{shard_idx:03d}.jsonl"

            with open(shard_path, "w", encoding="utf-8") as f:
                for triplet in shard_triplets:
                    record = {
                        "query": triplet.query,
                        "positive": triplet.positive,
                        "negative": triplet.negative,
                        "pair_type": triplet.pair_type,
                        "difficulty": triplet.difficulty,
                        "source": triplet.source,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            shard_paths.append(shard_path)
            logger.info(f"Saved {len(shard_triplets)} triplets to {shard_path}")

            # Only one file for val
            if prefix == "val":
                break

        return shard_paths

    def _save_metadata(
        self,
        output_dir: Path,
        train_triplets: List[Triplet],
        val_triplets: List[Triplet],
    ) -> None:
        """Save pipeline metadata.

        Args:
            output_dir: Output directory
            train_triplets: Training triplets
            val_triplets: Validation triplets
        """
        # Count pair types
        pair_type_counts: Dict[str, int] = {}
        difficulty_counts: Dict[str, int] = {}
        source_counts: Dict[str, int] = {}

        for triplet in train_triplets + val_triplets:
            pair_type_counts[triplet.pair_type] = (
                pair_type_counts.get(triplet.pair_type, 0) + 1
            )
            difficulty_counts[triplet.difficulty] = (
                difficulty_counts.get(triplet.difficulty, 0) + 1
            )
            source_counts[triplet.source] = source_counts.get(triplet.source, 0) + 1

        # Complete triplets (have negatives)
        complete_train = sum(1 for t in train_triplets if t.negative)
        complete_val = sum(1 for t in val_triplets if t.negative)

        metadata = {
            "config": {
                "max_seq_length": self.config.max_seq_length,
                "use_bge_m3_mining": self.config.use_bge_m3_mining,
                "mining_min_score": self.config.mining_min_score,
                "mining_max_score": self.config.mining_max_score,
                "shard_size": self.config.shard_size,
                "train_val_split": self.config.train_val_split,
            },
            "statistics": {
                "train_total": len(train_triplets),
                "train_complete": complete_train,
                "val_total": len(val_triplets),
                "val_complete": complete_val,
                "pair_types": pair_type_counts,
                "difficulties": difficulty_counts,
                "sources": source_counts,
            },
            "dataset_stats": self.stats,
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved metadata to {metadata_path}")

    def list_datasets(self) -> Dict[str, str]:
        """List available datasets.

        Returns:
            Dict of dataset name to description
        """
        return {
            name: info["description"] for name, info in DATASET_REGISTRY.items()
        }
