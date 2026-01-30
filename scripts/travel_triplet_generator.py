#!/usr/bin/env python3
"""Convert collected travel data to triplet format for V27 training.

This script processes raw travel data from various sources and converts
them into training triplets with hard negatives.

Usage:
    python scripts/travel_triplet_generator.py
    python scripts/travel_triplet_generator.py --input data/v27.0/raw --output data/v27.0
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document from raw data."""

    text: str
    title: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None
    source: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class TravelTriplet:
    """Travel domain triplet for training."""

    query: str
    positive: str
    negative: str
    pair_type: str
    difficulty: str
    source: str
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "positive": self.positive,
            "negative": self.negative,
            "pair_type": self.pair_type,
            "difficulty": self.difficulty,
            "source": self.source,
        }


class TravelTripletGenerator:
    """Generate triplets from collected travel data."""

    # Query templates for different content types
    QUERY_TEMPLATES = {
        "attraction": [
            "{title} 정보",
            "{title}",
            "{location} {category}",
            "{location} 관광지",
        ],
        "restaurant": [
            "{location} 맛집",
            "{location} 음식점 추천",
            "{location} 뭐 먹을까",
        ],
        "accommodation": [
            "{location} 숙소",
            "{location} 호텔 추천",
            "{location} 어디서 자지",
        ],
        "general": [
            "{location} 여행",
            "{location} 관광",
            "{location} 가볼만한 곳",
        ],
    }

    def __init__(
        self,
        input_dir: str = "data/v27.0/raw",
        output_dir: str = "data/v27.0",
        shard_size: int = 100000,
        val_ratio: float = 0.05,
        seed: int = 42,
    ):
        """Initialize generator.

        Args:
            input_dir: Directory with raw collected data
            output_dir: Directory to save triplets
            shard_size: Number of triplets per shard file
            val_ratio: Validation split ratio
            seed: Random seed
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.val_ratio = val_ratio
        self.seed = seed
        random.seed(seed)

        # Document index by location for hard negative mining
        self.docs_by_location: Dict[str, List[Document]] = defaultdict(list)
        self.docs_by_category: Dict[str, List[Document]] = defaultdict(list)
        self.all_docs: List[Document] = []

    def load_raw_data(self) -> int:
        """Load raw data from all sources.

        Returns:
            Total number of documents loaded
        """
        total = 0

        # Load Wikipedia articles
        wiki_file = self.input_dir / "wikipedia" / "travel_articles.jsonl"
        if wiki_file.exists():
            count = self._load_jsonl(wiki_file, "wikipedia")
            logger.info(f"Loaded {count:,} Wikipedia articles")
            total += count

        # Load Namuwiki articles
        namu_file = self.input_dir / "namuwiki" / "travel_articles.jsonl"
        if namu_file.exists():
            count = self._load_jsonl(namu_file, "namuwiki")
            logger.info(f"Loaded {count:,} Namuwiki articles")
            total += count

        # Load Korpora content
        korpora_file = self.input_dir / "korpora" / "travel_content.jsonl"
        if korpora_file.exists():
            count = self._load_jsonl(korpora_file, "korpora")
            logger.info(f"Loaded {count:,} Korpora samples")
            total += count

        logger.info(f"Total documents loaded: {total:,}")
        return total

    def _load_jsonl(self, path: Path, source: str) -> int:
        """Load documents from JSONL file."""
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    doc = Document(
                        text=data.get("text", ""),
                        title=data.get("title"),
                        location=data.get("location"),
                        category=data.get("category"),
                        source=source,
                        metadata=data.get("metadata", {}),
                    )

                    if len(doc.text) < 50:
                        continue

                    self.all_docs.append(doc)

                    if doc.location:
                        self.docs_by_location[doc.location].append(doc)
                    if doc.category:
                        self.docs_by_category[doc.category].append(doc)

                    count += 1
                except json.JSONDecodeError:
                    continue

        return count

    def _generate_query(self, doc: Document) -> str:
        """Generate query for a document."""
        category = doc.category or "general"
        templates = self.QUERY_TEMPLATES.get(
            category, self.QUERY_TEMPLATES["general"]
        )
        template = random.choice(templates)

        return template.format(
            title=doc.title or "",
            location=doc.location or "",
            category=category,
        ).strip()

    def _find_hard_negative(self, doc: Document) -> Optional[Document]:
        """Find hard negative (same category, different location)."""
        # First try: same category, different location
        if doc.location and doc.category:
            candidates = [
                d
                for d in self.docs_by_category.get(doc.category, [])
                if d.location and d.location != doc.location
            ]
            if candidates:
                return random.choice(candidates)

        # Second try: different location
        if doc.location:
            other_locs = [
                loc for loc in self.docs_by_location.keys()
                if loc != doc.location
            ]
            if other_locs:
                neg_loc = random.choice(other_locs)
                candidates = self.docs_by_location[neg_loc]
                if candidates:
                    return random.choice(candidates)

        # Fallback: random document
        if len(self.all_docs) > 1:
            candidates = [d for d in self.all_docs if d != doc]
            if candidates:
                return random.choice(candidates)

        return None

    def _determine_difficulty(
        self, doc: Document, neg_doc: Document
    ) -> str:
        """Determine triplet difficulty."""
        if doc.category == neg_doc.category:
            return "hard"
        elif doc.location and neg_doc.location:
            # Adjacent regions are harder
            adjacent_pairs = [
                ("서울", "인천"),
                ("서울", "경기"),
                ("부산", "경남"),
                ("광주", "전남"),
            ]
            for loc1, loc2 in adjacent_pairs:
                if (doc.location == loc1 and neg_doc.location == loc2) or (
                    doc.location == loc2 and neg_doc.location == loc1
                ):
                    return "medium"
        return "easy"

    def generate_triplets(self) -> Iterator[TravelTriplet]:
        """Generate triplets from loaded documents.

        Yields:
            TravelTriplet objects
        """
        for doc in self.all_docs:
            # Skip documents without text
            if not doc.text or len(doc.text) < 50:
                continue

            # Generate query
            query = self._generate_query(doc)
            if not query:
                continue

            # Find hard negative
            neg_doc = self._find_hard_negative(doc)
            if neg_doc is None:
                continue

            # Determine difficulty
            difficulty = self._determine_difficulty(doc, neg_doc)

            # Create triplet
            pair_type = f"travel_{doc.category or 'general'}"

            yield TravelTriplet(
                query=query,
                positive=doc.text[:1000],  # Truncate
                negative=neg_doc.text[:1000],
                pair_type=pair_type,
                difficulty=difficulty,
                source=doc.source,
                metadata={
                    "positive_location": doc.location,
                    "negative_location": neg_doc.location,
                },
            )

    def save(self) -> Tuple[List[Path], Path]:
        """Save generated triplets to shard files.

        Returns:
            Tuple of (train shard paths, val path)
        """
        # Load raw data if not already loaded
        if not self.all_docs:
            self.load_raw_data()

        # Generate all triplets
        triplets = list(self.generate_triplets())
        logger.info(f"Generated {len(triplets):,} triplets")

        # Shuffle
        random.shuffle(triplets)

        # Split train/val
        val_size = int(len(triplets) * self.val_ratio)
        val_triplets = triplets[:val_size]
        train_triplets = triplets[val_size:]

        logger.info(
            f"Train: {len(train_triplets):,}, Val: {len(val_triplets):,}"
        )

        # Save validation set
        val_path = self.output_dir / "val_travel.jsonl"
        self._save_jsonl(val_triplets, val_path)

        # Save training shards
        train_paths = []
        for i in range(0, len(train_triplets), self.shard_size):
            shard = train_triplets[i : i + self.shard_size]
            shard_num = (i // self.shard_size) + 1
            shard_path = self.output_dir / f"train_travel_shard_{shard_num:03d}.jsonl"
            self._save_jsonl(shard, shard_path)
            train_paths.append(shard_path)

        # Save metadata
        metadata = {
            "total_triplets": len(triplets),
            "train_triplets": len(train_triplets),
            "val_triplets": len(val_triplets),
            "num_shards": len(train_paths),
            "shard_size": self.shard_size,
            "sources": list(set(t.source for t in triplets)),
            "pair_types": list(set(t.pair_type for t in triplets)),
            "difficulty_distribution": {
                "easy": sum(1 for t in triplets if t.difficulty == "easy"),
                "medium": sum(1 for t in triplets if t.difficulty == "medium"),
                "hard": sum(1 for t in triplets if t.difficulty == "hard"),
            },
        }

        metadata_path = self.output_dir / "metadata_travel.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")

        return train_paths, val_path

    def _save_jsonl(self, triplets: List[TravelTriplet], path: Path):
        """Save triplets to JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for triplet in triplets:
                f.write(json.dumps(triplet.to_dict(), ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(triplets):,} triplets to {path}")


def load_template_triplets(input_dir: Path) -> List[TravelTriplet]:
    """Load pre-generated template triplets."""
    triplet_file = input_dir / "generated" / "travel_triplets.jsonl"
    triplets = []

    if triplet_file.exists():
        with open(triplet_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    triplets.append(
                        TravelTriplet(
                            query=data["query"],
                            positive=data["positive"],
                            negative=data["negative"],
                            pair_type=data.get("pair_type", "travel_template"),
                            difficulty=data.get("difficulty", "medium"),
                            source="template_generator",
                            metadata=data.get("metadata", {}),
                        )
                    )
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(triplets):,} template triplets")

    return triplets


def main():
    parser = argparse.ArgumentParser(
        description="Generate travel domain triplets for V27 training"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/v27.0/raw",
        help="Input directory with raw data",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/v27.0",
        help="Output directory",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100000,
        help="Number of triplets per shard",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--include-templates",
        action="store_true",
        help="Include template-generated triplets",
    )

    args = parser.parse_args()

    generator = TravelTripletGenerator(
        input_dir=args.input,
        output_dir=args.output,
        shard_size=args.shard_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Load and generate triplets
    generator.load_raw_data()
    train_paths, val_path = generator.save()

    # Optionally append template triplets
    if args.include_templates:
        template_triplets = load_template_triplets(Path(args.input))
        if template_triplets:
            if train_paths:
                # Append to last training shard
                with open(train_paths[-1], "a", encoding="utf-8") as f:
                    for triplet in template_triplets:
                        f.write(
                            json.dumps(triplet.to_dict(), ensure_ascii=False) + "\n"
                        )
                logger.info(
                    f"Appended {len(template_triplets):,} template triplets to {train_paths[-1]}"
                )
            else:
                # No existing shards, create new one from templates
                output_dir = Path(args.output)
                shard_path = output_dir / "train_travel_shard_001.jsonl"
                with open(shard_path, "w", encoding="utf-8") as f:
                    for triplet in template_triplets:
                        f.write(
                            json.dumps(triplet.to_dict(), ensure_ascii=False) + "\n"
                        )
                train_paths.append(shard_path)
                logger.info(
                    f"Created {shard_path} with {len(template_triplets):,} template triplets"
                )

    print("\n" + "=" * 50)
    print("Triplet Generation Complete")
    print("=" * 50)
    print(f"Train shards: {len(train_paths)}")
    for path in train_paths:
        print(f"  - {path}")
    print(f"Validation: {val_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
