"""
Data loading and sampling for benchmark.
"""
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from benchmark.config import BenchmarkConfig

logger = logging.getLogger(__name__)


@dataclass
class Triplet:
    """Single triplet with query, positive, and negative."""

    anchor: str
    positive: str
    negative: str
    difficulty: str
    length_class: str
    pair_type: str


@dataclass
class BenchmarkData:
    """Prepared benchmark data."""

    queries: List[str]
    query_positive_ids: Dict[str, str]  # query -> positive doc_id
    documents: Dict[str, str]  # doc_id -> content
    doc_ids: List[str]


def load_triplets(
    path: Path,
    max_samples: Optional[int] = None,
) -> List[Triplet]:
    """
    Load triplets from JSONL file.

    Args:
        path: Path to triplets file
        max_samples: Maximum number of samples to load

    Returns:
        List of Triplet objects
    """
    triplets = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data = json.loads(line)
            triplets.append(
                Triplet(
                    anchor=data["anchor"],
                    positive=data["positive"],
                    negative=data["negative"],
                    difficulty=data.get("difficulty", "unknown"),
                    length_class=data.get("length_class", "unknown"),
                    pair_type=data.get("pair_type", "unknown"),
                )
            )
    logger.info(f"Loaded {len(triplets)} triplets from {path}")
    return triplets


def sample_triplets(
    triplets: List[Triplet],
    sample_size: int,
    stratify_by: str = "difficulty",
    seed: int = 42,
) -> List[Triplet]:
    """
    Sample triplets with stratification.

    Args:
        triplets: Full list of triplets
        sample_size: Number of samples to select
        stratify_by: Field to stratify by
        seed: Random seed

    Returns:
        Sampled triplets
    """
    random.seed(seed)

    if len(triplets) <= sample_size:
        logger.info(
            f"Sample size {sample_size} >= total {len(triplets)}, "
            "returning all triplets"
        )
        return triplets

    # Group by stratification field
    groups: Dict[str, List[Triplet]] = {}
    for t in triplets:
        key = getattr(t, stratify_by, "unknown")
        if key not in groups:
            groups[key] = []
        groups[key].append(t)

    # Calculate samples per group
    group_sizes = {k: len(v) for k, v in groups.items()}
    total = sum(group_sizes.values())
    samples_per_group = {
        k: max(1, int(sample_size * v / total))
        for k, v in group_sizes.items()
    }

    # Sample from each group
    sampled = []
    for key, group in groups.items():
        n = min(samples_per_group[key], len(group))
        sampled.extend(random.sample(group, n))

    # Adjust if we have too few or too many
    if len(sampled) < sample_size:
        remaining = [t for t in triplets if t not in sampled]
        additional = sample_size - len(sampled)
        sampled.extend(random.sample(remaining, min(additional, len(remaining))))
    elif len(sampled) > sample_size:
        sampled = random.sample(sampled, sample_size)

    logger.info(
        f"Sampled {len(sampled)} triplets (stratified by {stratify_by})"
    )
    return sampled


def prepare_benchmark_data(
    triplets: List[Triplet],
) -> BenchmarkData:
    """
    Prepare benchmark data from triplets.

    Args:
        triplets: List of triplets

    Returns:
        BenchmarkData with queries, documents, and mappings
    """
    documents: Dict[str, str] = {}
    queries: List[str] = []
    query_positive_ids: Dict[str, str] = {}

    for i, t in enumerate(triplets):
        # Use anchor as query
        query = t.anchor
        queries.append(query)

        # Create document IDs
        pos_id = f"doc_{i}_pos"
        neg_id = f"doc_{i}_neg"

        # Store documents
        documents[pos_id] = t.positive
        documents[neg_id] = t.negative

        # Map query to positive doc
        query_positive_ids[query] = pos_id

    logger.info(
        f"Prepared {len(queries)} queries, {len(documents)} documents"
    )
    return BenchmarkData(
        queries=queries,
        query_positive_ids=query_positive_ids,
        documents=documents,
        doc_ids=list(documents.keys()),
    )


def load_benchmark_data(config: BenchmarkConfig) -> BenchmarkData:
    """
    Load and prepare benchmark data from config.

    Args:
        config: Benchmark configuration

    Returns:
        Prepared benchmark data
    """
    triplets = load_triplets(config.validation_triplets_path)
    sampled = sample_triplets(triplets, config.sample_size)
    return prepare_benchmark_data(sampled)
