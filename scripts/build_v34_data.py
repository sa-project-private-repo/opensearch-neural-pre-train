#!/usr/bin/env python3
"""Build V34 training data: merge v29.0 shards + v34.0 raw, deduplicate, shard."""

import argparse
import glob
import json
import logging
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SOURCE_PATTERNS: list[tuple[str, str]] = [
    ("data/v29.0/train_shard_*.jsonl", "v29"),
    ("data/v34.0/raw/*.jsonl", "v34_new"),
]

_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{2,}")


def clean_text(text: str) -> str:
    """Normalize whitespace: collapse spaces and newlines."""
    text = text.strip()
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n", text)
    return text


def clean_sample(
    sample: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Clean and validate a single sample.

    Returns None if the sample should be skipped.
    """
    query = sample.get("query", "")
    positive = sample.get("positive", "")

    if not query or not positive:
        return None

    query = clean_text(query)
    positive = clean_text(positive)

    if len(query) < 10 or len(positive) < 10:
        return None

    if query == positive:
        return None

    return {
        "query": query,
        "positive": positive,
        "negative": sample.get("negative"),
        "pair_type": sample.get("pair_type", "unknown"),
        "difficulty": sample.get("difficulty", "medium"),
        "source": sample.get("source", "unknown"),
    }


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load a JSONL file, returning list of dicts."""
    samples: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line in %s", path)
    return samples


def load_all_sources(
    extra_sources: list[str],
) -> tuple[list[dict[str, Any]], Counter]:
    """Load and clean samples from all configured sources.

    Returns (cleaned_samples, per_source_counts).
    """
    all_samples: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()

    for pattern, label in SOURCE_PATTERNS:
        matched = sorted(glob.glob(pattern))
        if not matched:
            logger.info(
                "No files for pattern %s (%s)", pattern, label
            )
            continue
        count_before = len(all_samples)
        for filepath in matched:
            raw = load_jsonl(filepath)
            for item in raw:
                cleaned = clean_sample(item)
                if cleaned is not None:
                    all_samples.append(cleaned)
        added = len(all_samples) - count_before
        source_counts[label] = added
        logger.info(
            "  %s: %s samples from %d file(s)",
            label,
            f"{added:,}",
            len(matched),
        )

    for extra in extra_sources:
        matched = sorted(glob.glob(extra))
        if not matched:
            logger.warning("No files for extra source: %s", extra)
            continue
        count_before = len(all_samples)
        for filepath in matched:
            raw = load_jsonl(filepath)
            for item in raw:
                cleaned = clean_sample(item)
                if cleaned is not None:
                    all_samples.append(cleaned)
        added = len(all_samples) - count_before
        label = Path(matched[0]).stem
        source_counts[label] = added
        logger.info(
            "  extra(%s): %s samples", label, f"{added:,}"
        )

    logger.info(
        "Total loaded: %s samples", f"{len(all_samples):,}"
    )
    return all_samples, source_counts


def create_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Create MinHash from text using 5-gram char shingles."""
    m = MinHash(num_perm=num_perm)
    text_lower = text.lower()
    for i in range(len(text_lower) - 4):
        shingle = text_lower[i : i + 5]
        m.update(shingle.encode("utf-8"))
    return m


def deduplicate(
    samples: list[dict[str, Any]],
    threshold: float,
    num_perm: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deduplicate samples using MinHash LSH.

    Processes in batches of 100K for memory efficiency.
    Returns (deduplicated_samples, dedup_stats).
    """
    total_before = len(samples)
    logger.info(
        "Starting deduplication on %s samples "
        "(threshold=%.2f, num_perm=%d)",
        f"{total_before:,}",
        threshold,
        num_perm,
    )

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep_indices: set[int] = set()
    dup_count = 0
    batch_size = 100_000

    for batch_start in tqdm(
        range(0, total_before, batch_size),
        desc="Dedup batches",
    ):
        batch_end = min(batch_start + batch_size, total_before)
        for idx in range(batch_start, batch_end):
            s = samples[idx]
            text = s["query"] + " [SEP] " + s["positive"]
            mh = create_minhash(text, num_perm)

            result = lsh.query(mh)
            if result:
                dup_count += 1
                continue

            key = f"s_{idx}"
            try:
                lsh.insert(key, mh)
                keep_indices.add(idx)
            except ValueError:
                dup_count += 1

    deduped = [samples[i] for i in sorted(keep_indices)]

    stats = {
        "total_before": total_before,
        "duplicates_found": dup_count,
        "total_after": len(deduped),
    }
    logger.info(
        "Dedup complete: %s -> %s (removed %s duplicates)",
        f"{total_before:,}",
        f"{len(deduped):,}",
        f"{dup_count:,}",
    )
    return deduped, stats


def shard_and_write(
    samples: list[dict[str, Any]],
    output_dir: str,
    shard_size: int,
    val_ratio: float,
) -> tuple[int, int, int, int]:
    """Split into train/val and write shards.

    Returns (train_count, val_count, num_shards, total).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = len(samples)
    val_count = int(total * val_ratio)
    train_count = total - val_count

    train_samples = samples[:train_count]
    val_samples = samples[train_count:]

    val_path = out / "val.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(
        "Wrote %s val samples to %s", f"{val_count:,}", val_path
    )

    num_shards = 0
    for start in range(0, train_count, shard_size):
        num_shards += 1
        end = min(start + shard_size, train_count)
        shard_name = f"train_shard_{num_shards:03d}.jsonl"
        shard_path = out / shard_name
        with open(shard_path, "w", encoding="utf-8") as f:
            for item in train_samples[start:end]:
                line = json.dumps(item, ensure_ascii=False)
                f.write(line + "\n")
        shard_len = end - start
        logger.info(
            "Wrote shard %s: %s samples",
            shard_name,
            f"{shard_len:,}",
        )

    return train_count, val_count, num_shards, total


def write_metadata(
    output_dir: str,
    total: int,
    train_count: int,
    val_count: int,
    num_shards: int,
    samples: list[dict[str, Any]],
    dedup_stats: Optional[dict[str, int]],
    source_load_counts: Counter,
) -> None:
    """Write metadata.json with distribution stats.

    Args:
        output_dir: Output directory path.
        total: Total sample count after all processing.
        train_count: Training sample count.
        val_count: Validation sample count.
        num_shards: Number of training shards.
        samples: Final sample list for distribution stats.
        dedup_stats: Deduplication statistics dict.
        source_load_counts: Per-source counts at load time.
    """
    source_dist: Counter[str] = Counter()
    pair_type_dist: Counter[str] = Counter()

    for s in samples:
        source_dist[s.get("source", "unknown")] += 1
        pair_type_dist[s.get("pair_type", "unknown")] += 1

    metadata: dict[str, Any] = {
        "version": "v34.0",
        "total_samples": total,
        "train_samples": train_count,
        "val_samples": val_count,
        "num_shards": num_shards,
        "source_patterns": [
            {"pattern": p, "label": l} for p, l in SOURCE_PATTERNS
        ],
        "source_load_counts": dict(source_load_counts),
        "source_distribution": dict(source_dist.most_common()),
        "pair_type_distribution": dict(pair_type_dist.most_common()),
        "dedup_stats": dedup_stats,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    meta_path = Path(output_dir) / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("Wrote metadata to %s", meta_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build V34 training data: merge v29.0 shards + v34.0 raw, "
            "deduplicate, shard."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/v34.0",
        help="Output directory (default: data/v34.0)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100_000,
        help="Samples per training shard (default: 100000)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation split ratio (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--skip-dedup",
        action="store_true",
        help="Skip deduplication step",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.8,
        help="Jaccard threshold for MinHash (default: 0.8)",
    )
    parser.add_argument(
        "--num-perm",
        type=int,
        default=128,
        help="MinHash permutations (default: 128)",
    )
    parser.add_argument(
        "--extra-sources",
        nargs="*",
        default=[],
        help="Additional JSONL glob patterns to include",
    )
    return parser.parse_args()


def main() -> None:
    """Run the V34 data build pipeline."""
    args = parse_args()

    random.seed(args.seed)
    logger.info("=== V34 Data Build Pipeline ===")
    logger.info("Output: %s", args.output_dir)
    logger.info("Seed: %d", args.seed)
    logger.info(
        "Sources: %s",
        [p for p, _ in SOURCE_PATTERNS],
    )

    # Step 1: Load all sources
    logger.info("--- Step 1: Loading sources ---")
    samples, source_load_counts = load_all_sources(args.extra_sources)

    if not samples:
        logger.error("No samples loaded. Exiting.")
        sys.exit(1)

    # Step 2: Cleaning is done during loading
    logger.info("--- Step 2: Cleaning done during load ---")

    # Step 3: Deduplication
    dedup_stats: Optional[dict[str, int]] = None
    if args.skip_dedup:
        logger.info("--- Step 3: Skipping dedup (--skip-dedup) ---")
    else:
        logger.info("--- Step 3: MinHash deduplication ---")
        samples, dedup_stats = deduplicate(
            samples,
            threshold=args.dedup_threshold,
            num_perm=args.num_perm,
        )

    # Step 4: Shuffle
    logger.info("--- Step 4: Shuffling ---")
    random.shuffle(samples)
    logger.info("Shuffled %s samples", f"{len(samples):,}")

    # Step 5 & 6: Split and shard
    logger.info("--- Step 5-6: Split and shard ---")
    train_count, val_count, num_shards, total = shard_and_write(
        samples,
        args.output_dir,
        args.shard_size,
        args.val_ratio,
    )

    # Step 7: Metadata
    logger.info("--- Step 7: Writing metadata ---")
    write_metadata(
        args.output_dir,
        total,
        train_count,
        val_count,
        num_shards,
        samples,
        dedup_stats,
        source_load_counts,
    )

    logger.info("=== Pipeline complete ===")
    logger.info(
        "  Train: %s (%d shards)",
        f"{train_count:,}",
        num_shards,
    )
    logger.info("  Val: %s", f"{val_count:,}")
    logger.info("  Total: %s", f"{total:,}")


if __name__ == "__main__":
    main()
