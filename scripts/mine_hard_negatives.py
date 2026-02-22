#!/usr/bin/env python3
"""TF-IDF-based hard negative mining for V29 training data.

Phase 1: Collect unique positive documents from all shards (up to --max-corpus).
Phase 2: Build TF-IDF index on documents using char n-gram vectorizer.
Phase 3: For each sample without a negative, batch-query TF-IDF cosine
         similarity, pick best non-positive per query.
"""

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_shard_range(shard_range: str, num_shards: int) -> list[int]:
    """Parse shard range string into list of shard indices.

    Args:
        shard_range: Range string like "0-10" or "5" or "all".
        num_shards: Total number of available shards.

    Returns:
        List of shard indices to process.
    """
    if shard_range == "all":
        return list(range(num_shards))
    if "-" in shard_range:
        start, end = shard_range.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(shard_range)]


def collect_shard_files(data_dir: Path, shard_range: str) -> list[Path]:
    """Collect shard file paths sorted by shard index.

    Args:
        data_dir: Directory containing shard files.
        shard_range: Range string for shard selection.

    Returns:
        Sorted list of shard file paths to process.
    """
    all_shards = sorted(data_dir.glob("train_shard_*.jsonl"))
    if not all_shards:
        raise FileNotFoundError(f"No train_shard_*.jsonl files in {data_dir}")

    indices = parse_shard_range(shard_range, len(all_shards))
    selected = []
    for idx in indices:
        if idx < len(all_shards):
            selected.append(all_shards[idx])
        else:
            logger.warning(
                f"Shard index {idx} out of range (max {len(all_shards) - 1})"
            )
    return selected


def build_corpus(shard_files: list[Path], max_corpus: int) -> list[str]:
    """Collect unique positive documents from shards up to max_corpus limit.

    Args:
        shard_files: List of shard file paths.
        max_corpus: Maximum number of unique documents to collect.

    Returns:
        List of unique positive document strings.
    """
    seen: set[str] = set()
    corpus: list[str] = []

    logger.info("Phase 1: Building corpus from positive documents...")
    for shard_file in tqdm(shard_files, desc="Collecting positives"):
        if len(corpus) >= max_corpus:
            break
        with open(shard_file, encoding="utf-8") as f:
            for line in f:
                if len(corpus) >= max_corpus:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                positive = record.get("positive", "")
                if positive and positive not in seen:
                    seen.add(positive)
                    corpus.append(positive)

    logger.info(f"Corpus size: {len(corpus)} unique documents")
    return corpus


def build_tfidf_index(
    corpus: list[str],
) -> tuple[TfidfVectorizer, csr_matrix, list[str]]:
    """Fit TF-IDF vectorizer and transform corpus into sparse matrix.

    Uses character n-gram (2-3) features which handle Korean well without
    requiring a language-specific tokenizer.

    Args:
        corpus: List of document strings.

    Returns:
        Tuple of (fitted TfidfVectorizer, corpus TF-IDF matrix, corpus list).
    """
    logger.info("Phase 2: Building TF-IDF index...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 3),
        max_features=100_000,
        sublinear_tf=True,
    )
    corpus_tfidf: csr_matrix = vectorizer.fit_transform(
        tqdm(corpus, desc="Fitting TF-IDF")
    )
    logger.info(
        f"TF-IDF index built: {corpus_tfidf.shape[0]} docs, "
        f"{corpus_tfidf.shape[1]} features"
    )
    return vectorizer, corpus_tfidf, corpus


def count_missing_negatives(shard_file: Path) -> tuple[int, int]:
    """Count total samples and samples missing negatives in a shard.

    Args:
        shard_file: Path to shard JSONL file.

    Returns:
        Tuple of (total_samples, missing_negatives_count).
    """
    total = 0
    missing = 0
    with open(shard_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if not record.get("negative"):
                missing += 1
    return total, missing


def process_shard(
    shard_file: Path,
    vectorizer: TfidfVectorizer,
    corpus_tfidf: csr_matrix,
    corpus: list[str],
    top_k: int,
    batch_size: int,
    output_dir: Path | None,
    dry_run: bool,
) -> dict[str, int]:
    """Process a single shard: fill missing negatives via batched TF-IDF search.

    Queries needing a negative are grouped into batches of `batch_size`,
    transformed with the fitted vectorizer, and scored against the full corpus
    in one cosine_similarity call.  The top-k candidates per query are scanned
    to find the best non-positive document.

    Args:
        shard_file: Path to input shard file.
        vectorizer: Fitted TfidfVectorizer.
        corpus_tfidf: Pre-computed TF-IDF matrix for the corpus (n_docs x vocab).
        corpus: Document corpus aligned with corpus_tfidf.
        top_k: Number of top candidates to retrieve per query.
        batch_size: Number of queries to process per TF-IDF batch.
        output_dir: Output directory (None = overwrite in place).
        dry_run: If True, skip writing.

    Returns:
        Dict with stats: total, already_had_negative, added, failed.
    """
    stats: dict[str, int] = {
        "total": 0,
        "already_had_negative": 0,
        "added": 0,
        "failed": 0,
    }

    # Read all records
    records: list[dict[str, Any]] = []
    with open(shard_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(record)

    stats["total"] = len(records)

    # Separate records that already have a negative
    need_negative_idx: list[int] = []
    for i, rec in enumerate(records):
        if rec.get("negative"):
            stats["already_had_negative"] += 1
        else:
            need_negative_idx.append(i)

    # Process records needing negatives in batches
    for batch_start in range(0, len(need_negative_idx), batch_size):
        batch_indices = need_negative_idx[batch_start : batch_start + batch_size]
        batch_queries = [records[i].get("query", "") for i in batch_indices]
        batch_positives = [records[i].get("positive", "") for i in batch_indices]

        # Vectorize batch queries
        q_tfidf: csr_matrix = vectorizer.transform(batch_queries)

        # Compute cosine similarity: (batch_size, n_corpus)
        scores: np.ndarray = cosine_similarity(q_tfidf, corpus_tfidf)

        # For each query find best non-positive from top-k candidates
        top_indices: np.ndarray = np.argsort(-scores, axis=1)[:, :top_k]

        for j, record_idx in enumerate(batch_indices):
            positive = batch_positives[j]
            negative: str | None = None

            for corpus_idx in top_indices[j]:
                candidate = corpus[corpus_idx]
                if candidate != positive:
                    negative = candidate
                    break

            if negative:
                records[record_idx]["negative"] = negative
                records[record_idx]["difficulty"] = "hard"
                stats["added"] += 1
            else:
                stats["failed"] += 1

    if dry_run:
        return stats

    # Determine output path
    if output_dir is not None:
        out_path = output_dir / shard_file.name
    else:
        out_path = shard_file  # overwrite in place

    # Atomic write via temp file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=out_path.parent, prefix=".tmp_", suffix=".jsonl"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_path, out_path)
    except Exception:
        os.unlink(tmp_path)
        raise

    return stats


def print_stats_summary(
    all_stats: list[dict[str, int]],
    shard_files: list[Path],
    dry_run: bool,
) -> None:
    """Print aggregated statistics across all shards.

    Args:
        all_stats: List of per-shard stat dicts.
        shard_files: Shard files processed (for labeling).
        dry_run: Whether this was a dry run.
    """
    total = sum(s["total"] for s in all_stats)
    already = sum(s["already_had_negative"] for s in all_stats)
    added = sum(s["added"] for s in all_stats)
    failed = sum(s["failed"] for s in all_stats)
    missing_before = added + failed

    coverage_before = (already / total * 100) if total else 0.0
    coverage_after = ((already + added) / total * 100) if total else 0.0

    mode = "[DRY RUN] " if dry_run else ""
    logger.info(f"{mode}=== Hard Negative Mining Summary ===")
    logger.info(f"  Shards processed:       {len(shard_files)}")
    logger.info(f"  Total samples:          {total:,}")
    logger.info(f"  Had negative already:   {already:,}")
    logger.info(f"  Missing negatives:      {missing_before:,}")
    logger.info(f"  Negatives added:        {added:,}")
    logger.info(f"  Failed to find:         {failed:,}")
    logger.info(f"  Coverage before:        {coverage_before:.1f}%")
    logger.info(f"  Coverage after:         {coverage_after:.1f}%")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TF-IDF hard negative mining for V29 training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/v29.0"),
        help="Directory containing train_shard_*.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for updated shards (default: overwrite in place)",
    )
    parser.add_argument(
        "--max-corpus",
        type=int,
        default=500_000,
        help="Maximum number of unique documents to index",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="TF-IDF search depth (top-k candidates to consider per query)",
    )
    parser.add_argument(
        "--shard-range",
        type=str,
        default="all",
        help='Shard range to process: "all", "0-10", or "5"',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of queries to batch per TF-IDF cosine similarity call",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview stats without writing output files",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for TF-IDF hard negative mining."""
    args = parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    shard_files = collect_shard_files(data_dir, args.shard_range)
    logger.info(f"Processing {len(shard_files)} shard(s) from {data_dir}")

    if args.dry_run:
        logger.info("DRY RUN mode: no files will be written")

    # Phase 1: build corpus
    corpus = build_corpus(shard_files, args.max_corpus)

    # Phase 2: build TF-IDF index
    vectorizer, corpus_tfidf, corpus = build_tfidf_index(corpus)

    # Phase 3: process shards
    logger.info("Phase 3: Mining hard negatives for samples without negatives...")
    all_stats: list[dict[str, int]] = []

    for shard_file in tqdm(shard_files, desc="Processing shards"):
        stats = process_shard(
            shard_file=shard_file,
            vectorizer=vectorizer,
            corpus_tfidf=corpus_tfidf,
            corpus=corpus,
            top_k=args.top_k,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
        all_stats.append(stats)
        logger.debug(
            f"{shard_file.name}: total={stats['total']}, "
            f"added={stats['added']}, failed={stats['failed']}"
        )

    print_stats_summary(all_stats, shard_files, args.dry_run)


if __name__ == "__main__":
    main()
