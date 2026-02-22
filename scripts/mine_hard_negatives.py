#!/usr/bin/env python3
"""TF-IDF-based hard negative mining for V29 training data.

Phase 1: Collect unique positive documents from all shards (up to --max-corpus).
Phase 2: Build TF-IDF index on documents using char n-gram vectorizer.
Phase 3: For each sample without a negative, batch-query TF-IDF cosine
         similarity, pick best non-positive per query.

Performance notes:
  - Corpus TF-IDF rows are L2-pre-normalized so dot product = cosine similarity.
  - Corpus is processed in chunks to avoid materializing full (batch, n_corpus)
    dense matrices.
  - argpartition (O(n)) replaces argsort (O(n log n)) for top-k selection.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
    force=True,
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
    max_features: int,
) -> tuple[TfidfVectorizer, csr_matrix, list[str]]:
    """Fit TF-IDF vectorizer and transform corpus into L2-normalized sparse matrix.

    Uses character n-gram (2-3) features which handle Korean well without
    requiring a language-specific tokenizer. The corpus matrix is L2-normalized
    per row so that sparse dot product equals cosine similarity.

    Args:
        corpus: List of document strings.
        max_features: Maximum number of TF-IDF features.

    Returns:
        Tuple of (fitted TfidfVectorizer, L2-normalized corpus TF-IDF matrix,
        corpus list).
    """
    logger.info("Phase 2: Building TF-IDF index...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 3),
        max_features=max_features,
        sublinear_tf=True,
    )
    corpus_tfidf: csr_matrix = vectorizer.fit_transform(
        tqdm(corpus, desc="Fitting TF-IDF")
    )
    # Pre-normalize rows to L2 so dot product = cosine similarity
    corpus_tfidf = normalize(corpus_tfidf, norm="l2", axis=1, copy=False)

    logger.info(
        f"TF-IDF index built: {corpus_tfidf.shape[0]} docs, "
        f"{corpus_tfidf.shape[1]} features (L2-normalized)"
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


def _chunked_topk(
    q_tfidf: csr_matrix,
    corpus_tfidf: csr_matrix,
    top_k: int,
    corpus_chunk_size: int,
) -> np.ndarray:
    """Find top-k corpus indices per query using chunked dot product.

    Instead of computing a full (n_queries, n_corpus) dense matrix, process
    the corpus in chunks and keep only the top-k scores across all chunks.
    Uses argpartition (O(n)) instead of argsort (O(n log n)) for top-k.

    Both q_tfidf and corpus_tfidf must be L2-normalized so dot product equals
    cosine similarity.

    Args:
        q_tfidf: L2-normalized query matrix (n_queries, vocab).
        corpus_tfidf: L2-normalized corpus matrix (n_corpus, vocab).
        top_k: Number of top candidates per query.
        corpus_chunk_size: Number of corpus docs per chunk.

    Returns:
        Array of shape (n_queries, top_k) with corpus indices sorted by
        descending similarity.
    """
    n_queries = q_tfidf.shape[0]
    n_corpus = corpus_tfidf.shape[0]
    effective_k = min(top_k, n_corpus)

    # Accumulate top-k scores and indices across chunks
    best_scores = np.full((n_queries, effective_k), -np.inf, dtype=np.float32)
    best_indices = np.zeros((n_queries, effective_k), dtype=np.int64)

    for chunk_start in range(0, n_corpus, corpus_chunk_size):
        chunk_end = min(chunk_start + corpus_chunk_size, n_corpus)
        chunk_tfidf = corpus_tfidf[chunk_start:chunk_end]

        # Sparse dot product -> dense (n_queries, chunk_size)
        chunk_scores = (q_tfidf @ chunk_tfidf.T).toarray().astype(np.float32)
        chunk_size = chunk_end - chunk_start

        # Merge this chunk's results with running best
        # Concatenate current best with chunk results
        merged_scores = np.concatenate([best_scores, chunk_scores], axis=1)
        merged_indices = np.concatenate(
            [
                best_indices,
                np.arange(chunk_start, chunk_end, dtype=np.int64)[
                    np.newaxis, :
                ].repeat(n_queries, axis=0),
            ],
            axis=1,
        )

        # argpartition to get top-k from merged (O(n) per row)
        merge_k = min(effective_k, merged_scores.shape[1])
        if merged_scores.shape[1] > merge_k:
            part_idx = np.argpartition(-merged_scores, merge_k, axis=1)[
                :, :merge_k
            ]
        else:
            part_idx = np.arange(merged_scores.shape[1])[np.newaxis, :].repeat(
                n_queries, axis=0
            )

        # Gather top-k from merged
        row_idx = np.arange(n_queries)[:, np.newaxis]
        best_scores = merged_scores[row_idx, part_idx]
        best_indices = merged_indices[row_idx, part_idx]

    # Final sort within top-k by descending score
    final_order = np.argsort(-best_scores, axis=1)
    row_idx = np.arange(n_queries)[:, np.newaxis]
    return best_indices[row_idx, final_order]


def process_shard(
    shard_file: Path,
    vectorizer: TfidfVectorizer,
    corpus_tfidf: csr_matrix,
    corpus: list[str],
    top_k: int,
    batch_size: int,
    corpus_chunk_size: int,
    output_dir: Path | None,
    dry_run: bool,
) -> dict[str, int]:
    """Process a single shard: fill missing negatives via batched TF-IDF search.

    Queries needing a negative are grouped into batches of `batch_size`,
    transformed with the fitted vectorizer, L2-normalized, and scored against
    the corpus in chunks via sparse dot product. The top-k candidates per
    query are scanned to find the best non-positive document.

    Args:
        shard_file: Path to input shard file.
        vectorizer: Fitted TfidfVectorizer.
        corpus_tfidf: Pre-computed L2-normalized TF-IDF matrix for the corpus.
        corpus: Document corpus aligned with corpus_tfidf.
        top_k: Number of top candidates to retrieve per query.
        batch_size: Number of queries to process per TF-IDF batch.
        corpus_chunk_size: Number of corpus docs per chunk for dot product.
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

    if not need_negative_idx:
        logger.info(
            f"  {shard_file.name}: 0 needing negatives, skipping"
        )
        return stats

    # Process records needing negatives in batches
    n_batches = (len(need_negative_idx) + batch_size - 1) // batch_size
    for batch_num, batch_start in enumerate(
        range(0, len(need_negative_idx), batch_size)
    ):
        batch_indices = need_negative_idx[batch_start : batch_start + batch_size]
        batch_queries = [records[i].get("query", "") for i in batch_indices]
        batch_positives = [records[i].get("positive", "") for i in batch_indices]

        # Vectorize and L2-normalize batch queries
        q_tfidf: csr_matrix = vectorizer.transform(batch_queries)
        q_tfidf = normalize(q_tfidf, norm="l2", axis=1, copy=False)

        # Chunked top-k via sparse dot product
        top_indices = _chunked_topk(
            q_tfidf, corpus_tfidf, top_k, corpus_chunk_size
        )

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
        default=50_000,
        help="Maximum number of unique documents to index",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=30_000,
        help="Maximum number of TF-IDF features",
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
        "--corpus-chunk-size",
        type=int,
        default=10_000,
        help="Number of corpus docs per chunk for dot product computation",
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
    logger.info(
        f"Config: max_corpus={args.max_corpus}, max_features={args.max_features}, "
        f"corpus_chunk_size={args.corpus_chunk_size}, "
        f"batch_size={args.batch_size}, top_k={args.top_k}"
    )

    if args.dry_run:
        logger.info("DRY RUN mode: no files will be written")

    # Phase 1: build corpus
    corpus = build_corpus(shard_files, args.max_corpus)

    # Phase 2: build TF-IDF index
    vectorizer, corpus_tfidf, corpus = build_tfidf_index(
        corpus, max_features=args.max_features
    )

    # Phase 3: process shards
    logger.info("Phase 3: Mining hard negatives for samples without negatives...")
    all_stats: list[dict[str, int]] = []

    for shard_i, shard_file in enumerate(shard_files):
        t0 = time.monotonic()
        logger.info(
            f"[{shard_i + 1}/{len(shard_files)}] Processing {shard_file.name}..."
        )
        sys.stderr.flush()

        stats = process_shard(
            shard_file=shard_file,
            vectorizer=vectorizer,
            corpus_tfidf=corpus_tfidf,
            corpus=corpus,
            top_k=args.top_k,
            batch_size=args.batch_size,
            corpus_chunk_size=args.corpus_chunk_size,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
        all_stats.append(stats)

        elapsed = time.monotonic() - t0
        logger.info(
            f"  {shard_file.name}: total={stats['total']}, "
            f"added={stats['added']}, failed={stats['failed']}, "
            f"elapsed={elapsed:.1f}s"
        )
        sys.stderr.flush()

    print_stats_summary(all_stats, shard_files, args.dry_run)


if __name__ == "__main__":
    main()
