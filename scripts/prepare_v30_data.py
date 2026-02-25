#!/usr/bin/env python3
"""Prepare V30 training data.

Pipeline:
  1. Download new HuggingFace datasets:
     - daekeun-ml/naver-news-summarization-ko (22K news articles)
     - BCCard/BCCard-Finance-Kor-QnA (31K finance Q&A)
  2. Convert to triplet format
  3. Mine TF-IDF hard negatives using char n-gram vectorizer
  4. Copy v29.0 shards to data/v30.0/
  5. Append new triplets as new shard(s)
  6. Update metadata.json
  7. Create val.jsonl (1% split from new data)
"""

import json
import logging
import random
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
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

V29_DIR = Path("data/v29.0")
V30_DIR = Path("data/v30.0")
SHARD_SIZE = 100_000
VAL_RATIO = 0.01
SEED = 42
TFIDF_MAX_FEATURES = 100_000
TFIDF_TOP_K = 20
TFIDF_BATCH_SIZE = 1_000
CORPUS_CHUNK_SIZE = 10_000


def download_naver_news() -> list[dict[str, Any]]:
    """Download daekeun-ml/naver-news-summarization-ko and return all splits."""
    logger.info("Downloading daekeun-ml/naver-news-summarization-ko...")
    ds = load_dataset("daekeun-ml/naver-news-summarization-ko", trust_remote_code=True)

    rows: list[dict[str, Any]] = []
    for split_name, split_ds in ds.items():
        logger.info("  Split %s: %d rows", split_name, len(split_ds))
        for row in split_ds:
            rows.append(dict(row))

    logger.info("Total rows: %d", len(rows))
    return rows


def download_bccard() -> list[dict[str, Any]]:
    """Download BCCard/BCCard-Finance-Kor-QnA and return all rows."""
    logger.info("Downloading BCCard/BCCard-Finance-Kor-QnA...")
    ds = load_dataset("BCCard/BCCard-Finance-Kor-QnA", trust_remote_code=True)

    rows: list[dict[str, Any]] = []
    for split_name, split_ds in ds.items():
        logger.info("  Split %s: %d rows", split_name, len(split_ds))
        for row in split_ds:
            rows.append(dict(row))

    logger.info("Total BCCard rows: %d", len(rows))
    return rows


def build_bccard_triplets(
    rows: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Convert BCCard Q&A rows to triplet format.

    Args:
        rows: Raw dataset rows with instruction, output fields.
        rng: Seeded random instance.

    Returns:
        List of triplets.
    """
    valid: list[dict[str, Any]] = []
    for row in rows:
        instruction = (row.get("instruction") or "").strip()
        output = (row.get("output") or "").strip()
        if instruction and output:
            valid.append(row)

    logger.info("BCCard valid rows: %d", len(valid))

    outputs = [r["output"][:512] for r in valid]

    triplets: list[dict[str, Any]] = []
    for i, row in enumerate(tqdm(valid, desc="Building BCCard triplets")):
        query = row["instruction"].strip()
        positive = row["output"][:512].strip()

        neg_idx = rng.randint(0, len(outputs) - 1)
        if neg_idx == i:
            neg_idx = (neg_idx + 1) % len(outputs)
        negative = outputs[neg_idx]

        triplets.append(
            {
                "query": query,
                "positive": positive,
                "negative": negative,
                "pair_type": "finance_qa",
                "difficulty": "medium",
                "source": "bccard_finance_qna",
                "_category": "finance",
            }
        )

    logger.info("Built %d BCCard triplets", len(triplets))
    return triplets


def build_triplets(
    rows: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Convert raw rows to triplet format with random negatives.

    Args:
        rows: Raw dataset rows with title, document, category fields.
        rng: Seeded random instance for reproducibility.

    Returns:
        List of triplets with random negatives (pre-hard-negative-mining).
    """
    # Filter rows with required fields
    valid: list[dict[str, Any]] = []
    for row in rows:
        title = (row.get("title") or "").strip()
        document = (row.get("document") or "").strip()
        if title and document:
            valid.append(row)

    logger.info("Valid rows after filtering: %d", len(valid))

    documents = [r.get("document", "")[:512] for r in valid]

    triplets: list[dict[str, Any]] = []
    for i, row in enumerate(tqdm(valid, desc="Building triplets")):
        query = row.get("title", "").strip()
        positive = (row.get("document") or "")[:512].strip()

        # Random negative: pick a different document
        neg_idx = rng.randint(0, len(documents) - 1)
        if neg_idx == i:
            neg_idx = (neg_idx + 1) % len(documents)
        negative = documents[neg_idx]

        triplets.append(
            {
                "query": query,
                "positive": positive,
                "negative": negative,
                "pair_type": "news_title",
                "difficulty": "medium",
                "source": "naver_news_summarization",
                "_category": row.get("category", ""),
            }
        )

    logger.info("Built %d triplets", len(triplets))
    return triplets


def build_tfidf_index(
    documents: list[str],
    max_features: int,
) -> tuple[TfidfVectorizer, csr_matrix]:
    """Fit TF-IDF vectorizer on documents and return L2-normalized matrix.

    Args:
        documents: List of document strings to index.
        max_features: Maximum TF-IDF vocabulary size.

    Returns:
        Tuple of (fitted vectorizer, L2-normalized sparse matrix).
    """
    logger.info("Building TF-IDF index on %d documents...", len(documents))
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 3),
        max_features=max_features,
        sublinear_tf=True,
    )
    matrix: csr_matrix = vectorizer.fit_transform(
        tqdm(documents, desc="Fitting TF-IDF")
    )
    matrix = normalize(matrix, norm="l2", axis=1, copy=False)
    logger.info(
        "TF-IDF index: %d docs, %d features", matrix.shape[0], matrix.shape[1]
    )
    return vectorizer, matrix


def _chunked_topk(
    q_tfidf: csr_matrix,
    corpus_tfidf: csr_matrix,
    top_k: int,
    chunk_size: int,
) -> np.ndarray:
    """Find top-k corpus indices per query using chunked dot product.

    Args:
        q_tfidf: L2-normalized query matrix (n_queries, vocab).
        corpus_tfidf: L2-normalized corpus matrix (n_corpus, vocab).
        top_k: Number of top candidates per query.
        chunk_size: Corpus chunk size for memory efficiency.

    Returns:
        Array of shape (n_queries, top_k) with corpus indices.
    """
    n_queries = q_tfidf.shape[0]
    n_corpus = corpus_tfidf.shape[0]
    effective_k = min(top_k, n_corpus)

    best_scores = np.full((n_queries, effective_k), -np.inf, dtype=np.float32)
    best_indices = np.zeros((n_queries, effective_k), dtype=np.int64)

    for chunk_start in range(0, n_corpus, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_corpus)
        chunk = corpus_tfidf[chunk_start:chunk_end]
        chunk_scores = (q_tfidf @ chunk.T).toarray().astype(np.float32)

        merged_scores = np.concatenate([best_scores, chunk_scores], axis=1)
        merged_indices = np.concatenate(
            [
                best_indices,
                np.arange(chunk_start, chunk_end, dtype=np.int64)[np.newaxis, :].repeat(
                    n_queries, axis=0
                ),
            ],
            axis=1,
        )

        merge_k = min(effective_k, merged_scores.shape[1])
        if merged_scores.shape[1] > merge_k:
            part_idx = np.argpartition(-merged_scores, merge_k, axis=1)[:, :merge_k]
        else:
            part_idx = np.arange(merged_scores.shape[1])[np.newaxis, :].repeat(
                n_queries, axis=0
            )

        row_idx = np.arange(n_queries)[:, np.newaxis]
        best_scores = merged_scores[row_idx, part_idx]
        best_indices = merged_indices[row_idx, part_idx]

    final_order = np.argsort(-best_scores, axis=1)
    row_idx = np.arange(n_queries)[:, np.newaxis]
    return best_indices[row_idx, final_order]


def mine_hard_negatives(
    triplets: list[dict[str, Any]],
    vectorizer: TfidfVectorizer,
    corpus_tfidf: csr_matrix,
    documents: list[str],
    top_k: int,
    batch_size: int,
    chunk_size: int,
) -> list[dict[str, Any]]:
    """Replace random negatives with TF-IDF hard negatives.

    For each triplet, finds the most similar document from a different
    category as the hard negative.

    Args:
        triplets: Triplets with random negatives.
        vectorizer: Fitted TF-IDF vectorizer.
        corpus_tfidf: L2-normalized corpus matrix.
        documents: Document list aligned with corpus_tfidf.
        top_k: Candidate depth for hard negative search.
        batch_size: Query batch size for TF-IDF.
        chunk_size: Corpus chunk size for dot product.

    Returns:
        Triplets with hard negatives where found; medium difficulty
        otherwise retained.
    """
    logger.info(
        "Phase 3: Mining hard negatives for %d triplets...", len(triplets)
    )

    categories = [t.get("_category", "") for t in triplets]
    positives = [t["positive"] for t in triplets]
    updated = [dict(t) for t in triplets]
    added = 0
    failed = 0

    for batch_start in tqdm(
        range(0, len(triplets), batch_size), desc="Mining hard negatives"
    ):
        batch_end = min(batch_start + batch_size, len(triplets))
        batch_queries = [triplets[i]["query"] for i in range(batch_start, batch_end)]

        q_tfidf: csr_matrix = vectorizer.transform(batch_queries)
        q_tfidf = normalize(q_tfidf, norm="l2", axis=1, copy=False)

        top_indices = _chunked_topk(q_tfidf, corpus_tfidf, top_k, chunk_size)

        for j, global_i in enumerate(range(batch_start, batch_end)):
            positive = positives[global_i]
            category = categories[global_i]
            negative: str | None = None

            for corpus_idx in top_indices[j]:
                candidate = documents[corpus_idx]
                # Skip if same as positive or same category
                if candidate == positive:
                    continue
                cand_category = categories[corpus_idx] if corpus_idx < len(categories) else ""
                if cand_category and cand_category == category:
                    continue
                negative = candidate
                break

            if negative:
                updated[global_i]["negative"] = negative
                updated[global_i]["difficulty"] = "hard"
                added += 1
            else:
                failed += 1

    logger.info("Hard negatives: added=%d, failed=%d", added, failed)
    return updated


def split_train_val(
    triplets: list[dict[str, Any]],
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Shuffle and split triplets into train/val sets.

    Args:
        triplets: All triplets to split.
        val_ratio: Fraction for validation.
        rng: Seeded random instance.

    Returns:
        Tuple of (train_triplets, val_triplets).
    """
    shuffled = list(triplets)
    rng.shuffle(shuffled)
    val_n = max(1, int(len(shuffled) * val_ratio))
    return shuffled[val_n:], shuffled[:val_n]


def strip_internal_fields(triplet: dict[str, Any]) -> dict[str, Any]:
    """Remove internal fields not part of the output format."""
    return {k: v for k, v in triplet.items() if not k.startswith("_")}


def copy_v29_shards(src_dir: Path, dst_dir: Path) -> list[str]:
    """Copy all v29.0 shards and val.jsonl to v30.0 directory.

    Args:
        src_dir: Source v29.0 directory.
        dst_dir: Destination v30.0 directory.

    Returns:
        List of copied shard filenames.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    shard_files = sorted(src_dir.glob("train_shard_*.jsonl"))
    logger.info("Copying %d shards from %s to %s...", len(shard_files), src_dir, dst_dir)

    for shard in tqdm(shard_files, desc="Copying shards"):
        shutil.copy2(shard, dst_dir / shard.name)

    val_src = src_dir / "val.jsonl"
    if val_src.exists():
        shutil.copy2(val_src, dst_dir / "val.jsonl")
        logger.info("Copied val.jsonl from v29.0")

    return [f.name for f in shard_files]


def count_shard_lines(shard_path: Path) -> int:
    """Count lines in a JSONL shard file."""
    count = 0
    with open(shard_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def write_new_shards(
    train_triplets: list[dict[str, Any]],
    dst_dir: Path,
    next_shard_idx: int,
    shard_size: int,
) -> list[str]:
    """Write naver-news train triplets as new shards.

    Args:
        train_triplets: Train triplets to write.
        dst_dir: Output directory.
        next_shard_idx: Starting shard index for new shards.
        shard_size: Max samples per shard.

    Returns:
        List of new shard filenames written.
    """
    new_shards: list[str] = []
    shard_idx = next_shard_idx

    for start in range(0, len(train_triplets), shard_size):
        end = min(start + shard_size, len(train_triplets))
        shard_name = f"train_shard_{shard_idx:03d}.jsonl"
        shard_path = dst_dir / shard_name

        with open(shard_path, "w", encoding="utf-8") as f:
            for triplet in train_triplets[start:end]:
                f.write(json.dumps(strip_internal_fields(triplet), ensure_ascii=False) + "\n")

        new_shards.append(shard_name)
        logger.info("Wrote %s: %d samples", shard_name, end - start)
        shard_idx += 1

    return new_shards


def append_val_jsonl(
    val_triplets: list[dict[str, Any]],
    val_path: Path,
) -> None:
    """Append naver-news val triplets to existing val.jsonl.

    Args:
        val_triplets: New validation triplets.
        val_path: Path to val.jsonl to append to.
    """
    with open(val_path, "a", encoding="utf-8") as f:
        for triplet in val_triplets:
            f.write(json.dumps(strip_internal_fields(triplet), ensure_ascii=False) + "\n")
    logger.info("Appended %d val triplets to %s", len(val_triplets), val_path)


def build_metadata(
    dst_dir: Path,
    v29_shard_names: list[str],
    new_shard_names: list[str],
    naver_train_count: int,
    naver_val_count: int,
    v29_meta: dict[str, Any],
) -> dict[str, Any]:
    """Build and write metadata.json for v30.0.

    Args:
        dst_dir: v30.0 directory.
        v29_shard_names: Shard filenames from v29.0.
        new_shard_names: New shard filenames from naver-news.
        naver_train_count: Number of naver-news train triplets.
        naver_val_count: Number of naver-news val triplets.
        v29_meta: Original v29.0 metadata.

    Returns:
        Metadata dict written to disk.
    """
    all_shard_names = v29_shard_names + new_shard_names
    v29_train = v29_meta.get("total_train", 0)
    v29_val = v29_meta.get("total_val", 0)

    total_train = v29_train + naver_train_count
    total_val = v29_val + naver_val_count
    total_unique = total_train + total_val

    source_dist: Counter[str] = Counter()
    # Count new sources from the actual shards
    for shard_name in new_shard_names:
        shard_path = dst_dir / shard_name
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    source_dist[row.get("source", "unknown")] += 1
    # Add val counts
    source_dist["new_data_val"] = naver_val_count
    # Carry over v29 source distribution if available
    for src, cnt in v29_meta.get("source_distribution", {}).items():
        source_dist[src] += cnt

    metadata: dict[str, Any] = {
        "total_train": total_train,
        "total_val": total_val,
        "total_unique": total_unique,
        "shard_size": SHARD_SIZE,
        "num_shards": len(all_shard_names),
        "val_ratio": VAL_RATIO,
        "shards": all_shard_names,
        "source_distribution": dict(source_dist.most_common()),
        "v29_base": {
            "total_train": v29_train,
            "total_val": v29_val,
            "num_shards": len(v29_shard_names),
        },
        "naver_news_added": {
            "train": naver_train_count,
            "val": naver_val_count,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    meta_path = dst_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("Wrote metadata to %s", meta_path)
    return metadata


def main() -> None:
    """Run V30 data preparation pipeline."""
    rng = random.Random(SEED)

    logger.info("=== V30 Data Preparation Pipeline ===")
    logger.info("Output: %s", V30_DIR)

    # Step 1: Download datasets
    logger.info("--- Step 1a: Download naver-news ---")
    naver_rows = download_naver_news()

    logger.info("--- Step 1b: Download BCCard Finance Q&A ---")
    bccard_rows = download_bccard()

    # Step 2: Convert to triplets with random negatives
    logger.info("--- Step 2a: Build naver-news triplets ---")
    naver_triplets = build_triplets(naver_rows, rng)

    logger.info("--- Step 2b: Build BCCard triplets ---")
    bccard_triplets = build_bccard_triplets(bccard_rows, rng)

    # Merge all new triplets
    all_new_triplets = naver_triplets + bccard_triplets
    logger.info("Total new triplets: %d (naver=%d, bccard=%d)",
                len(all_new_triplets), len(naver_triplets), len(bccard_triplets))

    # Step 3: TF-IDF hard negative mining on combined new data
    logger.info("--- Step 3: TF-IDF hard negative mining ---")
    documents = [t["positive"] for t in all_new_triplets]
    vectorizer, corpus_tfidf = build_tfidf_index(documents, TFIDF_MAX_FEATURES)
    all_new_triplets = mine_hard_negatives(
        all_new_triplets,
        vectorizer,
        corpus_tfidf,
        documents,
        top_k=TFIDF_TOP_K,
        batch_size=TFIDF_BATCH_SIZE,
        chunk_size=CORPUS_CHUNK_SIZE,
    )

    # Step 4: Split train/val
    logger.info("--- Step 4: Split train/val (val_ratio=%.2f) ---", VAL_RATIO)
    train_triplets, val_triplets = split_train_val(all_new_triplets, VAL_RATIO, rng)
    logger.info("New data: train=%d, val=%d", len(train_triplets), len(val_triplets))

    # Step 5: Copy v29.0 shards to v30.0
    logger.info("--- Step 5: Copy v29.0 shards to v30.0 ---")
    v29_shard_names = copy_v29_shards(V29_DIR, V30_DIR)

    # Load v29 metadata
    v29_meta_path = V29_DIR / "metadata.json"
    v29_meta: dict[str, Any] = {}
    if v29_meta_path.exists():
        with open(v29_meta_path, encoding="utf-8") as f:
            v29_meta = json.load(f)

    # Step 6: Append new shards
    logger.info("--- Step 6: Write new data shards ---")
    next_shard_idx = len(v29_shard_names)
    new_shard_names = write_new_shards(
        train_triplets, V30_DIR, next_shard_idx, SHARD_SIZE
    )

    # Step 7: Update val.jsonl
    logger.info("--- Step 7: Update val.jsonl ---")
    val_path = V30_DIR / "val.jsonl"
    append_val_jsonl(val_triplets, val_path)

    # Step 8: Write metadata
    logger.info("--- Step 8: Write metadata ---")
    metadata = build_metadata(
        V30_DIR,
        v29_shard_names,
        new_shard_names,
        len(train_triplets),
        len(val_triplets),
        v29_meta,
    )

    logger.info("=== Pipeline complete ===")
    logger.info("  V29 shards copied: %d", len(v29_shard_names))
    logger.info("  New shards added:  %d", len(new_shard_names))
    logger.info("  Total shards:      %d", metadata["num_shards"])
    logger.info("  Total train:       %d", metadata["total_train"])
    logger.info("  Total val:         %d", metadata["total_val"])


if __name__ == "__main__":
    main()
