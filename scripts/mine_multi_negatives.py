"""
Mine multiple hard negatives per query using BGE-M3 embeddings + FAISS.

Reuses cached teacher embeddings from precompute_teacher_scores.py to
find k hard negatives for each query from retrieval ranks 10-50.

Usage:
    python scripts/mine_multi_negatives.py \
        --input-pattern "data/v29.0_kd/train_*.jsonl" \
        --embeddings data/v29.0_kd/teacher_embeddings.npy \
        --text-index data/v29.0_kd/text_to_idx.json \
        --output-dir data/v29.0_multi_neg/ \
        --k 7

Output format (per line):
    {
        "query": "...", "positive": "...",
        "negatives": ["neg1", ..., "neg7"],
        "teacher_pos_score": 0.85,
        "teacher_neg_scores": [0.42, 0.38, ...],
        ... (original fields preserved)
    }
"""

import argparse
import glob
import hashlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def text_hash(text: str) -> str:
    """Compute short hash for text deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def load_embeddings_and_index(
    embeddings_path: str,
    text_index_path: str,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Load cached teacher embeddings and text-to-index mapping."""
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    logger.info(f"Loading text index from {text_index_path}")
    with open(text_index_path, "r") as f:
        text_to_idx = json.load(f)
    logger.info(f"Text index entries: {len(text_to_idx):,}")

    return embeddings, text_to_idx


def build_text_mapping(
    file_patterns: List[str],
) -> Dict[str, str]:
    """
    Build hash -> text mapping by re-scanning JSONL files.

    Needed to recover actual text from embedding index.
    """
    hash_to_text: Dict[str, str] = {}

    all_files: List[Path] = []
    for pattern in file_patterns:
        all_files.extend(sorted(Path(f) for f in glob.glob(pattern)))

    logger.info(f"Building text mapping from {len(all_files)} files...")

    for file_path in tqdm(all_files, desc="Mapping texts"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                for field in ("query", "positive", "negative"):
                    text = item.get(field)
                    if text is None:
                        continue
                    h = text_hash(text)
                    if h not in hash_to_text:
                        hash_to_text[h] = text

    logger.info(f"Mapped {len(hash_to_text):,} unique texts")
    return hash_to_text


def collect_doc_indices(
    file_patterns: List[str],
    text_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Collect all unique document indices (positives + negatives).

    Returns:
        doc_indices: sorted array of embedding indices for documents
        doc_idx_to_faiss: mapping from embedding index to FAISS position
    """
    doc_hashes: Set[str] = set()

    all_files: List[Path] = []
    for pattern in file_patterns:
        all_files.extend(sorted(Path(f) for f in glob.glob(pattern)))

    for file_path in tqdm(all_files, desc="Collecting docs"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                for field in ("positive", "negative"):
                    text = item.get(field)
                    if text:
                        doc_hashes.add(text_hash(text))

    doc_indices = sorted(
        text_to_idx[h] for h in doc_hashes if h in text_to_idx
    )
    doc_idx_to_faiss = {idx: pos for pos, idx in enumerate(doc_indices)}

    logger.info(f"Collected {len(doc_indices):,} unique document embeddings")
    return np.array(doc_indices, dtype=np.int64), doc_idx_to_faiss


def batch_search_unique_queries(
    embeddings: np.ndarray,
    text_to_idx: Dict[str, int],
    file_patterns: List[str],
    doc_indices: np.ndarray,
    search_k: int = 100,
    batch_size: int = 4096,
    gpu_id: int = 0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Batch-search all unique queries using PyTorch GPU matmul.

    Uses torch.mm on GPU for inner-product search (B200 compatible).

    Returns:
        query_cache: {query_hash: (scores[search_k], ids[search_k])}
    """
    import torch

    # Collect unique query hashes
    unique_q_hashes: List[str] = []
    seen: Set[str] = set()

    all_files: List[Path] = []
    for pattern in file_patterns:
        all_files.extend(sorted(Path(f) for f in glob.glob(pattern)))

    for file_path in all_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                q_hash = text_hash(item.get("query", ""))
                if q_hash not in seen and q_hash in text_to_idx:
                    seen.add(q_hash)
                    unique_q_hashes.append(q_hash)

    logger.info(
        f"Batch searching {len(unique_q_hashes):,} unique queries "
        f"against {len(doc_indices):,} docs on GPU {gpu_id}..."
    )

    # Load document embeddings to GPU
    device = torch.device(f"cuda:{gpu_id}")
    doc_embs = torch.from_numpy(
        embeddings[doc_indices].astype(np.float32)
    ).to(device)
    logger.info(
        f"Doc embeddings on GPU: {doc_embs.shape}, "
        f"{doc_embs.element_size() * doc_embs.nelement() / 1e9:.1f}GB"
    )

    # Batch search with PyTorch matmul
    query_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for start in tqdm(
        range(0, len(unique_q_hashes), batch_size),
        desc="GPU batch search",
    ):
        batch_hashes = unique_q_hashes[start : start + batch_size]
        batch_indices = [text_to_idx[h] for h in batch_hashes]
        q_batch = torch.from_numpy(
            embeddings[batch_indices].astype(np.float32)
        ).to(device)

        # Inner product: [batch, dim] x [dim, n_docs] -> [batch, n_docs]
        scores = torch.mm(q_batch, doc_embs.T)
        topk_scores, topk_ids = torch.topk(scores, search_k, dim=1)

        topk_scores_np = topk_scores.cpu().numpy()
        topk_ids_np = topk_ids.cpu().numpy()

        for i, q_hash in enumerate(batch_hashes):
            query_cache[q_hash] = (topk_scores_np[i], topk_ids_np[i])

    del doc_embs
    torch.cuda.empty_cache()

    logger.info(f"Cached search results for {len(query_cache):,} queries")
    return query_cache


def mine_negatives(
    file_patterns: List[str],
    output_dir: Path,
    embeddings: np.ndarray,
    text_to_idx: Dict[str, int],
    hash_to_text: Dict[str, str],
    doc_indices: np.ndarray,
    doc_idx_to_faiss: Dict[int, int],
    k: int = 7,
    rank_start: int = 10,
    rank_end: int = 50,
    search_k: int = 100,
) -> int:
    """
    Mine hard negatives for each training example.

    Phase 1: Batch GPU search for all unique queries (PyTorch matmul).
    Phase 2: Assign k negatives per triplet from cached results.

    Returns:
        Total number of examples processed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build reverse mapping: embedding index -> hash
    idx_to_hash: Dict[int, str] = {}
    for h, idx in text_to_idx.items():
        idx_to_hash[idx] = h

    # Phase 1: Batch GPU search for all unique queries
    query_cache = batch_search_unique_queries(
        embeddings, text_to_idx, file_patterns,
        doc_indices, search_k,
    )

    # Phase 2: Assign negatives per triplet
    total = 0
    skipped = 0

    all_files: List[Path] = []
    for pattern in file_patterns:
        all_files.extend(sorted(Path(f) for f in glob.glob(pattern)))

    for file_path in tqdm(all_files, desc="Assigning negatives"):
        out_path = output_dir / file_path.name

        with (
            open(file_path, "r", encoding="utf-8") as fin,
            open(out_path, "w", encoding="utf-8") as fout,
        ):
            for line in fin:
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                query = item.get("query", "")
                positive = item.get("positive", "")
                q_hash = text_hash(query)
                p_hash = text_hash(positive)

                cached = query_cache.get(q_hash)
                if cached is None:
                    fout.write(
                        json.dumps(item, ensure_ascii=False) + "\n"
                    )
                    skipped += 1
                    continue

                scores, faiss_ids = cached

                p_faiss_id = doc_idx_to_faiss.get(
                    text_to_idx.get(p_hash, -1), -1
                )

                neg_texts: List[str] = []
                neg_scores: List[float] = []

                for rank_pos in range(
                    min(rank_start, len(faiss_ids)),
                    min(rank_end, len(faiss_ids)),
                ):
                    fid = int(faiss_ids[rank_pos])
                    if fid < 0 or fid == p_faiss_id:
                        continue

                    emb_idx = int(doc_indices[fid])
                    neg_hash = idx_to_hash.get(emb_idx)
                    if neg_hash is None:
                        continue

                    neg_text = hash_to_text.get(neg_hash)
                    if neg_text is None:
                        continue

                    neg_texts.append(neg_text)
                    neg_scores.append(
                        round(float(scores[rank_pos]), 6)
                    )

                    if len(neg_texts) >= k:
                        break

                # Pad if not enough
                if len(neg_texts) < k:
                    orig_neg = item.get("negative", positive)
                    orig_score = item.get("teacher_neg_score", 0.0)
                    while len(neg_texts) < k:
                        neg_texts.append(orig_neg)
                        neg_scores.append(orig_score)

                out_item = {
                    "query": query,
                    "positive": positive,
                    "negatives": neg_texts,
                    "teacher_pos_score": item.get(
                        "teacher_pos_score", 0.0
                    ),
                    "teacher_neg_scores": neg_scores,
                }

                for meta_key in ("pair_type", "difficulty", "source"):
                    if meta_key in item:
                        out_item[meta_key] = item[meta_key]

                fout.write(
                    json.dumps(out_item, ensure_ascii=False) + "\n"
                )
                total += 1

    if skipped > 0:
        logger.warning(
            f"Skipped {skipped:,} examples (missing embeddings)"
        )

    return total


def copy_val_files(
    file_patterns: List[str],
    output_dir: Path,
) -> None:
    """Copy validation files to output directory."""
    import shutil

    for pattern in file_patterns:
        for file_path in glob.glob(pattern):
            src = Path(file_path)
            dst = output_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                logger.info(f"Copied val file: {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine multi-hard-negatives using cached BGE-M3 embeddings",
    )
    parser.add_argument(
        "--input-pattern",
        type=str,
        default="data/v29.0_kd/train_*.jsonl",
        help="Glob pattern for input JSONL files (with teacher scores)",
    )
    parser.add_argument(
        "--val-pattern",
        type=str,
        default="data/v29.0_kd/val.jsonl",
        help="Glob pattern for validation JSONL files",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="data/v29.0_kd/teacher_embeddings.npy",
        help="Path to cached teacher embeddings (.npy)",
    )
    parser.add_argument(
        "--text-index",
        type=str,
        default="data/v29.0_kd/text_to_idx.json",
        help="Path to text-to-index mapping (.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/v29.0_multi_neg",
        help="Output directory for multi-negative JSONL",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=7,
        help="Number of hard negatives per query",
    )
    parser.add_argument(
        "--rank-start",
        type=int,
        default=10,
        help="Start rank for negative selection (skip top-N)",
    )
    parser.add_argument(
        "--rank-end",
        type=int,
        default=50,
        help="End rank for negative selection",
    )
    parser.add_argument(
        "--search-k",
        type=int,
        default=100,
        help="Number of candidates to retrieve from FAISS",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Load cached embeddings
    embeddings, text_to_idx = load_embeddings_and_index(
        args.embeddings, args.text_index,
    )

    # Build text mapping (hash -> actual text)
    hash_to_text = build_text_mapping([args.input_pattern])

    # Collect document indices
    doc_indices, doc_idx_to_faiss = collect_doc_indices(
        [args.input_pattern], text_to_idx,
    )

    # Mine negatives (uses PyTorch GPU matmul for search)
    start = time.time()
    total = mine_negatives(
        file_patterns=[args.input_pattern],
        output_dir=output_dir,
        embeddings=embeddings,
        text_to_idx=text_to_idx,
        hash_to_text=hash_to_text,
        doc_indices=doc_indices,
        doc_idx_to_faiss=doc_idx_to_faiss,
        k=args.k,
        rank_start=args.rank_start,
        rank_end=args.rank_end,
        search_k=args.search_k,
    )
    elapsed = time.time() - start

    # Copy validation files
    copy_val_files([args.val_pattern], output_dir)

    logger.info(
        f"Done! Mined {args.k} negatives for {total:,} triplets "
        f"in {elapsed:.1f}s -> {output_dir}"
    )


if __name__ == "__main__":
    main()
