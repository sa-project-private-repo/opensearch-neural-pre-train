"""
Pre-compute BGE-M3 teacher scores for Knowledge Distillation training.

Encodes all unique texts in the training data with BGE-M3, then computes
cosine similarity scores for each (query, positive) and (query, negative)
pair. Augmented data is saved as new JSONL files with teacher scores.

Usage:
    python scripts/precompute_teacher_scores.py \
        --input-pattern "data/v29.0/train_*.jsonl" \
        --output-dir data/v29.0_kd/ \
        --batch-size 256 \
        --device cuda

Output format (per line):
    {
        "query": "...", "positive": "...", "negative": "...",
        "teacher_pos_score": 0.85, "teacher_neg_score": 0.23,
        ... (original fields preserved)
    }
"""

import argparse
import glob
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def text_hash(text: str) -> str:
    """Compute short hash for text deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def collect_unique_texts(
    file_patterns: List[str],
) -> Tuple[List[str], Dict[str, int]]:
    """
    Collect all unique texts from training JSONL files.

    Returns:
        texts: List of unique texts
        text_to_idx: Mapping from text hash to index in texts list
    """
    seen: Dict[str, int] = {}
    texts: List[str] = []

    all_files: List[Path] = []
    for pattern in file_patterns:
        all_files.extend(sorted(Path(f) for f in glob.glob(pattern)))

    logger.info(f"Scanning {len(all_files)} files for unique texts...")

    for file_path in tqdm(all_files, desc="Scanning"):
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
                    if h not in seen:
                        seen[h] = len(texts)
                        texts.append(text)

    logger.info(f"Found {len(texts):,} unique texts")
    return texts, seen


def encode_texts(
    texts: List[str],
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 256,
    device: str = "cuda",
    max_length: int = 256,
) -> np.ndarray:
    """
    Encode all texts with BGE-M3.

    Returns:
        embeddings: numpy array [num_texts, 1024]
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading teacher model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_length

    logger.info(
        f"Encoding {len(texts):,} texts "
        f"(batch_size={batch_size}, device={device})..."
    )
    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    elapsed = time.time() - start
    logger.info(
        f"Encoding done in {elapsed:.1f}s "
        f"({len(texts) / elapsed:.0f} texts/s)"
    )

    return embeddings


def compute_and_save_scores(
    file_patterns: List[str],
    output_dir: Path,
    embeddings: np.ndarray,
    text_to_idx: Dict[str, int],
) -> int:
    """
    Compute teacher scores for each triplet and save augmented JSONL.

    Returns:
        Total number of triplets processed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    all_files: List[Path] = []
    for pattern in file_patterns:
        all_files.extend(sorted(Path(f) for f in glob.glob(pattern)))

    for file_path in tqdm(all_files, desc="Computing scores"):
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
                negative = item.get("negative")

                q_idx = text_to_idx.get(text_hash(query))
                p_idx = text_to_idx.get(text_hash(positive))

                if q_idx is None or p_idx is None:
                    # Skip if text not found (shouldn't happen)
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    continue

                # Cosine similarity (embeddings are L2-normalized)
                q_emb = embeddings[q_idx]
                p_emb = embeddings[p_idx]
                pos_score = float(np.dot(q_emb, p_emb))

                neg_score = 0.0
                if negative:
                    n_idx = text_to_idx.get(text_hash(negative))
                    if n_idx is not None:
                        n_emb = embeddings[n_idx]
                        neg_score = float(np.dot(q_emb, n_emb))

                item["teacher_pos_score"] = round(pos_score, 6)
                item["teacher_neg_score"] = round(neg_score, 6)

                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                total += 1

    return total


def copy_val_files(
    file_patterns: List[str],
    output_dir: Path,
) -> None:
    """Copy validation files to output directory (no teacher scores)."""
    for pattern in file_patterns:
        for file_path in glob.glob(pattern):
            src = Path(file_path)
            dst = output_dir / src.name
            if not dst.exists():
                import shutil
                shutil.copy2(src, dst)
                logger.info(f"Copied val file: {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute BGE-M3 teacher scores for KD training",
    )
    parser.add_argument(
        "--input-pattern",
        type=str,
        default="data/v29.0/train_*.jsonl",
        help="Glob pattern for training JSONL files",
    )
    parser.add_argument(
        "--val-pattern",
        type=str,
        default="data/v29.0/val.jsonl",
        help="Glob pattern for validation JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/v29.0_kd",
        help="Output directory for augmented JSONL files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-m3",
        help="Teacher model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for teacher",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for encoding",
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save embeddings to disk for reuse",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Step 1: Collect unique texts
    texts, text_to_idx = collect_unique_texts([args.input_pattern])

    # Step 2: Encode with BGE-M3
    emb_cache = output_dir / "teacher_embeddings.npy"
    idx_cache = output_dir / "text_to_idx.json"

    if emb_cache.exists() and idx_cache.exists():
        logger.info(f"Loading cached embeddings from {emb_cache}")
        embeddings = np.load(str(emb_cache))
        with open(idx_cache, "r") as f:
            text_to_idx = json.load(f)
    else:
        embeddings = encode_texts(
            texts,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
            max_length=args.max_length,
        )

        if args.save_embeddings:
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(emb_cache), embeddings)
            with open(idx_cache, "w") as f:
                json.dump(text_to_idx, f)
            logger.info(f"Saved embeddings to {emb_cache}")

    # Step 3: Compute scores and save
    total = compute_and_save_scores(
        [args.input_pattern],
        output_dir,
        embeddings,
        text_to_idx,
    )

    # Step 4: Copy validation files
    copy_val_files([args.val_pattern], output_dir)

    logger.info(f"Done! Processed {total:,} triplets -> {output_dir}")
    logger.info(
        f"Memory: embeddings={embeddings.nbytes / 1e9:.2f}GB "
        f"({embeddings.shape})"
    )


if __name__ == "__main__":
    main()
