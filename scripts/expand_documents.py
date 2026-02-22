#!/usr/bin/env python3
"""Expand training documents using a fine-tuned doc2query (pko-t5) model.

Reads JSONL shards from a data directory (same format as data/v29.0/),
generates N synthetic queries per document's `positive` field, and appends
them using the separator:

    "{original_positive} [SEP] {q1} {q2} {q3} {q4} {q5}"

Expanded shards are written to the output directory preserving original
file names. Multiple shards are processed in parallel using a thread pool.
"""

import argparse
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_PREFIX = "generate question: "


def parse_shard_range(shard_range: str, num_shards: int) -> list[int]:
    """Parse shard range string into list of shard indices.

    Args:
        shard_range: Range string like "0-10", "5", or "all".
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
        data_dir: Directory containing train_shard_*.jsonl files.
        shard_range: Range string for shard selection.

    Returns:
        Sorted list of shard file paths to process.
    """
    all_shards = sorted(data_dir.glob("train_shard_*.jsonl"))
    if not all_shards:
        raise FileNotFoundError(f"No train_shard_*.jsonl files in {data_dir}")

    indices = parse_shard_range(shard_range, len(all_shards))
    selected: list[Path] = []
    for idx in indices:
        if idx < len(all_shards):
            selected.append(all_shards[idx])
        else:
            logger.warning(f"Shard index {idx} out of range (max {len(all_shards) - 1})")
    return selected


def load_model(
    model_dir: Path,
    device: torch.device,
) -> tuple[T5ForConditionalGeneration, AutoTokenizer]:
    """Load fine-tuned T5 model and tokenizer from checkpoint directory.

    Args:
        model_dir: Path to saved model directory (from finetune_doc2query.py).
        device: Torch device to load the model onto.

    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info(f"Loading model from {model_dir}...")
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        str(model_dir)
    )
    model.to(device)
    model.eval()
    logger.info("Model loaded")
    return model, tokenizer


def generate_queries_batch(
    documents: list[str],
    model: T5ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    device: torch.device,
    num_queries: int,
    max_input_length: int,
    max_output_length: int,
) -> list[list[str]]:
    """Generate synthetic queries for a batch of documents using beam search.

    Args:
        documents: List of document strings to generate queries for.
        model: Fine-tuned T5 model.
        tokenizer: Tokenizer aligned with the model.
        device: Torch device for inference.
        num_queries: Number of queries to generate per document.
        max_input_length: Maximum input token length.
        max_output_length: Maximum output token length per query.

    Returns:
        List of query lists; result[i] contains up to num_queries strings
        for documents[i].
    """
    inputs_text = [INPUT_PREFIX + doc for doc in documents]
    encoding = tokenizer(
        inputs_text,
        max_length=max_input_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_queries,
            num_return_sequences=num_queries,
            max_new_tokens=max_output_length,
            early_stopping=True,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Group by document: outputs are [doc0_q0, doc0_q1, ..., doc1_q0, ...]
    batch_size = len(documents)
    result: list[list[str]] = []
    for i in range(batch_size):
        start = i * num_queries
        end = start + num_queries
        queries = [q.strip() for q in decoded[start:end] if q.strip()]
        result.append(queries)
    return result


def expand_positive(original: str, queries: list[str]) -> str:
    """Append generated queries to the original positive text.

    Args:
        original: Original positive document string.
        queries: List of generated query strings.

    Returns:
        Expanded string: "{original} [SEP] {q1} {q2} ... {qN}".
    """
    if not queries:
        return original
    return original + " [SEP] " + " ".join(queries)


def process_shard(
    shard_file: Path,
    output_dir: Path,
    model: T5ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    device: torch.device,
    num_queries: int,
    batch_size: int,
    max_input_length: int,
    max_output_length: int,
) -> dict[str, int]:
    """Expand all documents in a single shard and write to output directory.

    Reads records in batches, generates queries for each positive field,
    and writes expanded records atomically to prevent partial output files.

    Args:
        shard_file: Input JSONL shard path.
        output_dir: Directory for writing expanded shards.
        model: Fine-tuned T5 model.
        tokenizer: Tokenizer aligned with the model.
        device: Torch device for inference.
        num_queries: Number of queries to generate per document.
        batch_size: Number of documents to process per inference call.
        max_input_length: Maximum input token length.
        max_output_length: Maximum generated token length per query.

    Returns:
        Stats dict: total, expanded, skipped (no positive field).
    """
    stats: dict[str, int] = {"total": 0, "expanded": 0, "skipped": 0}

    records: list[dict[str, Any]] = []
    with open(shard_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    stats["total"] = len(records)

    # Separate records by presence of positive field
    indices_with_positive: list[int] = []
    for i, rec in enumerate(records):
        if rec.get("positive"):
            indices_with_positive.append(i)
        else:
            stats["skipped"] += 1

    # Process in batches
    for batch_start in range(0, len(indices_with_positive), batch_size):
        batch_indices = indices_with_positive[batch_start : batch_start + batch_size]
        documents = [records[i]["positive"] for i in batch_indices]

        query_lists = generate_queries_batch(
            documents=documents,
            model=model,
            tokenizer=tokenizer,
            device=device,
            num_queries=num_queries,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
        )

        for record_idx, queries in zip(batch_indices, query_lists):
            records[record_idx]["positive"] = expand_positive(
                records[record_idx]["positive"], queries
            )
            stats["expanded"] += 1

    # Atomic write
    out_path = output_dir / shard_file.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=out_path.parent, prefix=".tmp_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_path, out_path)
    except Exception:
        os.unlink(tmp_path)
        raise

    return stats


def print_stats_summary(all_stats: list[dict[str, int]], shard_files: list[Path]) -> None:
    """Print aggregated expansion statistics across all processed shards.

    Args:
        all_stats: List of per-shard stat dicts.
        shard_files: Processed shard files (for count reporting).
    """
    total = sum(s["total"] for s in all_stats)
    expanded = sum(s["expanded"] for s in all_stats)
    skipped = sum(s["skipped"] for s in all_stats)

    logger.info("=== Document Expansion Summary ===")
    logger.info(f"  Shards processed: {len(shard_files)}")
    logger.info(f"  Total records:    {total:,}")
    logger.info(f"  Expanded:         {expanded:,}")
    logger.info(f"  Skipped (no pos): {skipped:,}")
    if total:
        logger.info(f"  Expansion rate:   {expanded / total * 100:.1f}%")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Expand training documents with doc2query synthetic queries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/doc2query_ko"),
        help="Path to fine-tuned doc2query model directory",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/v29.0"),
        help="Directory containing train_shard_*.jsonl input files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/v29.0_expanded"),
        help="Directory for writing expanded shard files",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=5,
        help="Number of synthetic queries to generate per document",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of documents per inference batch",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=512,
        help="Maximum input token length for the document context",
    )
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=64,
        help="Maximum output token length per generated query",
    )
    parser.add_argument(
        "--shard-range",
        type=str,
        default="all",
        help='Shard range to process: "all", "0-10", or "5"',
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker threads for shard processing. "
            "Use 1 for GPU inference (model is not thread-safe across workers). "
            "Increase only when running CPU inference or with multiple model copies."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for document expansion with doc2query."""
    args = parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model, tokenizer = load_model(args.model_dir, device)

    shard_files = collect_shard_files(args.data_dir, args.shard_range)
    logger.info(
        f"Processing {len(shard_files)} shard(s) from {args.data_dir} "
        f"-> {args.output_dir}"
    )

    all_stats: list[dict[str, int]] = []

    if args.workers == 1:
        # Sequential processing (safe for GPU)
        for shard_file in tqdm(shard_files, desc="Expanding shards"):
            stats = process_shard(
                shard_file=shard_file,
                output_dir=args.output_dir,
                model=model,
                tokenizer=tokenizer,
                device=device,
                num_queries=args.num_queries,
                batch_size=args.batch_size,
                max_input_length=args.max_input_length,
                max_output_length=args.max_output_length,
            )
            all_stats.append(stats)
            logger.debug(
                f"{shard_file.name}: total={stats['total']}, "
                f"expanded={stats['expanded']}"
            )
    else:
        # Parallel processing (CPU only or when multiple model copies exist)
        logger.info(f"Running with {args.workers} parallel workers")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_shard = {
                executor.submit(
                    process_shard,
                    shard_file,
                    args.output_dir,
                    model,
                    tokenizer,
                    device,
                    args.num_queries,
                    args.batch_size,
                    args.max_input_length,
                    args.max_output_length,
                ): shard_file
                for shard_file in shard_files
            }
            with tqdm(total=len(shard_files), desc="Expanding shards") as pbar:
                for future in as_completed(future_to_shard):
                    shard_file = future_to_shard[future]
                    try:
                        stats = future.result()
                        all_stats.append(stats)
                        logger.debug(
                            f"{shard_file.name}: total={stats['total']}, "
                            f"expanded={stats['expanded']}"
                        )
                    except Exception as exc:
                        logger.error(f"{shard_file.name} failed: {exc}")
                    pbar.update(1)

    print_stats_summary(all_stats, shard_files)


if __name__ == "__main__":
    main()
