"""Hard negative mining script for AI Hub data using BGE-M3.

Processes AI Hub JSONL files and mines hard negatives for entries that don't
have them yet. Uses BGEM3HardNegativeMiner with FAISS for efficient similarity
search over large document collections.

Usage:
    python scripts/mine_aihub_negatives.py
    python scripts/mine_aihub_negatives.py --chunk_size 100000
    python scripts/mine_aihub_negatives.py --input_dir data/custom
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from src.preprocessing.miners.bge_m3_miner import BGEM3HardNegativeMiner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of JSON objects
    """
    entries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def save_jsonl(entries: List[Dict], file_path: Path) -> None:
    """Save entries to JSONL file.

    Args:
        entries: List of JSON objects
        file_path: Output JSONL file path
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def mine_negatives_for_file(
    input_path: Path,
    output_path: Path,
    miner: BGEM3HardNegativeMiner,
    chunk_size: int = 50000,
) -> None:
    """Mine hard negatives for a single AI Hub file.

    Args:
        input_path: Input JSONL file path
        output_path: Output JSONL file path
        miner: BGEM3HardNegativeMiner instance
        chunk_size: Number of entries per chunk for FAISS index
    """
    logger.info(f"Processing: {input_path}")

    # Load all entries
    entries = load_jsonl(input_path)
    logger.info(f"Loaded {len(entries)} entries")

    # Separate entries needing mining vs already complete
    needs_mining = [e for e in entries if e.get("negative") is None]
    complete = [e for e in entries if e.get("negative") is not None]

    logger.info(
        f"Status: {len(complete)} complete, {len(needs_mining)} need mining"
    )

    if not needs_mining:
        logger.info("All entries already have negatives, skipping")
        return

    # Process in chunks to manage memory
    mined_entries = []
    total_chunks = (len(needs_mining) + chunk_size - 1) // chunk_size

    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(needs_mining))
        chunk = needs_mining[start_idx:end_idx]

        logger.info(
            f"Processing chunk {chunk_idx + 1}/{total_chunks} "
            f"({len(chunk)} entries)"
        )

        # Collect unique positives from this chunk
        positives = list(set(e["positive"] for e in chunk))
        logger.info(f"Building FAISS index with {len(positives)} unique documents")

        # Build FAISS index for this chunk
        miner.build_index(positives, index_type="flat")

        # Mine negatives
        queries = [e["query"] for e in chunk]
        chunk_positives = [e["positive"] for e in chunk]

        results = miner.mine_negatives(
            queries=queries,
            positives=chunk_positives,
            num_negatives=1,
            min_score=0.3,
            max_score=0.85,
        )

        # Update entries with mined negatives
        for entry, result in zip(chunk, results):
            if result.negatives:
                entry["negative"] = result.negatives[0]
                entry["difficulty"] = "hard"
                if "metadata" not in entry:
                    entry["metadata"] = {}
                entry["metadata"]["negative_score"] = result.negative_scores[0]
                mined_entries.append(entry)
            else:
                # No suitable negative found, keep as is
                mined_entries.append(entry)

        # Clear index to free memory
        miner.clear_index()

        success_count = sum(1 for e in mined_entries if e.get("negative"))
        logger.info(
            f"Chunk complete: {success_count}/{len(mined_entries)} "
            f"entries now have negatives"
        )

    # Combine complete and mined entries
    all_entries = complete + mined_entries

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(all_entries, output_path)

    final_success = sum(1 for e in all_entries if e.get("negative"))
    logger.info(
        f"Saved to {output_path}: {final_success}/{len(all_entries)} "
        f"entries have negatives"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mine hard negatives for AI Hub data using BGE-M3"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/aihub/processed",
        help="Input directory containing aihub_*.jsonl files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/aihub/processed",
        help="Output directory for *_mined.jsonl files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for BGE-M3 encoding",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50000,
        help="Number of entries per chunk for FAISS index building",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    # Find all aihub_*.jsonl files (excluding *_mined.jsonl)
    input_files = [
        f
        for f in input_dir.glob("aihub_*.jsonl")
        if not f.name.endswith("_mined.jsonl")
    ]

    if not input_files:
        logger.error(f"No aihub_*.jsonl files found in {input_dir}")
        return

    logger.info(f"Found {len(input_files)} files to process")

    # Initialize miner
    logger.info(
        f"Initializing BGE-M3 miner (batch_size={args.batch_size}, "
        f"chunk_size={args.chunk_size})"
    )
    miner = BGEM3HardNegativeMiner(
        model_name="BAAI/bge-m3",
        batch_size=args.batch_size,
        max_length=192,
        use_fp16=True,
    )

    # Process each file
    for input_file in sorted(input_files):
        # Extract dataset ID from filename
        # aihub_624.jsonl -> aihub_624_mined.jsonl
        dataset_id = input_file.stem.replace("aihub_", "")
        output_file = output_dir / f"aihub_{dataset_id}_mined.jsonl"

        try:
            mine_negatives_for_file(
                input_path=input_file,
                output_path=output_file,
                miner=miner,
                chunk_size=args.chunk_size,
            )
        except Exception as e:
            logger.error(f"Failed to process {input_file}: {e}", exc_info=True)
            continue

    logger.info("All files processed successfully")


if __name__ == "__main__":
    main()
