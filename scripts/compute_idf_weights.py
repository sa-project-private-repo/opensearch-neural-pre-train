#!/usr/bin/env python3
"""Compute IDF weights for V25 training."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

from src.train.idf.idf_computer import load_or_compute_idf


def main():
    output_dir = Path("outputs/idf_weights")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "xlmr_v25_idf"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    corpus_files = ["data/v24.0/train_*.jsonl"]

    print("Computing IDF weights from training corpus...")
    idf_weights = load_or_compute_idf(
        cache_path=output_path,
        corpus_files=corpus_files,
        tokenizer=tokenizer,
        recompute=True,
        smoothing="bm25",
    )
    print(f"Done! IDF weights shape: {idf_weights.shape}")
    print(f"Saved to {output_path}.pt")


if __name__ == "__main__":
    main()
