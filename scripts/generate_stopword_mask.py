#!/usr/bin/env python3
"""Generate Korean stopword mask for V25 training."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer

from src.train.idf import create_stopword_mask, get_korean_stopword_ids


def main():
    output_dir = Path("outputs/idf_weights")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "xlmr_stopword_mask.pt"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    stopword_ids = get_korean_stopword_ids(tokenizer)
    print(f"Found {len(stopword_ids)} Korean stopword tokens")

    mask = create_stopword_mask(tokenizer)
    torch.save(mask, str(output_path))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
