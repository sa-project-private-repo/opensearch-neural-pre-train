#!/usr/bin/env python3
"""Load Rust-computed IDF weights into PyTorch .pt format."""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch


def load_idf_bin(path: str) -> torch.Tensor:
    """Load IDF weights from Rust binary output."""
    base = Path(path)
    bin_path = base.with_suffix(".bin")
    meta_path = base.with_suffix(".json")

    with open(meta_path) as f:
        meta = json.load(f)

    vocab_size = meta["vocab_size"]
    weights = np.fromfile(str(bin_path), dtype=np.float32, count=vocab_size)
    return torch.from_numpy(weights)


def convert_to_pt(input_path: str, output_path: str | None = None) -> None:
    """Convert .bin IDF to .pt for PyTorch."""
    weights = load_idf_bin(input_path)
    out = Path(output_path or input_path).with_suffix(".pt")
    torch.save(weights, out)
    print(f"Saved {out} (shape={weights.shape})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_idf.py <path_without_ext> [output.pt]")
        sys.exit(1)
    convert_to_pt(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
