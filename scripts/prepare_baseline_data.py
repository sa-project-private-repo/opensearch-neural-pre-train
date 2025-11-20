#!/usr/bin/env python3
"""
Prepare baseline training data (10K samples) for quick testing.

Samples:
- 5,000 from Korean Wikipedia (ko_wiki)
- 5,000 from NamuWiki (namuwiki)

Output: dataset/baseline_samples/
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Configuration
SAMPLE_SIZE = 10000
KO_WIKI_SAMPLES = 5000
NAMUWIKI_SAMPLES = 5000
RANDOM_SEED = 42

# Paths
DATA_DIR = Path("dataset/paired_data_split")
OUTPUT_DIR = Path("dataset/baseline_samples")

random.seed(RANDOM_SEED)


def load_jsonl_files(patterns: List[str]) -> List[Dict]:
    """Load data from JSONL files matching patterns."""
    data = []
    for pattern in patterns:
        files = sorted(DATA_DIR.glob(pattern))
        print(f"  Found {len(files)} files matching '{pattern}'")
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], output_path: Path):
    """Save data to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Saved {len(data)} samples to {output_path}")


def main():
    print("=" * 70)
    print("Preparing Baseline Training Data (10K samples)")
    print("=" * 70)

    # 1. Load Korean Wikipedia data
    print("\n[1/4] Loading Korean Wikipedia data")
    ko_wiki_patterns = [
        "ko_wiki_title_summary_*_train_*.jsonl",
        "ko_wiki_title_paragraph_*_train_*.jsonl",
    ]
    ko_wiki_data = load_jsonl_files(ko_wiki_patterns)
    print(f"  Total Korean Wikipedia: {len(ko_wiki_data):,} samples")

    # 2. Load NamuWiki data
    print("\n[2/4] Loading NamuWiki data")
    namuwiki_patterns = [
        "namuwiki_title_summary_*_train_*.jsonl",
        "namuwiki_title_paragraph_*_train_*.jsonl",
    ]
    namuwiki_data = load_jsonl_files(namuwiki_patterns)
    print(f"  Total NamuWiki: {len(namuwiki_data):,} samples")

    # 3. Sample data
    print("\n[3/4] Sampling data")

    # Sample from Korean Wikipedia
    if len(ko_wiki_data) >= KO_WIKI_SAMPLES:
        ko_wiki_sampled = random.sample(ko_wiki_data, KO_WIKI_SAMPLES)
    else:
        print(f"  Warning: Only {len(ko_wiki_data)} Korean Wikipedia samples available")
        ko_wiki_sampled = ko_wiki_data
    print(f"  Sampled {len(ko_wiki_sampled):,} from Korean Wikipedia")

    # Sample from NamuWiki
    if len(namuwiki_data) >= NAMUWIKI_SAMPLES:
        namuwiki_sampled = random.sample(namuwiki_data, NAMUWIKI_SAMPLES)
    else:
        print(f"  Warning: Only {len(namuwiki_data)} NamuWiki samples available")
        namuwiki_sampled = namuwiki_data
    print(f"  Sampled {len(namuwiki_sampled):,} from NamuWiki")

    # Combine and shuffle
    all_samples = ko_wiki_sampled + namuwiki_sampled
    random.shuffle(all_samples)
    print(f"  Total samples: {len(all_samples):,}")

    # 4. Split into train/val (90/10)
    print("\n[4/4] Splitting into train/val")
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"  Train: {len(train_samples):,} samples")
    print(f"  Val: {len(val_samples):,} samples")

    # Save
    save_jsonl(train_samples, OUTPUT_DIR / "train_baseline.jsonl")
    save_jsonl(val_samples, OUTPUT_DIR / "val_baseline.jsonl")

    print("\n" + "=" * 70)
    print("✓ Baseline data preparation complete!")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print(f"  - train_baseline.jsonl: {len(train_samples):,} samples")
    print(f"  - val_baseline.jsonl: {len(val_samples):,} samples")
    print("\nNext step:")
    print("  source .venv/bin/activate")
    print("  python train.py --config configs/baseline_dgx.yaml")
    print("=" * 70)


if __name__ == "__main__":
    main()
