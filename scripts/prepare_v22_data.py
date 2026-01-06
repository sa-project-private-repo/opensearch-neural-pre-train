#!/usr/bin/env python3
"""
Prepare v22.0 training data by combining:
1. Expanded single-term data (29,322 triplets)
2. Existing v21.4 training data

Creates phase-specific files for curriculum learning with InfoNCE.
"""
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

random.seed(42)

# Paths
DATA_DIR_V21 = Path("data/v21.4")
DATA_DIR_V22 = Path("data/v22.0")
DATA_DIR_V22.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: Path):
    """Save to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data):,} triplets to {path}")


def main():
    # Load v22.0 expanded single-term data
    single_term_path = DATA_DIR_V22 / "single_term_expanded.jsonl"
    single_term_data = load_jsonl(single_term_path)
    print(f"Loaded {len(single_term_data):,} expanded single-term triplets")

    # Load v21.4 phase data
    phase1_v21 = load_jsonl(DATA_DIR_V21 / "phase1_single_term_focus_triplets.jsonl")
    phase2_v21 = load_jsonl(DATA_DIR_V21 / "phase2_balanced_triplets.jsonl")
    phase3_v21 = load_jsonl(DATA_DIR_V21 / "phase3_full_triplets.jsonl")

    print(f"\nv21.4 data:")
    print(f"  Phase 1: {len(phase1_v21):,}")
    print(f"  Phase 2: {len(phase2_v21):,}")
    print(f"  Phase 3: {len(phase3_v21):,}")

    # Separate expanded data by difficulty
    easy = [d for d in single_term_data if d.get("difficulty") == "easy"]
    medium = [d for d in single_term_data if d.get("difficulty") == "medium"]
    hard = [d for d in single_term_data if d.get("difficulty") == "hard"]

    print(f"\nExpanded single-term by difficulty:")
    print(f"  Easy: {len(easy):,}")
    print(f"  Medium: {len(medium):,}")
    print(f"  Hard: {len(hard):,}")

    # Phase 1: Single-term focus (50% new single-term + v21.4 phase1)
    # Goal: Maximize single-term learning with InfoNCE
    phase1_v22 = []

    # Add all expanded single-term data
    phase1_v22.extend(single_term_data)

    # Add v21.4 single-term data (filter to avoid duplicates)
    existing_pairs = set()
    for item in single_term_data:
        key = (item["anchor"], item["positive"])
        existing_pairs.add(key)
        existing_pairs.add((item["positive"], item["anchor"]))

    for item in phase1_v21:
        key = (item["anchor"], item["positive"])
        if key not in existing_pairs:
            phase1_v22.append(item)
            existing_pairs.add(key)

    random.shuffle(phase1_v22)

    # Phase 2: Balanced (33% single-term + diverse sentence pairs)
    phase2_v22 = []

    # Sample from expanded single-term
    n_single = min(len(single_term_data), 20000)
    phase2_v22.extend(random.sample(single_term_data, n_single))

    # Add v21.4 phase2 data
    phase2_v22.extend(phase2_v21)
    random.shuffle(phase2_v22)

    # Phase 3: Full data + hard negatives
    phase3_v22 = []

    # All expanded single-term data
    phase3_v22.extend(single_term_data)

    # All v21.4 phase3 data
    phase3_v22.extend(phase3_v21)
    random.shuffle(phase3_v22)

    # Create validation set from expanded data
    val_size = min(5000, len(single_term_data) // 5)
    val_indices = random.sample(range(len(single_term_data)), val_size)
    validation_data = [single_term_data[i] for i in val_indices]

    # Also include v21.4 validation
    val_v21_path = DATA_DIR_V21 / "validation_triplets.jsonl"
    if val_v21_path.exists():
        val_v21 = load_jsonl(val_v21_path)
        validation_data.extend(val_v21)
    random.shuffle(validation_data)

    print(f"\nv22.0 data distribution:")
    print(f"  Phase 1 (single-term focus): {len(phase1_v22):,}")
    print(f"  Phase 2 (balanced): {len(phase2_v22):,}")
    print(f"  Phase 3 (full): {len(phase3_v22):,}")
    print(f"  Validation: {len(validation_data):,}")

    # Save files
    save_jsonl(phase1_v22, DATA_DIR_V22 / "phase1_single_term_focus_triplets.jsonl")
    save_jsonl(phase2_v22, DATA_DIR_V22 / "phase2_balanced_triplets.jsonl")
    save_jsonl(phase3_v22, DATA_DIR_V22 / "phase3_full_triplets.jsonl")
    save_jsonl(validation_data, DATA_DIR_V22 / "validation_triplets.jsonl")

    # Print pair type distribution for phase 1
    print("\nPhase 1 pair_type distribution:")
    pair_types = defaultdict(int)
    for item in phase1_v22:
        pair_types[item.get("pair_type", "unknown")] += 1
    for pt, count in sorted(pair_types.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(phase1_v22)
        print(f"  {pt}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
