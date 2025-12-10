#!/usr/bin/env python3
"""
Create v16 dataset by combining large-scale external data with v15 data.

Data sources:
- ko_en_terms_cleaned_v2.jsonl: 1.5M+ cleaned Korean-English term pairs
- v15_aggressive/term_pairs.jsonl: 192K curated term pairs

Key improvements:
1. Much larger dataset (10x more data)
2. Better coverage from external sources
3. Balanced sampling to prevent imbalance
"""

import json
import random
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = DATASET_DIR / "v16_large"


def load_jsonl(path: Path, key_mapping: dict | None = None) -> list[dict]:
    """Load JSONL file with optional key remapping."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            if key_mapping:
                item = {key_mapping.get(k, k): v for k, v in item.items()}
            data.append(item)
    return data


def filter_valid_pairs(data: list[dict]) -> list[dict]:
    """Filter out invalid or low-quality pairs."""
    valid = []
    for item in data:
        ko = item.get("ko_text") or item.get("ko_term", "")
        en = item.get("en_text") or item.get("en_term", "")

        # Skip empty
        if not ko or not en:
            continue

        # Skip if Korean term is too short
        if len(ko) < 2:
            continue

        # Skip if English term is too short
        if len(en) < 2:
            continue

        # Skip if Korean term contains only English/numbers
        has_korean = any("\uac00" <= c <= "\ud7a3" for c in ko)
        if not has_korean and not any(c.isupper() for c in ko):  # Allow abbreviations
            continue

        # Normalize keys
        valid.append({
            "ko_text": ko,
            "en_text": en,
            "source": item.get("source", "unknown"),
        })

    return valid


def deduplicate_pairs(data: list[dict]) -> list[dict]:
    """Remove duplicate pairs while preserving order."""
    seen = set()
    unique = []
    for item in data:
        key = (item["ko_text"], item["en_text"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Creating v16 Dataset from External Data Sources")
    print("=" * 70)

    all_data = []

    # 1. Load large-scale external data
    print("\n1. Loading external data sources...")

    # ko_en_terms_cleaned_v2.jsonl (1.5M pairs)
    large_scale_path = DATASET_DIR / "large_scale" / "ko_en_terms_cleaned_v2.jsonl"
    if large_scale_path.exists():
        print(f"   Loading {large_scale_path.name}...")
        large_data = load_jsonl(
            large_scale_path,
            key_mapping={"ko_term": "ko_text", "en_term": "en_text"}
        )
        print(f"   Loaded {len(large_data):,} pairs")
        all_data.extend(large_data)
    else:
        print(f"   WARNING: {large_scale_path} not found!")

    # 2. Load v15 curated data
    print("\n2. Loading v15 curated data...")
    v15_path = DATASET_DIR / "v15_aggressive" / "term_pairs.jsonl"
    if v15_path.exists():
        v15_data = load_jsonl(v15_path)
        print(f"   Loaded {len(v15_data):,} pairs from v15")
        # Oversample v15 data 3x to ensure curated terms are well represented
        all_data.extend(v15_data * 3)
    else:
        print(f"   WARNING: {v15_path} not found!")

    # 3. Filter valid pairs
    print("\n3. Filtering valid pairs...")
    valid_data = filter_valid_pairs(all_data)
    print(f"   Valid pairs: {len(valid_data):,} (from {len(all_data):,})")

    # 4. Deduplicate
    print("\n4. Deduplicating...")
    unique_data = deduplicate_pairs(valid_data)
    print(f"   Unique pairs: {len(unique_data):,}")

    # 5. Shuffle
    print("\n5. Shuffling data...")
    random.seed(42)
    random.shuffle(unique_data)

    # 6. Statistics
    print("\n6. Dataset statistics:")
    sources = Counter(item.get("source", "unknown") for item in unique_data)
    for source, count in sources.most_common():
        pct = count / len(unique_data) * 100
        print(f"   {source:20s}: {count:10,} ({pct:5.1f}%)")

    # Check Korean term length distribution
    ko_lengths = Counter(len(item["ko_text"]) for item in unique_data)
    print("\n   Korean term length distribution:")
    for length in sorted(ko_lengths.keys())[:10]:
        count = ko_lengths[length]
        print(f"      Length {length}: {count:,}")

    # 7. Save
    output_path = OUTPUT_DIR / "term_pairs.jsonl"
    print(f"\n7. Saving to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in unique_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Created v16 dataset: {len(unique_data):,} pairs")

    # 8. Sample verification
    print("\n8. Sample pairs:")
    for i in range(10):
        item = unique_data[i * 1000]
        print(f"   {item['ko_text']:20s} → {item['en_text']}")


if __name__ == "__main__":
    main()
