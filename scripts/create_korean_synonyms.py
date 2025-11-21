#!/usr/bin/env python3
"""
Create Korean-Korean synonym pairs for technical terms.

This adds Korean-Korean synonyms that are missing from the translation-based
synonym database, such as:
- 기계학습 ↔ 머신러닝
- 인공지능 ↔ AI
- 딥러닝 ↔ 심층학습
"""

import json
from pathlib import Path
from typing import List, Dict


# Korean technical term synonyms
KOREAN_SYNONYMS = [
    # AI/ML terms
    {"term1": "인공지능", "term2": "AI", "confidence": 1.0},
    {"term1": "기계학습", "term2": "머신러닝", "confidence": 1.0},
    {"term1": "기계학습", "term2": "machine learning", "confidence": 1.0},
    {"term1": "머신러닝", "term2": "machine learning", "confidence": 1.0},
    {"term1": "딥러닝", "term2": "deep learning", "confidence": 1.0},
    {"term1": "딥러닝", "term2": "심층학습", "confidence": 0.95},
    {"term1": "신경망", "term2": "neural network", "confidence": 1.0},
    {"term1": "신경망", "term2": "뉴럴네트워크", "confidence": 0.95},
    {"term1": "자연어처리", "term2": "NLP", "confidence": 1.0},
    {"term1": "자연어처리", "term2": "자연어 처리", "confidence": 1.0},
    {"term1": "컴퓨터비전", "term2": "computer vision", "confidence": 1.0},
    {"term1": "컴퓨터비전", "term2": "컴퓨터 비전", "confidence": 1.0},

    # Search/Information Retrieval terms
    {"term1": "검색엔진", "term2": "search engine", "confidence": 1.0},
    {"term1": "검색엔진", "term2": "검색 엔진", "confidence": 1.0},
    {"term1": "정보검색", "term2": "information retrieval", "confidence": 1.0},
    {"term1": "정보검색", "term2": "정보 검색", "confidence": 1.0},

    # Data Science terms
    {"term1": "데이터과학", "term2": "data science", "confidence": 1.0},
    {"term1": "데이터과학", "term2": "데이터 과학", "confidence": 1.0},
    {"term1": "빅데이터", "term2": "big data", "confidence": 1.0},
    {"term1": "빅데이터", "term2": "빅 데이터", "confidence": 1.0},

    # Computing terms
    {"term1": "클라우드", "term2": "cloud", "confidence": 1.0},
    {"term1": "클라우드컴퓨팅", "term2": "cloud computing", "confidence": 1.0},
    {"term1": "데이터베이스", "term2": "database", "confidence": 1.0},
    {"term1": "데이터베이스", "term2": "DB", "confidence": 0.95},
]


def create_synonym_entries(synonyms: List[Dict]) -> List[Dict]:
    """
    Create bidirectional synonym entries.

    For each synonym pair, create two entries:
    - term1 → term2
    - term2 → term1

    This ensures bidirectional lookup.
    """
    entries = []

    for syn in synonyms:
        term1 = syn["term1"]
        term2 = syn["term2"]
        confidence = syn["confidence"]

        # Add both directions
        entries.append({
            "korean": term1,
            "english": term2,
            "confidence": confidence,
            "sources": ["korean_synonyms"]
        })

        entries.append({
            "korean": term2,
            "english": term1,
            "confidence": confidence,
            "sources": ["korean_synonyms"]
        })

    return entries


def merge_with_existing(
    new_entries: List[Dict],
    existing_file: Path
) -> List[Dict]:
    """Merge new entries with existing synonym database."""
    # Load existing
    with open(existing_file, 'r', encoding='utf-8') as f:
        existing = json.load(f)

    print(f"Existing entries: {len(existing)}")
    print(f"New entries: {len(new_entries)}")

    # Simple merge (no deduplication for now)
    merged = existing + new_entries

    print(f"Merged entries: {len(merged)}")

    return merged


def main():
    print("=" * 70)
    print("Creating Korean-Korean Technical Term Synonyms")
    print("=" * 70)

    # Create bidirectional entries
    new_entries = create_synonym_entries(KOREAN_SYNONYMS)

    print(f"\nCreated {len(new_entries)} synonym entries from {len(KOREAN_SYNONYMS)} pairs")

    # Show samples
    print("\nSample entries:")
    for i, entry in enumerate(new_entries[:6], 1):
        print(f"  {i}. {entry['korean']} ↔ {entry['english']} (conf: {entry['confidence']})")

    # Merge with existing
    existing_file = Path("dataset/synonyms/combined_synonyms.json")
    merged = merge_with_existing(new_entries, existing_file)

    # Save merged
    output_file = Path("dataset/synonyms/combined_synonyms_v2.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved to {output_file}")

    # Backup original and replace
    backup_file = Path("dataset/synonyms/combined_synonyms_v1.json")
    if not backup_file.exists():
        import shutil
        shutil.copy2(existing_file, backup_file)
        print(f"✓ Backed up original to {backup_file}")

    # Replace with new version
    with open(existing_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"✓ Updated {existing_file}")

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Regenerate synonym training data:")
    print("     python scripts/generate_synonym_training_data.py")
    print("  2. Merge with baseline data:")
    print("     cat dataset/baseline_samples/train_baseline.jsonl \\")
    print("         dataset/baseline_samples/synonym_pairs.jsonl > \\")
    print("         dataset/baseline_samples/train_with_synonyms.jsonl")
    print("  3. Retrain: make train-baseline")
    print("=" * 70)


if __name__ == "__main__":
    main()
