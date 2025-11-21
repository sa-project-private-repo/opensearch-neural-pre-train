#!/usr/bin/env python3
"""
Extract synonyms from ALL Wikipedia chunk files.

This script processes all 92 chunk files (12 Korean + 80 English) to extract
Korean-English synonym pairs, specifically targeting technical terms like
"기계 학습 또는 머신 러닝".

Input: dataset/wikipedia/*_chunk_*.jsonl (92 files, ~19GB)
Output: dataset/synonyms/combined_synonyms.json
"""

import sys
sys.path.append('..')

from pathlib import Path
import json
from typing import List, Dict
from src.data.synonym_extractor import SynonymExtractor, SynonymAugmenter
from glob import glob
from tqdm import tqdm


def extract_from_multiple_files(
    file_paths: List[str],
    language: str,
    extraction_method: str,
    extractor: SynonymExtractor
) -> List[Dict]:
    """
    Extract synonyms from multiple chunk files.

    Args:
        file_paths: List of chunk file paths
        language: "ko" or "en"
        extraction_method: "parentheses" or "first_sentence"
        extractor: SynonymExtractor instance

    Returns:
        Combined list of synonyms from all chunks
    """
    all_synonyms = []

    for file_path in tqdm(file_paths, desc=f"Processing {language} chunks"):
        if extraction_method == "parentheses":
            synonyms = extractor.extract_from_parentheses(
                articles_path=file_path,
                language=language,
            )
        elif extraction_method == "first_sentence":
            synonyms = extractor.extract_from_first_sentence(
                articles_path=file_path,
                language=language,
            )
        else:
            raise ValueError(f"Unknown extraction method: {extraction_method}")

        all_synonyms.extend(synonyms)

        # Print progress
        print(f"  {Path(file_path).name}: {len(synonyms)} synonyms")

    # Deduplicate
    deduplicated = extractor._deduplicate_synonyms(all_synonyms)
    print(f"  Total: {len(all_synonyms)} → {len(deduplicated)} unique synonyms\n")

    return deduplicated


def main():
    print("=" * 70)
    print("Extracting Synonyms from ALL Wikipedia Chunks")
    print("=" * 70)

    # Find all chunk files
    ko_chunks = sorted(glob("dataset/wikipedia/ko_articles_chunk_*.jsonl"))
    en_chunks = sorted(glob("dataset/wikipedia/en_articles_chunk_*.jsonl"))

    print(f"\nFound:")
    print(f"  Korean chunks: {len(ko_chunks)}")
    print(f"  English chunks: {len(en_chunks)}")
    print(f"  Total: {len(ko_chunks) + len(en_chunks)} chunks")

    # Initialize extractor
    extractor = SynonymExtractor()

    # ========================================
    # 1. Extract from Parentheses (Korean)
    # ========================================
    print("\n" + "=" * 70)
    print("1. Extracting from Parentheses (Korean articles)")
    print("=" * 70)

    paren_ko_synonyms = extract_from_multiple_files(
        file_paths=ko_chunks,
        language="ko",
        extraction_method="parentheses",
        extractor=extractor,
    )

    print(f"Korean parentheses: {len(paren_ko_synonyms)} synonyms")

    # Show samples
    print("\nSample Korean parentheses extractions:")
    for i, syn in enumerate(paren_ko_synonyms[:10], 1):
        print(f"  {i}. {syn['korean']:20s} → {syn['english']}")

    # ========================================
    # 2. Extract from First Sentences (Korean)
    # ========================================
    print("\n" + "=" * 70)
    print("2. Extracting from First Sentences (Korean articles)")
    print("=" * 70)

    def_ko_synonyms = extract_from_multiple_files(
        file_paths=ko_chunks,
        language="ko",
        extraction_method="first_sentence",
        extractor=extractor,
    )

    print(f"Korean definitions: {len(def_ko_synonyms)} synonyms")

    # ========================================
    # 3. Combine and Filter
    # ========================================
    print("\n" + "=" * 70)
    print("3. Combining and Filtering Synonyms")
    print("=" * 70)

    # Combine all sources
    combined_synonyms = extractor.combine_and_filter(
        synonym_lists=[
            paren_ko_synonyms,
            def_ko_synonyms,
        ],
        min_confidence=0.5,
    )

    print(f"Combined unique synonyms: {len(combined_synonyms)}")

    # ========================================
    # 4. Augment with Variations
    # ========================================
    print("\n" + "=" * 70)
    print("4. Augmenting with Variations")
    print("=" * 70)

    augmenter = SynonymAugmenter()
    augmented_synonyms = augmenter.generate_variations(combined_synonyms)

    print(f"Augmented: {len(combined_synonyms)} → {len(augmented_synonyms)} synonyms")

    # ========================================
    # 5. Save Results
    # ========================================
    output_path = Path("dataset/synonyms/combined_synonyms.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(augmented_synonyms, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved {len(augmented_synonyms)} synonyms to {output_path}")

    # ========================================
    # 6. Verify Target Synonyms
    # ========================================
    print("\n" + "=" * 70)
    print("5. Verifying Target Technical Terms")
    print("=" * 70)

    # Check for target terms
    target_terms = [
        ("기계학습", "머신러닝"),
        ("기계학습", "machine learning"),
        ("기계 학습", "머신 러닝"),
        ("인공지능", "AI"),
        ("딥러닝", "deep learning"),
    ]

    found_targets = []
    for korean, english in target_terms:
        # Search in augmented synonyms
        for syn in augmented_synonyms:
            if (syn['korean'].lower() == korean.lower() and
                english.lower() in syn['english'].lower()):
                found_targets.append((korean, english))
                print(f"  ✓ Found: {korean} ↔ {english}")
                break
        else:
            print(f"  ✗ Missing: {korean} ↔ {english}")

    print(f"\nFound {len(found_targets)}/{len(target_terms)} target term pairs")

    # ========================================
    # 7. Statistics
    # ========================================
    print("\n" + "=" * 70)
    print("6. Statistics")
    print("=" * 70)

    from collections import Counter

    # Confidence distribution
    confidences = [syn['confidence'] for syn in augmented_synonyms]
    print(f"Confidence:")
    print(f"  Mean: {sum(confidences) / len(confidences):.2f}")
    print(f"  Min: {min(confidences):.2f}")
    print(f"  Max: {max(confidences):.2f}")

    # Source distribution
    all_sources = []
    for syn in augmented_synonyms:
        all_sources.extend(syn.get('sources', []))

    source_counts = Counter(all_sources)
    print(f"\nSources:")
    for source, count in source_counts.most_common():
        print(f"  {source:25s}: {count:6d}")

    # ========================================
    # 8. Show High-Quality Samples
    # ========================================
    print("\n" + "=" * 70)
    print("7. High-Quality Synonym Samples")
    print("=" * 70)

    # Sort by confidence
    high_quality = sorted(
        augmented_synonyms,
        key=lambda x: x['confidence'],
        reverse=True
    )[:30]

    for i, syn in enumerate(high_quality, 1):
        sources = ", ".join(syn.get('sources', []))
        print(f"{i:2d}. {syn['korean']:25s} → {syn['english']:30s} [{syn['confidence']:.2f}]")

    print("\n" + "=" * 70)
    print("Next Steps:")
    print("  1. Regenerate synonym training data:")
    print("     python scripts/generate_synonym_training_data.py")
    print("  2. Merge with baseline:")
    print("     cat dataset/baseline_samples/train_baseline.jsonl \\")
    print("         dataset/baseline_samples/synonym_pairs.jsonl > \\")
    print("         dataset/baseline_samples/train_with_synonyms.jsonl")
    print("  3. Retrain model:")
    print("     make train-baseline")
    print("=" * 70)


if __name__ == "__main__":
    main()
