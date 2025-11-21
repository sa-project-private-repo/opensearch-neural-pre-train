#!/usr/bin/env python3
"""
Generate synonym-based training data for term expansion learning.

This script creates (query, document) pairs from synonym data to teach
the model to expand queries with related terms (e.g., AI ↔ 인공지능).

Input: dataset/synonyms/combined_synonyms.json
Output: dataset/baseline_samples/synonym_pairs.jsonl
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
from tqdm import tqdm


# Configuration
SYNONYM_FILE = Path("dataset/synonyms/combined_synonyms.json")
OUTPUT_FILE = Path("dataset/baseline_samples/synonym_pairs.jsonl")
MIN_CONFIDENCE = 0.8
NUM_SAMPLES = 2000  # Add 2000 synonym pairs to baseline
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def load_synonyms(file_path: Path, min_confidence: float) -> Dict[str, Set[str]]:
    """
    Load synonym data and create bidirectional mapping.

    Args:
        file_path: Path to synonym JSON file
        min_confidence: Minimum confidence threshold

    Returns:
        Dictionary mapping terms to their synonyms
    """
    print(f"Loading synonyms from {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        synonym_data = json.load(f)

    # Build bidirectional synonym map
    synonym_map = defaultdict(set)

    for entry in synonym_data:
        korean = entry['korean']
        english = entry['english']
        confidence = entry['confidence']

        # Only use high-confidence synonyms
        if confidence < min_confidence:
            continue

        # Skip identical terms
        if korean == english:
            continue

        # Add bidirectional mapping
        synonym_map[korean].add(english)
        synonym_map[english].add(korean)

    print(f"  Loaded {len(synonym_map)} terms with synonyms")
    return dict(synonym_map)


def generate_synonym_pairs(
    synonym_map: Dict[str, Set[str]],
    num_samples: int
) -> List[Dict]:
    """
    Generate (query, document) training pairs from synonyms.

    Creates diverse training examples:
    1. Simple synonym replacement: "AI" → "인공지능은 중요한 기술이다"
    2. Multiple synonyms: "AI" → "인공지능과 머신러닝은 관련 기술이다"
    3. Context sentences with synonyms

    Args:
        synonym_map: Dictionary of term -> synonyms
        num_samples: Number of training pairs to generate

    Returns:
        List of training pair dictionaries
    """
    print(f"\nGenerating {num_samples} synonym-based training pairs")

    # Template sentences for different patterns
    templates_ko = [
        "{term}은 {syn}와 관련된 개념이다.",
        "{term}는 {syn}를 포함하는 기술이다.",
        "{term}와 {syn}는 유사한 의미를 가진다.",
        "{term} 또는 {syn}로 불리는 기술",
        "{term}({syn})는 중요한 개념이다.",
    ]

    templates_en = [
        "{term} is related to {syn}.",
        "{term} includes {syn} technology.",
        "{term} and {syn} have similar meanings.",
        "Technology called {term} or {syn}",
        "{term} ({syn}) is an important concept.",
    ]

    training_pairs = []

    # Priority Korean technical terms (must include)
    priority_terms = [
        "기계학습", "머신러닝", "machine learning",
        "인공지능", "AI", "artificial intelligence",
        "딥러닝", "deep learning", "심층학습",
        "신경망", "neural network", "뉴럴네트워크",
        "자연어처리", "NLP",
        "컴퓨터비전", "computer vision",
        "검색엔진", "search engine",
        "빅데이터", "big data",
    ]

    # Get all terms with synonyms
    terms_with_synonyms = [(term, syns) for term, syns in synonym_map.items()
                          if len(syns) > 0]

    # Separate priority and regular terms
    priority_pairs = []
    regular_pairs = []

    for term, syns in terms_with_synonyms:
        if term in priority_terms:
            priority_pairs.append((term, syns))
        else:
            regular_pairs.append((term, syns))

    print(f"  Priority terms: {len(priority_pairs)}")
    print(f"  Regular terms: {len(regular_pairs)}")

    # Sample: priority first, then random
    num_priority = min(len(priority_pairs), num_samples // 2)  # Use half for priority
    num_regular = min(len(regular_pairs), num_samples - num_priority)

    sampled_terms = (
        priority_pairs[:num_priority] +
        random.sample(regular_pairs, num_regular)
    )

    for term, synonyms in tqdm(sampled_terms, desc="Generating pairs"):
        # Pick a random synonym
        synonym = random.choice(list(synonyms))

        # Determine language and select appropriate templates
        is_korean = any('\uac00' <= c <= '\ud7a3' for c in term)
        templates = templates_ko if is_korean else templates_en

        # Generate document using template
        template = random.choice(templates)

        # Create multiple variations
        # 1. Query: term, Document: synonym in context
        document1 = template.format(term=synonym, syn=term)
        training_pairs.append({
            'query': term,
            'document': document1,
            'query_type': 'synonym',
            'doc_type': 'expansion',
            'language': 'ko' if is_korean else 'en',
            'expansion_terms': [synonym],
        })

        # 2. Query: synonym, Document: term in context (bidirectional)
        document2 = template.format(term=term, syn=synonym)
        training_pairs.append({
            'query': synonym,
            'document': document2,
            'query_type': 'synonym',
            'doc_type': 'expansion',
            'language': 'ko' if is_korean else 'en',
            'expansion_terms': [term],
        })

        # Stop if we have enough pairs
        if len(training_pairs) >= num_samples:
            break

    print(f"  Generated {len(training_pairs)} training pairs")
    return training_pairs[:num_samples]


def save_training_data(training_pairs: List[Dict], output_path: Path):
    """Save training pairs to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in training_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"\n✓ Saved {len(training_pairs)} pairs to {output_path}")


def main():
    print("=" * 70)
    print("Generating Synonym-Based Training Data")
    print("=" * 70)

    # Load synonyms
    synonym_map = load_synonyms(SYNONYM_FILE, MIN_CONFIDENCE)

    # Generate training pairs
    training_pairs = generate_synonym_pairs(synonym_map, NUM_SAMPLES)

    # Save to file
    save_training_data(training_pairs, OUTPUT_FILE)

    # Print statistics
    print("\n" + "=" * 70)
    print("Statistics")
    print("=" * 70)
    print(f"Total training pairs: {len(training_pairs)}")

    # Count by language
    ko_count = sum(1 for p in training_pairs if p['language'] == 'ko')
    en_count = sum(1 for p in training_pairs if p['language'] == 'en')
    print(f"  Korean pairs: {ko_count}")
    print(f"  English pairs: {en_count}")

    # Show samples
    print("\nSample pairs:")
    for i, pair in enumerate(training_pairs[:3], 1):
        print(f"\n  {i}. Query: {pair['query']}")
        print(f"     Document: {pair['document']}")
        print(f"     Expansion: {pair['expansion_terms']}")

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Merge with baseline data:")
    print("     cat dataset/baseline_samples/train_baseline.jsonl \\")
    print("         dataset/baseline_samples/synonym_pairs.jsonl > \\")
    print("         dataset/baseline_samples/train_with_synonyms.jsonl")
    print("  2. Update config to use train_with_synonyms.jsonl")
    print("  3. Run: make train-baseline")
    print("=" * 70)


if __name__ == "__main__":
    main()
