#!/usr/bin/env python3
"""
Generate cross-lingual training data - FINAL VERSION.

This version uses a simple, targeted approach:
1. Extract Wikipedia docs containing SPECIFIC ML/AI terms
2. Create cross-lingual pairs with verified synonyms
3. Exclude common words that cause noise

Targets ONLY these core terms:
- 기계학습, machine learning, 머신러닝
- 인공지능, artificial intelligence
- 딥러닝, deep learning, 심층학습
- 신경망, neural network, 뉴럴네트워크
- 자연어처리, natural language processing
- 컴퓨터비전, computer vision

Input:
    - dataset/wikipedia/*_chunk_*.jsonl
    - dataset/synonyms/combined_synonyms.json

Output:
    - dataset/baseline_samples/crosslingual_pairs.jsonl (4,500 samples)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from glob import glob
from tqdm import tqdm


# Configuration
WIKIPEDIA_PATTERN = "dataset/wikipedia/*_chunk_*.jsonl"
SYNONYM_FILE = Path("dataset/synonyms/combined_synonyms.json")
OUTPUT_FILE = Path("dataset/baseline_samples/crosslingual_pairs.jsonl")

NUM_SAMPLES = 4500
MIN_CONFIDENCE = 0.8
MIN_DOC_LENGTH = 50
MAX_DOC_LENGTH = 500
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Specific target terms (NO generic words like "분류", "알고리즘")
TARGET_TERMS = {
    # Machine Learning - PRIMARY TARGET
    "기계학습", "기계 학습",
    "머신러닝", "머신 러닝",
    "machine learning",

    # Artificial Intelligence
    "인공지능", "인공 지능",
    "artificial intelligence",

    # Deep Learning
    "딥러닝", "딥 러닝",
    "심층학습", "심층 학습",
    "deep learning",

    # Neural Networks
    "신경망",
    "뉴럴네트워크", "뉴럴 네트워크",
    "neural network", "neural networks",

    # Natural Language Processing
    "자연어처리", "자연어 처리",
    "natural language processing",

    # Computer Vision
    "컴퓨터비전", "컴퓨터 비전",
    "computer vision",

    # Data Science
    "데이터과학", "데이터 과학",
    "data science",
    "빅데이터", "빅 데이터",
    "big data",
}

# Generic terms to EXCLUDE (too common, cause noise)
EXCLUDE_TERMS = {
    "분류", "classification", "category",
    "학습", "learning", "training",
    "알고리즘", "algorithm",
    "훈련",
    "AI", "ML", "DL", "NN", "IR", "CV",  # Short abbreviations
}


def load_synonym_map(synonym_file: Path, min_confidence: float) -> Dict[str, Set[str]]:
    """Load high-quality synonym mappings."""
    print(f"Loading synonyms from {synonym_file}")

    with open(synonym_file, 'r', encoding='utf-8') as f:
        synonym_data = json.load(f)

    synonym_map = defaultdict(set)

    for entry in synonym_data:
        korean = entry['korean']
        english = entry['english']
        confidence = entry['confidence']

        if confidence < min_confidence or korean == english:
            continue

        # Exclude generic terms
        if korean in EXCLUDE_TERMS or english in EXCLUDE_TERMS:
            continue

        # Bidirectional
        synonym_map[korean].add(english)
        synonym_map[english].add(korean)

    print(f"  Loaded {len(synonym_map)} terms with synonyms")
    return dict(synonym_map)


def extract_target_documents(
    wiki_files: List[str],
    target_terms: Set[str],
    min_length: int,
    max_length: int
) -> List[Dict]:
    """
    Extract Wikipedia documents containing target terms.

    Only processes Korean chunks for efficiency.
    """
    print(f"\nExtracting documents with target terms")

    # Focus on Korean chunks (most ML/AI terms appear here)
    ko_chunks = [f for f in wiki_files if '/ko_articles_chunk' in f]
    print(f"  Processing {len(ko_chunks)} Korean chunks")

    documents = []

    for file_path in tqdm(ko_chunks, desc="Processing chunks"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    article = json.loads(line.strip())

                    title = article.get('title', '')
                    text = article.get('text', '')

                    # Length filter
                    if len(text) < min_length or len(text) > max_length:
                        continue

                    # Check for target terms
                    combined = f"{title} {text}"
                    combined_lower = combined.lower()

                    found_terms = set()
                    for term in target_terms:
                        if term.lower() in combined_lower:
                            found_terms.add(term)

                    if found_terms:
                        documents.append({
                            'title': title,
                            'text': text,
                            'found_terms': list(found_terms),
                            'source_id': article.get('id', ''),
                            'source_url': article.get('url', ''),
                        })

                except json.JSONDecodeError:
                    continue

    print(f"  Found {len(documents)} documents with target terms")
    return documents


def generate_crosslingual_pairs(
    documents: List[Dict],
    synonym_map: Dict[str, Set[str]],
    num_samples: int
) -> List[Dict]:
    """
    Generate cross-lingual training pairs.

    For each document containing term A, create pairs with synonyms of A as queries.
    """
    print(f"\nGenerating {num_samples} cross-lingual training pairs")

    training_pairs = []

    # Shuffle for diversity
    random.shuffle(documents)

    for doc in tqdm(documents, desc="Creating pairs"):
        title = doc['title']
        text = doc['text']
        found_terms = doc['found_terms']

        # For each found term, create pairs with its synonyms
        for term in found_terms:
            if term not in synonym_map:
                continue

            synonyms = synonym_map[term]

            for synonym in synonyms:
                # Skip if synonym also in document (no cross-lingual benefit)
                if synonym.lower() in text.lower():
                    continue

                # Skip if synonym in exclude list
                if synonym in EXCLUDE_TERMS:
                    continue

                # Determine language
                is_korean = any('\uac00' <= c <= '\ud7a3' for c in synonym)

                # Create training pair
                training_pairs.append({
                    'query': synonym,
                    'document': text,
                    'query_type': 'crosslingual',
                    'doc_type': 'wikipedia',
                    'language': 'ko' if is_korean else 'en',
                    'expansion_terms': [term],
                    'title': title,
                    'source_id': doc.get('source_id', ''),
                    'source_url': doc.get('source_url', ''),
                })

                if len(training_pairs) >= num_samples:
                    break

            if len(training_pairs) >= num_samples:
                break

        if len(training_pairs) >= num_samples:
            break

    print(f"  Generated {len(training_pairs)} pairs")

    # Shuffle final pairs
    random.shuffle(training_pairs)

    return training_pairs[:num_samples]


def save_training_data(training_pairs: List[Dict], output_path: Path):
    """Save training pairs to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in training_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"\n✓ Saved {len(training_pairs)} pairs to {output_path}")


def print_statistics(training_pairs: List[Dict]):
    """Print statistics about generated pairs."""
    print("\n" + "=" * 70)
    print("Statistics")
    print("=" * 70)

    total = len(training_pairs)
    print(f"Total training pairs: {total}")

    # Language distribution
    ko_count = sum(1 for p in training_pairs if p['language'] == 'ko')
    en_count = sum(1 for p in training_pairs if p['language'] == 'en')
    print(f"  Korean queries: {ko_count} ({ko_count/total*100:.1f}%)")
    print(f"  English queries: {en_count} ({en_count/total*100:.1f}%)")

    # Unique counts
    unique_queries = len(set(p['query'] for p in training_pairs))
    unique_docs = len(set(p['document'] for p in training_pairs))
    print(f"  Unique queries: {unique_queries}")
    print(f"  Unique documents: {unique_docs}")
    print(f"  Avg pairs per document: {total/unique_docs:.1f}")

    # Count expansion terms
    expansion_counts = defaultdict(int)
    for pair in training_pairs:
        for term in pair.get('expansion_terms', []):
            expansion_counts[term] += 1

    print(f"\nTop expansion terms:")
    for term, count in sorted(expansion_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {term:30s}: {count:4d} pairs")

    # Sample pairs
    print("\nSample cross-lingual pairs:")
    for i, pair in enumerate(training_pairs[:10], 1):
        print(f"\n  {i}. Query: {pair['query']}")
        print(f"     Expansion: {pair['expansion_terms']}")
        print(f"     Title: {pair.get('title', 'N/A')}")
        print(f"     Document: {pair['document'][:120]}...")


def main():
    print("=" * 70)
    print("Generating Cross-Lingual Training Data - FINAL VERSION")
    print("=" * 70)

    # Load synonym database
    synonym_map = load_synonym_map(SYNONYM_FILE, MIN_CONFIDENCE)

    # Find Wikipedia files
    wiki_files = sorted(glob(WIKIPEDIA_PATTERN))
    print(f"\nWikipedia files: {len(wiki_files)}")

    # Extract documents with target terms
    documents = extract_target_documents(
        wiki_files=wiki_files,
        target_terms=TARGET_TERMS,
        min_length=MIN_DOC_LENGTH,
        max_length=MAX_DOC_LENGTH
    )

    if len(documents) == 0:
        print("\n❌ No documents found with target terms")
        return

    # Generate cross-lingual pairs
    training_pairs = generate_crosslingual_pairs(
        documents=documents,
        synonym_map=synonym_map,
        num_samples=NUM_SAMPLES
    )

    if len(training_pairs) == 0:
        print("\n❌ Failed to generate training pairs")
        return

    # Save results
    save_training_data(training_pairs, OUTPUT_FILE)

    # Print statistics
    print_statistics(training_pairs)

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Merge with baseline:")
    print("     cat dataset/baseline_samples/train_baseline.jsonl \\")
    print("         dataset/baseline_samples/crosslingual_pairs.jsonl > \\")
    print("         dataset/baseline_samples/train_with_crosslingual.jsonl")
    print("  2. Update config: configs/baseline_dgx.yaml")
    print("     train_patterns:")
    print("       - dataset/baseline_samples/train_with_crosslingual.jsonl")
    print("  3. Run: make train-baseline")
    print("=" * 70)


if __name__ == "__main__":
    main()
