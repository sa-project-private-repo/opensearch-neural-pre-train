#!/usr/bin/env python3
"""
Generate cross-lingual training data from existing baseline samples.

This script uses the existing train_baseline.jsonl (9,000 Wikipedia docs) and
creates cross-lingual query-document pairs by finding synonym terms in the docs.

Strategy:
    1. Load baseline Wikipedia documents (9,000 samples)
    2. Find documents containing priority ML/AI terms
    3. Create cross-lingual pairs with synonym queries
    4. Generate 4,500 training pairs (40-50% of total)

Example:
    Baseline doc: "딥러닝은... 기계학습 알고리즘..."
    Found term: "기계학습"
    Synonyms: ["machine learning", "ML", "머신러닝"]

    Generate pairs:
    - query="machine learning" → doc="딥러닝은... 기계학습 알고리즘..."
    - query="ML" → doc="딥러닝은... 기계학습 알고리즘..."
    - query="머신러닝" → doc="딥러닝은... 기계학습 알고리즘..."

Input:
    - dataset/baseline_samples/train_baseline.jsonl (9,000 docs)
    - dataset/synonyms/combined_synonyms.json (synonym DB)

Output:
    - dataset/baseline_samples/crosslingual_pairs.jsonl (4,500 samples)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from tqdm import tqdm


# Configuration
BASELINE_FILE = Path("dataset/baseline_samples/train_baseline.jsonl")
SYNONYM_FILE = Path("dataset/synonyms/combined_synonyms.json")
OUTPUT_FILE = Path("dataset/baseline_samples/crosslingual_pairs.jsonl")

NUM_SAMPLES = 4500  # Target 40-50% of total training data
MIN_CONFIDENCE = 0.7  # Higher confidence for better quality
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Priority ML/AI/Search terms (must include)
PRIORITY_TERMS = {
    # Machine Learning
    "기계학습", "기계 학습", "머신러닝", "머신 러닝",
    "machine learning", "ML",

    # Artificial Intelligence
    "인공지능", "인공 지능", "AI", "artificial intelligence",

    # Deep Learning
    "딥러닝", "딥 러닝", "심층학습", "심층 학습",
    "deep learning", "DL",

    # Neural Networks
    "신경망", "신경 망", "뉴럴네트워크", "뉴럴 네트워크",
    "neural network", "neural networks", "NN",

    # NLP
    "자연어처리", "자연어 처리", "자연언어처리", "자연언어 처리",
    "natural language processing", "NLP",

    # Computer Vision
    "컴퓨터비전", "컴퓨터 비전", "computer vision", "CV",

    # Search/IR
    "검색엔진", "검색 엔진", "search engine",
    "정보검색", "정보 검색", "information retrieval", "IR",

    # Data Science
    "데이터과학", "데이터 과학", "data science",
    "빅데이터", "빅 데이터", "big data",

    # Algorithms
    "알고리즘", "algorithm", "algorithms",

    # Training/Learning
    "훈련", "학습", "training", "learning",
}


def load_synonym_map(
    synonym_file: Path,
    min_confidence: float
) -> Dict[str, Set[str]]:
    """Load high-quality bidirectional synonym mappings."""
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

        # Bidirectional
        synonym_map[korean].add(english)
        synonym_map[english].add(korean)

    print(f"  Loaded {len(synonym_map)} terms with synonyms")
    return dict(synonym_map)


def load_baseline_documents(baseline_file: Path) -> List[Dict]:
    """Load existing baseline Wikipedia documents."""
    print(f"\nLoading baseline documents from {baseline_file}")

    documents = []
    with open(baseline_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                doc = json.loads(line.strip())
                documents.append(doc)
            except json.JSONDecodeError:
                continue

    print(f"  Loaded {len(documents)} baseline documents")
    return documents


def find_terms_in_text(text: str, search_terms: Set[str]) -> Set[str]:
    """
    Find which search terms appear in the text.

    Uses case-insensitive matching.
    """
    text_lower = text.lower()
    found_terms = set()

    for term in search_terms:
        if term.lower() in text_lower:
            found_terms.add(term)

    return found_terms


def create_crosslingual_pairs(
    documents: List[Dict],
    synonym_map: Dict[str, Set[str]],
    priority_terms: Set[str],
    num_samples: int
) -> List[Dict]:
    """
    Create cross-lingual training pairs from baseline documents.

    Strategy:
        1. Find documents containing priority terms
        2. For each found term, create pairs with ALL its synonyms
        3. This creates many pairs per document (cross-lingual expansion)

    Args:
        documents: Baseline Wikipedia documents
        synonym_map: Term -> synonyms mapping
        priority_terms: High-priority terms to focus on
        num_samples: Target number of pairs

    Returns:
        List of training pairs
    """
    print(f"\nCreating {num_samples} cross-lingual training pairs")

    # Find documents with priority terms
    print("  Finding documents with priority terms...")
    docs_with_terms = []

    for doc in tqdm(documents, desc="Scanning documents"):
        query = doc.get('query', '')
        document_text = doc.get('document', '')

        # Combine query and document for searching
        full_text = f"{query} {document_text}"

        # Find priority terms in this document
        found_terms = find_terms_in_text(full_text, priority_terms)

        if found_terms:
            docs_with_terms.append({
                'original_doc': doc,
                'found_terms': found_terms,
                'full_text': full_text,
            })

    print(f"  Found {len(docs_with_terms)} documents with priority terms")

    if len(docs_with_terms) == 0:
        print("  ⚠ No documents with priority terms found")
        return []

    # Generate cross-lingual pairs
    print("  Generating cross-lingual pairs...")
    training_pairs = []

    # Shuffle for diversity
    random.shuffle(docs_with_terms)

    for item in tqdm(docs_with_terms, desc="Creating pairs"):
        original_doc = item['original_doc']
        found_terms = item['found_terms']
        document_text = original_doc.get('document', '')

        # For each found term, create pairs with its synonyms
        for term in found_terms:
            if term not in synonym_map:
                continue

            synonyms = synonym_map[term]

            # Create a pair for each synonym
            for synonym in synonyms:
                # Skip if synonym is also in the document (no cross-lingual benefit)
                if synonym.lower() in document_text.lower():
                    continue

                # Determine language
                is_korean = any('\uac00' <= c <= '\ud7a3' for c in synonym)

                # Create training pair
                training_pairs.append({
                    'query': synonym,
                    'document': document_text,
                    'query_type': 'crosslingual',
                    'doc_type': 'wikipedia',
                    'language': 'ko' if is_korean else 'en',
                    'expansion_terms': [term],
                    'original_query': original_doc.get('query', ''),
                    'source_id': original_doc.get('source_id', ''),
                    'source_url': original_doc.get('source_url', ''),
                })

                # Check if we have enough
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
    """Print statistics about generated training pairs."""
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

    # Unique queries and documents
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
    for term, count in sorted(expansion_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {term:25s}: {count:4d} pairs")

    # Sample pairs
    print("\nSample cross-lingual pairs:")
    for i, pair in enumerate(training_pairs[:5], 1):
        print(f"\n  {i}. Query: {pair['query']}")
        print(f"     Expansion: {pair['expansion_terms']}")
        print(f"     Document: {pair['document'][:120]}...")


def main():
    print("=" * 70)
    print("Generating Cross-Lingual Training Data (Optimized)")
    print("=" * 70)

    # Load synonym database
    synonym_map = load_synonym_map(SYNONYM_FILE, MIN_CONFIDENCE)

    # Load baseline documents
    baseline_docs = load_baseline_documents(BASELINE_FILE)

    if len(baseline_docs) == 0:
        print("\n❌ No baseline documents found")
        return

    # Create cross-lingual pairs
    training_pairs = create_crosslingual_pairs(
        documents=baseline_docs,
        synonym_map=synonym_map,
        priority_terms=PRIORITY_TERMS,
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
    print("  2. Update config train_patterns:")
    print("     - dataset/baseline_samples/train_with_crosslingual.jsonl")
    print("  3. Run: make train-baseline")
    print("=" * 70)


if __name__ == "__main__":
    main()
