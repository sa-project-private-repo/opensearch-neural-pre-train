#!/usr/bin/env python3
"""
Generate cross-lingual training data from Wikipedia articles.

This script extracts Wikipedia documents containing synonym terms and creates
cross-lingual query-document pairs for teaching term expansion.

Example:
    - Original doc: "딥러닝은... 기계학습 알고리즘..."
    - Create pair: query="machine learning" → document=original_doc
    - This teaches the model that "machine learning" query should match "기계학습" in docs

Input:
    - dataset/wikipedia/*_chunk_*.jsonl (Wikipedia articles)
    - dataset/synonyms/combined_synonyms.json (Synonym database)

Output:
    - dataset/baseline_samples/crosslingual_pairs.jsonl (4,500 samples)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict
from glob import glob
from tqdm import tqdm


# Configuration
SYNONYM_FILE = Path("dataset/synonyms/combined_synonyms.json")
WIKIPEDIA_PATTERN = "dataset/wikipedia/*_chunk_*.jsonl"
OUTPUT_FILE = Path("dataset/baseline_samples/crosslingual_pairs.jsonl")

NUM_SAMPLES = 4500  # Target 40-50% of total training data
MIN_CONFIDENCE = 0.7  # Higher confidence for better quality
MIN_DOC_LENGTH = 50  # Minimum document length (characters)
MAX_DOC_LENGTH = 500  # Maximum document length for efficiency
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def load_synonym_pairs(
    synonym_file: Path,
    min_confidence: float
) -> Dict[str, Set[str]]:
    """
    Load high-quality synonym pairs.

    Returns:
        Dict mapping term -> set of synonyms
    """
    print(f"Loading synonyms from {synonym_file}")

    with open(synonym_file, 'r', encoding='utf-8') as f:
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


def extract_documents_with_terms(
    wiki_files: List[str],
    target_terms: Set[str],
    min_length: int,
    max_length: int
) -> List[Dict]:
    """
    Extract Wikipedia documents containing target terms.

    Args:
        wiki_files: List of Wikipedia chunk file paths
        target_terms: Set of terms to search for
        min_length: Minimum document length
        max_length: Maximum document length

    Returns:
        List of documents with metadata
    """
    print(f"\nExtracting documents containing {len(target_terms)} target terms")

    # Group terms by first character for efficient searching
    term_index = defaultdict(set)
    for term in target_terms:
        if term:
            term_index[term[0].lower()].add(term.lower())

    documents = []

    for file_path in tqdm(wiki_files, desc="Processing chunks"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    article = json.loads(line.strip())

                    title = article.get('title', '')
                    text = article.get('text', '')

                    # Skip short or long documents
                    if len(text) < min_length or len(text) > max_length:
                        continue

                    # Check if any target term appears in text
                    text_lower = text.lower()
                    found_terms = set()

                    for term in target_terms:
                        if term.lower() in text_lower:
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
    Generate cross-lingual query-document training pairs.

    Strategy:
        1. For each document containing term A
        2. Create training pairs with synonyms of A as queries
        3. This teaches: query(synonym_B) → document(containing A)

    Example:
        Doc: "딥러닝은... 기계학습 알고리즘..."
        Found term: "기계학습"
        Synonyms: ["machine learning", "ML"]

        Generate pairs:
        - query="machine learning" → doc="딥러닝은... 기계학습 알고리즘..."
        - query="ML" → doc="딥러닝은... 기계학습 알고리즘..."

    Args:
        documents: List of Wikipedia documents with found terms
        synonym_map: Dictionary of term -> synonyms
        num_samples: Target number of training pairs

    Returns:
        List of training pair dictionaries
    """
    print(f"\nGenerating {num_samples} cross-lingual training pairs")

    training_pairs = []

    # Priority terms for ML/AI domain
    priority_terms = {
        "기계학습", "머신러닝", "machine learning", "ML",
        "인공지능", "AI", "artificial intelligence",
        "딥러닝", "deep learning", "심층학습",
        "신경망", "neural network", "뉴럴네트워크",
        "자연어처리", "자연어 처리", "NLP",
        "컴퓨터비전", "컴퓨터 비전", "computer vision",
        "검색엔진", "검색 엔진", "search engine",
        "빅데이터", "빅 데이터", "big data",
    }

    # Separate priority and regular documents
    priority_docs = []
    regular_docs = []

    for doc in documents:
        # Check if any found term is in priority list
        is_priority = any(term in priority_terms for term in doc['found_terms'])

        if is_priority:
            priority_docs.append(doc)
        else:
            regular_docs.append(doc)

    print(f"  Priority documents: {len(priority_docs)}")
    print(f"  Regular documents: {len(regular_docs)}")

    # Sample documents: priority first
    num_priority = min(len(priority_docs), num_samples // 2)
    num_regular = min(len(regular_docs), num_samples - num_priority)

    sampled_docs = (
        random.sample(priority_docs, num_priority) if num_priority > 0 else []
    ) + (
        random.sample(regular_docs, num_regular) if num_regular > 0 else []
    )

    # Shuffle for diversity
    random.shuffle(sampled_docs)

    for doc in tqdm(sampled_docs[:num_samples], desc="Generating pairs"):
        title = doc['title']
        text = doc['text']
        found_terms = doc['found_terms']

        # For each found term, create pairs with its synonyms
        for term in found_terms:
            if term not in synonym_map:
                continue

            synonyms = synonym_map[term]

            # Create cross-lingual pairs
            for synonym in synonyms:
                # Determine language
                is_korean = any('\uac00' <= c <= '\ud7a3' for c in synonym)

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

                # Stop if we have enough pairs
                if len(training_pairs) >= num_samples:
                    break

            if len(training_pairs) >= num_samples:
                break

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
    print("Generating Cross-Lingual Training Data from Wikipedia")
    print("=" * 70)

    # Load synonym database
    synonym_map = load_synonym_pairs(SYNONYM_FILE, MIN_CONFIDENCE)

    # Get all target terms
    all_terms = set()
    for term, synonyms in synonym_map.items():
        all_terms.add(term)
        all_terms.update(synonyms)

    print(f"\nTarget terms: {len(all_terms)}")

    # Show sample terms
    print("\nSample target terms:")
    sample_terms = list(all_terms)[:20]
    for i, term in enumerate(sample_terms, 1):
        synonyms = synonym_map.get(term, set())
        if synonyms:
            print(f"  {i}. {term:25s} ↔ {', '.join(list(synonyms)[:3])}")

    # Find Wikipedia chunk files
    wiki_files = sorted(glob(WIKIPEDIA_PATTERN))
    print(f"\nWikipedia files: {len(wiki_files)}")

    # Extract documents containing target terms
    documents = extract_documents_with_terms(
        wiki_files=wiki_files,
        target_terms=all_terms,
        min_length=MIN_DOC_LENGTH,
        max_length=MAX_DOC_LENGTH
    )

    if len(documents) == 0:
        print("\n❌ No documents found with target terms")
        return

    # Generate cross-lingual training pairs
    training_pairs = generate_crosslingual_pairs(
        documents=documents,
        synonym_map=synonym_map,
        num_samples=NUM_SAMPLES
    )

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
    print(f"  Korean queries: {ko_count} ({ko_count/len(training_pairs)*100:.1f}%)")
    print(f"  English queries: {en_count} ({en_count/len(training_pairs)*100:.1f}%)")

    # Count unique queries
    unique_queries = len(set(p['query'] for p in training_pairs))
    print(f"  Unique queries: {unique_queries}")

    # Show samples
    print("\nSample cross-lingual pairs:")
    for i, pair in enumerate(training_pairs[:5], 1):
        print(f"\n  {i}. Query: {pair['query']}")
        print(f"     Document: {pair['document'][:150]}...")
        print(f"     Expansion: {pair['expansion_terms']}")

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Merge with baseline data:")
    print("     cat dataset/baseline_samples/train_baseline.jsonl \\")
    print("         dataset/baseline_samples/crosslingual_pairs.jsonl > \\")
    print("         dataset/baseline_samples/train_with_crosslingual.jsonl")
    print("  2. Update config to use train_with_crosslingual.jsonl")
    print("  3. Run: make train-baseline")
    print("=" * 70)


if __name__ == "__main__":
    main()
