#!/usr/bin/env python3
"""
Generate high-quality cross-lingual training data from Wikipedia.

This version targets ONLY core ML/AI/Search terms and extracts documents
directly from Wikipedia chunks for high quality matches.

Strategy:
    1. Focus on core terms only (exclude short abbreviations)
    2. Extract documents from Wikipedia chunks (not baseline)
    3. Create cross-lingual pairs with verified term presence
    4. Generate 4,500 high-quality training pairs

Input:
    - dataset/wikipedia/*_chunk_*.jsonl (Wikipedia articles)
    - dataset/synonyms/combined_synonyms.json (Synonym DB)

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
WIKIPEDIA_PATTERN = "dataset/wikipedia/*_chunk_*.jsonl"
SYNONYM_FILE = Path("dataset/synonyms/combined_synonyms.json")
OUTPUT_FILE = Path("dataset/baseline_samples/crosslingual_pairs.jsonl")

NUM_SAMPLES = 4500  # Target samples
MIN_CONFIDENCE = 0.7  # Synonym confidence
MIN_DOC_LENGTH = 50  # Minimum document length
MAX_DOC_LENGTH = 500  # Maximum document length
MIN_TERM_LENGTH = 4  # Exclude short abbreviations
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Core ML/AI/Search terms (excluding short abbreviations)
CORE_TERMS = {
    # Machine Learning (must include)
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
    "신경망", "신경 망",
    "뉴럴네트워크", "뉴럴 네트워크",
    "neural network", "neural networks",

    # Natural Language Processing
    "자연어처리", "자연어 처리",
    "자연언어처리", "자연언어 처리",
    "natural language processing",

    # Computer Vision
    "컴퓨터비전", "컴퓨터 비전",
    "computer vision",

    # Search/Information Retrieval
    "검색엔진", "검색 엔진",
    "search engine",
    "정보검색", "정보 검색",
    "information retrieval",

    # Data Science
    "데이터과학", "데이터 과학",
    "data science",
    "빅데이터", "빅 데이터",
    "big data",

    # Algorithms
    "알고리즘", "algorithm", "algorithms",

    # Optimization
    "최적화", "optimization",

    # Classification/Regression (longer forms only)
    "분류", "classification",
    "회귀", "regression",
}


def load_synonym_map(
    synonym_file: Path,
    min_confidence: float,
    min_term_length: int
) -> Dict[str, Set[str]]:
    """
    Load high-quality synonym mappings.

    Filter out short abbreviations to avoid false matches.
    """
    print(f"Loading synonyms from {synonym_file}")

    with open(synonym_file, 'r', encoding='utf-8') as f:
        synonym_data = json.load(f)

    synonym_map = defaultdict(set)

    for entry in synonym_data:
        korean = entry['korean']
        english = entry['english']
        confidence = entry['confidence']

        # Quality filters
        if confidence < min_confidence:
            continue
        if korean == english:
            continue
        # Exclude short abbreviations
        if len(korean) < min_term_length and len(english) < min_term_length:
            continue

        # Bidirectional
        synonym_map[korean].add(english)
        synonym_map[english].add(korean)

    print(f"  Loaded {len(synonym_map)} terms with synonyms")
    return dict(synonym_map)


def extract_core_documents(
    wiki_files: List[str],
    core_terms: Set[str],
    min_length: int,
    max_length: int,
    max_docs: int = 2000
) -> List[Dict]:
    """
    Extract Wikipedia documents containing core terms.

    Args:
        wiki_files: List of Wikipedia chunk files
        core_terms: Set of core terms to search for
        min_length: Minimum document length
        max_length: Maximum document length
        max_docs: Maximum documents to extract

    Returns:
        List of documents with metadata
    """
    print(f"\nExtracting documents with core terms from {len(wiki_files)} chunks")
    print(f"  Core terms: {len(core_terms)}")

    documents = []

    # Sample chunks to process (not all 92)
    sampled_chunks = random.sample(wiki_files, min(len(wiki_files), 40))

    for file_path in tqdm(sampled_chunks, desc="Processing chunks"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    article = json.loads(line.strip())

                    title = article.get('title', '')
                    text = article.get('text', '')

                    # Length filter
                    if len(text) < min_length or len(text) > max_length:
                        continue

                    # Check for core terms
                    text_lower = text.lower()
                    title_lower = title.lower()
                    combined_lower = f"{title_lower} {text_lower}"

                    found_terms = set()
                    for term in core_terms:
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

                        # Stop if we have enough
                        if len(documents) >= max_docs:
                            break

                except json.JSONDecodeError:
                    continue

        if len(documents) >= max_docs:
            break

    print(f"  Extracted {len(documents)} documents with core terms")
    return documents


def generate_crosslingual_pairs(
    documents: List[Dict],
    synonym_map: Dict[str, Set[str]],
    num_samples: int
) -> List[Dict]:
    """
    Generate cross-lingual training pairs.

    For each document containing term A, create pairs with synonyms of A.

    Args:
        documents: Wikipedia documents with found terms
        synonym_map: Term -> synonyms mapping
        num_samples: Target number of pairs

    Returns:
        List of training pairs
    """
    print(f"\nGenerating {num_samples} cross-lingual training pairs")

    training_pairs = []
    used_doc_ids = set()

    # Shuffle for diversity
    random.shuffle(documents)

    for doc in tqdm(documents, desc="Creating pairs"):
        title = doc['title']
        text = doc['text']
        found_terms = doc['found_terms']
        doc_id = f"{doc.get('source_id', '')}_{title}"

        # Skip if already used
        if doc_id in used_doc_ids:
            continue

        # For each found term, create cross-lingual pairs
        for term in found_terms:
            if term not in synonym_map:
                continue

            synonyms = synonym_map[term]

            # Create pairs with each synonym
            for synonym in synonyms:
                # Skip if synonym also appears in document (no cross-lingual benefit)
                if synonym.lower() in text.lower():
                    continue

                # Skip short abbreviations
                if len(synonym) < MIN_TERM_LENGTH:
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

                # Check limit
                if len(training_pairs) >= num_samples:
                    break

            if len(training_pairs) >= num_samples:
                break

        # Mark document as used
        used_doc_ids.add(doc_id)

        if len(training_pairs) >= num_samples:
            break

    print(f"  Generated {len(training_pairs)} pairs from {len(used_doc_ids)} documents")

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
    for term, count in sorted(expansion_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {term:30s}: {count:4d} pairs")

    # Sample pairs
    print("\nSample cross-lingual pairs:")
    for i, pair in enumerate(training_pairs[:5], 1):
        print(f"\n  {i}. Query: {pair['query']}")
        print(f"     Expansion: {pair['expansion_terms']}")
        print(f"     Title: {pair.get('title', 'N/A')}")
        print(f"     Document: {pair['document'][:150]}...")


def main():
    print("=" * 70)
    print("Generating High-Quality Cross-Lingual Training Data")
    print("=" * 70)

    # Load synonym database
    synonym_map = load_synonym_map(
        SYNONYM_FILE,
        MIN_CONFIDENCE,
        MIN_TERM_LENGTH
    )

    # Find Wikipedia chunk files
    wiki_files = sorted(glob(WIKIPEDIA_PATTERN))
    print(f"\nWikipedia files: {len(wiki_files)}")

    # Extract documents with core terms
    documents = extract_core_documents(
        wiki_files=wiki_files,
        core_terms=CORE_TERMS,
        min_length=MIN_DOC_LENGTH,
        max_length=MAX_DOC_LENGTH,
        max_docs=2000
    )

    if len(documents) == 0:
        print("\n❌ No documents found with core terms")
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
    print("  2. Update config train_patterns:")
    print("     - dataset/baseline_samples/train_with_crosslingual.jsonl")
    print("  3. Run: make train-baseline")
    print("=" * 70)


if __name__ == "__main__":
    main()
