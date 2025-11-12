"""
Negative sampling strategies for neural sparse retrieval.

This module implements hard negative mining using BM25 and other strategies
to improve training data quality.
"""

from typing import List, Tuple, Dict, Optional
import random

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def add_hard_negatives_bm25(
    qd_pairs: List[Tuple[str, str, float]],
    documents: List[str],
    tokenizer,
    num_hard_negatives: int = 2,
    top_k: int = 100,
    exclude_positive: bool = True,
) -> List[Tuple[str, str, float]]:
    """
    Add hard negative samples using BM25 retrieval.

    Hard negatives are documents that lexically match the query but are not
    the correct answer. This makes training more challenging and effective.

    Args:
        qd_pairs: List of (query, positive_doc, relevance=1.0) tuples
        documents: Full document corpus
        tokenizer: Tokenizer for text processing
        num_hard_negatives: Number of hard negatives per query
        top_k: Consider top-k BM25 results as candidates
        exclude_positive: Don't use positive doc as hard negative

    Returns:
        Augmented list with hard negatives added

    Example:
        >>> qd_pairs = [("AI research", "Paper about AI", 1.0), ...]
        >>> augmented = add_hard_negatives_bm25(
        ...     qd_pairs, documents, tokenizer, num_hard_negatives=2
        ... )
        >>> # Now includes hard negatives that match "AI research" but wrong docs
    """
    print(f"\n🎯 Adding Hard Negatives (BM25-based)")
    print(f"   Num hard negatives per query: {num_hard_negatives}")
    print(f"   Top-K candidates: {top_k}")

    # Tokenize all documents for BM25
    print(f"   Tokenizing {len(documents):,} documents...")
    tokenized_docs = []
    for doc in tqdm(documents, desc="Tokenizing docs"):
        tokens = tokenizer.tokenize(doc.lower())
        tokenized_docs.append(tokens)

    # Build BM25 index
    print(f"   Building BM25 index...")
    bm25 = BM25Okapi(tokenized_docs)

    # Add hard negatives
    augmented_pairs = []
    hard_neg_count = 0

    print(f"   Mining hard negatives...")
    for query, pos_doc, relevance in tqdm(
        qd_pairs, desc="Mining hard negatives"
    ):
        # Add original positive pair
        augmented_pairs.append((query, pos_doc, relevance))

        # Only add hard negatives for positive pairs
        if relevance != 1.0:
            continue

        # Tokenize query
        query_tokens = tokenizer.tokenize(query.lower())

        # Get BM25 scores
        doc_scores = bm25.get_scores(query_tokens)

        # Get top-k candidates
        top_indices = np.argsort(doc_scores)[-top_k:][::-1]

        # Filter out positive document
        hard_neg_candidates = []
        for idx in top_indices:
            candidate_doc = documents[idx]

            if exclude_positive and candidate_doc == pos_doc:
                continue

            hard_neg_candidates.append(candidate_doc)

            if len(hard_neg_candidates) >= num_hard_negatives:
                break

        # Add hard negatives
        for hard_neg_doc in hard_neg_candidates:
            augmented_pairs.append((query, hard_neg_doc, 0.0))
            hard_neg_count += 1

    print(f"\n✓ Added {hard_neg_count:,} hard negative samples")
    print(f"  Total pairs: {len(qd_pairs):,} → {len(augmented_pairs):,}")

    return augmented_pairs


def add_random_negatives(
    qd_pairs: List[Tuple[str, str, float]],
    documents: List[str],
    num_random_negatives: int = 2,
    exclude_positive: bool = True,
) -> List[Tuple[str, str, float]]:
    """
    Add random negative samples.

    Args:
        qd_pairs: List of (query, positive_doc, relevance=1.0) tuples
        documents: Full document corpus
        num_random_negatives: Number of random negatives per query
        exclude_positive: Don't use positive doc as negative

    Returns:
        Augmented list with random negatives added

    Example:
        >>> augmented = add_random_negatives(
        ...     qd_pairs, documents, num_random_negatives=2
        ... )
    """
    print(f"\n🎲 Adding Random Negatives")
    print(f"   Num random negatives per query: {num_random_negatives}")

    doc_list = list(documents)
    n_docs = len(doc_list)

    # Create document → index mapping for positive docs
    doc_to_idx = {doc: idx for idx, doc in enumerate(doc_list)}

    augmented_pairs = []
    random_neg_count = 0

    for query, pos_doc, relevance in qd_pairs:
        # Add original pair
        augmented_pairs.append((query, pos_doc, relevance))

        # Only add negatives for positive pairs
        if relevance != 1.0:
            continue

        pos_idx = doc_to_idx.get(pos_doc, -1)

        for _ in range(num_random_negatives):
            # Random sampling
            neg_idx = random.randint(0, n_docs - 1)

            # Avoid positive document
            if exclude_positive and neg_idx == pos_idx:
                neg_idx = (neg_idx + 1) % n_docs

            neg_doc = doc_list[neg_idx]
            augmented_pairs.append((query, neg_doc, 0.0))
            random_neg_count += 1

    print(f"✓ Added {random_neg_count:,} random negative samples")
    print(f"  Total pairs: {len(qd_pairs):,} → {len(augmented_pairs):,}")

    return augmented_pairs


def add_mixed_negatives(
    qd_pairs: List[Tuple[str, str, float]],
    documents: List[str],
    tokenizer,
    num_hard: int = 2,
    num_random: int = 2,
    hard_ratio: float = 0.5,
) -> List[Tuple[str, str, float]]:
    """
    Add mixed hard and random negatives.

    Args:
        qd_pairs: List of (query, positive_doc, relevance) tuples
        documents: Full document corpus
        tokenizer: Tokenizer
        num_hard: Number of hard negatives per query
        num_random: Number of random negatives per query
        hard_ratio: Ratio of hard to random negatives (0.0-1.0)

    Returns:
        Augmented pairs with mixed negatives

    Example:
        >>> augmented = add_mixed_negatives(
        ...     qd_pairs, documents, tokenizer,
        ...     num_hard=2, num_random=2, hard_ratio=0.5
        ... )
    """
    print(f"\n🎯🎲 Adding Mixed Negatives (Hard + Random)")
    print(f"   Hard negatives: {num_hard}")
    print(f"   Random negatives: {num_random}")
    print(f"   Hard ratio: {hard_ratio:.1%}")

    # Add hard negatives
    if num_hard > 0 and hard_ratio > 0:
        augmented_pairs = add_hard_negatives_bm25(
            qd_pairs=qd_pairs,
            documents=documents,
            tokenizer=tokenizer,
            num_hard_negatives=num_hard,
        )
    else:
        augmented_pairs = list(qd_pairs)

    # Add random negatives
    if num_random > 0 and hard_ratio < 1.0:
        # Extract only positive pairs for random neg sampling
        positive_pairs = [
            (q, d, r) for q, d, r in augmented_pairs if r == 1.0
        ]

        random_augmented = add_random_negatives(
            qd_pairs=positive_pairs,
            documents=documents,
            num_random_negatives=num_random,
        )

        # Merge: keep positives and hard negs, add random negs
        final_pairs = []
        for q, d, r in augmented_pairs:
            final_pairs.append((q, d, r))

        # Add only the new random negatives
        for q, d, r in random_augmented:
            if r == 0.0:  # Only negatives
                final_pairs.append((q, d, r))

        augmented_pairs = final_pairs

    print(f"\n✓ Total pairs after mixed sampling: {len(augmented_pairs):,}")

    return augmented_pairs


def balance_positive_negative_ratio(
    qd_pairs: List[Tuple[str, str, float]],
    target_ratio: float = 1.0,
) -> List[Tuple[str, str, float]]:
    """
    Balance the ratio of positive to negative samples.

    Args:
        qd_pairs: List of (query, doc, relevance) tuples
        target_ratio: Target positive/negative ratio (default: 1.0 = equal)

    Returns:
        Balanced dataset

    Example:
        >>> balanced = balance_positive_negative_ratio(
        ...     qd_pairs, target_ratio=1.0
        ... )
    """
    positive_pairs = [(q, d, r) for q, d, r in qd_pairs if r == 1.0]
    negative_pairs = [(q, d, r) for q, d, r in qd_pairs if r == 0.0]

    num_pos = len(positive_pairs)
    num_neg = len(negative_pairs)

    print(f"\n⚖️  Balancing Positive/Negative Ratio")
    print(f"   Original - Positives: {num_pos:,}, Negatives: {num_neg:,}")
    print(f"   Current ratio: {num_pos}/{num_neg} = {num_pos/max(num_neg,1):.2f}")
    print(f"   Target ratio: {target_ratio:.2f}")

    target_negatives = int(num_pos / target_ratio)

    if num_neg > target_negatives:
        # Downsample negatives
        negative_pairs = random.sample(negative_pairs, target_negatives)
        print(f"   Downsampled negatives: {num_neg:,} → {len(negative_pairs):,}")
    elif num_neg < target_negatives:
        # Could upsample negatives, but usually not needed
        print(f"   ⚠️  Not enough negatives. Consider adding more.")

    balanced_pairs = positive_pairs + negative_pairs
    random.shuffle(balanced_pairs)

    print(f"✓ Balanced pairs: {len(balanced_pairs):,}")

    return balanced_pairs


def create_query_document_pairs_from_corpus(
    documents: List[str],
    tokenizer,
    queries_per_doc: int = 1,
    max_query_length: int = 64,
) -> List[Tuple[str, str, float]]:
    """
    Generate query-document pairs from corpus (unsupervised).

    Extracts n-grams or sentences from documents as pseudo-queries.

    Args:
        documents: List of documents
        tokenizer: Tokenizer
        queries_per_doc: Number of queries to generate per document
        max_query_length: Maximum query length in characters

    Returns:
        List of (query, document, relevance=1.0) pairs

    Example:
        >>> qd_pairs = create_query_document_pairs_from_corpus(
        ...     documents, tokenizer, queries_per_doc=2
        ... )
    """
    print(f"\n📝 Generating Query-Document Pairs from Corpus")
    print(f"   Documents: {len(documents):,}")
    print(f"   Queries per document: {queries_per_doc}")

    qd_pairs = []

    for doc in tqdm(documents, desc="Generating QD pairs"):
        # Split into sentences (simple approach)
        sentences = doc.split(". ")

        # Take first few sentences as queries
        for i in range(min(queries_per_doc, len(sentences))):
            query = sentences[i].strip()

            # Filter by length
            if len(query) < 10 or len(query) > max_query_length:
                continue

            # Add question mark for query-like format (optional)
            if not query.endswith("?"):
                # Could append "?" but keeping original for now
                pass

            qd_pairs.append((query, doc, 1.0))

    print(f"✓ Generated {len(qd_pairs):,} query-document pairs")

    return qd_pairs
