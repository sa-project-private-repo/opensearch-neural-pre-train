"""
Cross-lingual synonym discovery for Korean-English bilingual terms.

This module discovers synonyms across languages (Korean ↔ English) using:
1. Co-occurrence patterns in bilingual documents
2. Token embedding similarity in multilingual space
3. Transliteration matching
4. Common usage patterns
"""

from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
import re

import numpy as np
from tqdm import tqdm


def extract_bilingual_terms(
    documents: List[str],
    tokenizer,
) -> Dict[str, Set[str]]:
    """
    Extract bilingual term co-occurrences from documents.

    Finds patterns like "모델(model)", "검색 (search)", etc.

    Args:
        documents: List of documents
        tokenizer: Tokenizer

    Returns:
        Dictionary mapping Korean terms to English terms found together

    Example:
        >>> bilingual = extract_bilingual_terms(documents, tokenizer)
        >>> print(bilingual['모델'])  # {'model', 'Model'}
    """
    print("\n🔍 Extracting Bilingual Term Patterns")

    # Patterns to match bilingual terms
    patterns = [
        r'(\w+)\s*\(([a-zA-Z]+)\)',      # 모델(model)
        r'(\w+)\s*\[([a-zA-Z]+)\]',      # 모델[model]
        r'([a-zA-Z]+)\s*\((\w+)\)',      # model(모델)
        r'(\w+)[,\s]+([a-zA-Z]+)',       # 모델, model
    ]

    bilingual_map = defaultdict(set)

    for doc in tqdm(documents, desc="Extracting bilingual terms"):
        for pattern in patterns:
            matches = re.findall(pattern, doc)
            for term1, term2 in matches:
                # Determine which is Korean and which is English
                korean_term = term1 if not term1.isascii() else term2
                english_term = term2 if term2.isascii() else term1

                if korean_term and english_term:
                    # Normalize
                    korean_term = korean_term.strip()
                    english_term = english_term.strip().lower()

                    if len(korean_term) > 1 and len(english_term) > 1:
                        bilingual_map[korean_term].add(english_term)
                        bilingual_map[english_term].add(korean_term)

    print(f"✓ Extracted {len(bilingual_map):,} bilingual term pairs")

    # Show examples
    print(f"\n  Sample bilingual mappings:")
    for i, (term, translations) in enumerate(list(bilingual_map.items())[:10]):
        trans_str = ", ".join(list(translations)[:3])
        print(f"    {term} ↔ {trans_str}")

    return dict(bilingual_map)


def discover_cross_lingual_synonyms_by_embedding(
    token_embeddings: np.ndarray,
    tokenizer,
    korean_tokens: List[str],
    english_tokens: List[str],
    similarity_threshold: float = 0.70,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Discover cross-lingual synonyms using embedding similarity.

    Args:
        token_embeddings: BERT token embeddings
        tokenizer: Tokenizer
        korean_tokens: List of Korean tokens to match
        english_tokens: List of English tokens to match
        similarity_threshold: Minimum cosine similarity

    Returns:
        Dictionary mapping tokens to cross-lingual synonyms

    Example:
        >>> synonyms = discover_cross_lingual_synonyms_by_embedding(
        ...     embeddings, tokenizer, ['모델', '검색'], ['model', 'search']
        ... )
    """
    print(f"\n🌐 Discovering Cross-Lingual Synonyms via Embeddings")
    print(f"  Korean tokens: {len(korean_tokens)}")
    print(f"  English tokens: {len(english_tokens)}")
    print(f"  Similarity threshold: {similarity_threshold}")

    # Convert tokens to IDs
    korean_ids = []
    for token in korean_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            korean_ids.append((token, token_id))

    english_ids = []
    for token in english_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            english_ids.append((token, token_id))

    print(f"  Valid Korean IDs: {len(korean_ids)}")
    print(f"  Valid English IDs: {len(english_ids)}")

    # Compute cross-lingual similarities
    cross_lingual_synonyms = {}

    for kor_token, kor_id in tqdm(korean_ids, desc="Computing similarities"):
        kor_embedding = token_embeddings[kor_id]

        candidates = []
        for eng_token, eng_id in english_ids:
            eng_embedding = token_embeddings[eng_id]

            # Cosine similarity
            similarity = np.dot(kor_embedding, eng_embedding) / (
                np.linalg.norm(kor_embedding) * np.linalg.norm(eng_embedding) + 1e-10
            )

            if similarity >= similarity_threshold:
                candidates.append((eng_token, float(similarity)))

        if candidates:
            # Sort by similarity
            candidates.sort(key=lambda x: -x[1])
            cross_lingual_synonyms[kor_token] = candidates[:10]

    # Also do reverse (English → Korean)
    for eng_token, eng_id in tqdm(english_ids, desc="Reverse direction"):
        eng_embedding = token_embeddings[eng_id]

        candidates = []
        for kor_token, kor_id in korean_ids:
            kor_embedding = token_embeddings[kor_id]

            similarity = np.dot(kor_embedding, eng_embedding) / (
                np.linalg.norm(kor_embedding) * np.linalg.norm(eng_embedding) + 1e-10
            )

            if similarity >= similarity_threshold:
                candidates.append((kor_token, float(similarity)))

        if candidates:
            candidates.sort(key=lambda x: -x[1])
            cross_lingual_synonyms[eng_token] = candidates[:10]

    print(f"\n✓ Found {len(cross_lingual_synonyms):,} cross-lingual synonym pairs")

    return cross_lingual_synonyms


def build_comprehensive_bilingual_dictionary(
    documents: List[str],
    token_embeddings: np.ndarray,
    tokenizer,
    bert_model,
    manual_pairs: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    """
    Build comprehensive bilingual synonym dictionary using multiple methods.

    Combines:
    1. Pattern-based extraction (모델(model))
    2. Embedding similarity
    3. Manual curated pairs

    Args:
        documents: Document corpus
        token_embeddings: BERT embeddings
        tokenizer: Tokenizer
        bert_model: BERT model
        manual_pairs: Optional manual Korean-English pairs

    Returns:
        Comprehensive bilingual dictionary

    Example:
        >>> bilingual_dict = build_comprehensive_bilingual_dictionary(
        ...     documents, embeddings, tokenizer, model
        ... )
        >>> print(bilingual_dict['모델'])  # ['model', 'Model', ...]
    """
    print("\n" + "="*70)
    print("Building Comprehensive Bilingual Synonym Dictionary")
    print("="*70)

    final_dict = defaultdict(set)

    # Method 1: Pattern-based extraction
    print("\n[Method 1] Pattern-based extraction")
    pattern_dict = extract_bilingual_terms(documents, tokenizer)

    for term, translations in pattern_dict.items():
        final_dict[term].update(translations)

    # Method 2: Identify Korean and English terms from corpus
    print("\n[Method 2] Corpus analysis")
    korean_terms = set()
    english_terms = set()

    for doc in tqdm(documents[:5000], desc="Analyzing corpus"):  # Sample for speed
        tokens = tokenizer.tokenize(doc)
        for token in tokens:
            if not token.startswith("##") and len(token) > 1:
                if token.isascii() and token.isalpha():
                    english_terms.add(token.lower())
                elif not token.isascii():
                    korean_terms.add(token)

    print(f"  Korean terms: {len(korean_terms):,}")
    print(f"  English terms: {len(english_terms):,}")

    # Method 3: Embedding-based cross-lingual matching
    print("\n[Method 3] Embedding-based matching")

    # Sample for computational efficiency
    korean_sample = list(korean_terms)[:500]
    english_sample = list(english_terms)[:500]

    cross_lingual = discover_cross_lingual_synonyms_by_embedding(
        token_embeddings=token_embeddings,
        tokenizer=tokenizer,
        korean_tokens=korean_sample,
        english_tokens=english_sample,
        similarity_threshold=0.65,  # Lower threshold for cross-lingual
    )

    for term, synonyms in cross_lingual.items():
        for syn, _ in synonyms:
            final_dict[term].add(syn)
            final_dict[syn].add(term)

    # Method 4: Add manual curated pairs
    if manual_pairs:
        print("\n[Method 4] Adding manual pairs")
        for term, synonyms in manual_pairs.items():
            final_dict[term].update(synonyms)
            for syn in synonyms:
                final_dict[syn].add(term)
        print(f"  Added {len(manual_pairs):,} manual pairs")

    # Convert sets to lists
    final_dict_cleaned = {
        term: list(syns) for term, syns in final_dict.items()
        if len(syns) > 0
    }

    print("\n" + "="*70)
    print(f"✓ Bilingual Dictionary Complete")
    print(f"  Total entries: {len(final_dict_cleaned):,}")
    print("="*70)

    # Show examples
    print(f"\n  Sample bilingual synonyms:")
    for i, (term, synonyms) in enumerate(list(final_dict_cleaned.items())[:10]):
        syn_str = ", ".join(synonyms[:5])
        print(f"    {term} ↔ {syn_str}")

    return final_dict_cleaned


def get_default_korean_english_pairs() -> Dict[str, List[str]]:
    """
    Get curated Korean-English term pairs for common ML/AI terms.

    Returns:
        Dictionary of Korean terms to English synonyms
    """
    return {
        # Core ML/AI terms
        "모델": ["model", "Model"],
        "학습": ["learning", "training", "train"],
        "검색": ["search", "retrieval", "retrieve"],
        "데이터": ["data", "Data"],
        "알고리즘": ["algorithm", "Algorithm"],
        "신경망": ["network", "neural network", "net"],
        "벡터": ["vector", "Vector"],
        "임베딩": ["embedding", "Embedding"],

        # NLP terms
        "토큰": ["token", "Token"],
        "문서": ["document", "doc", "Document"],
        "쿼리": ["query", "Query"],
        "언어": ["language", "Language"],
        "처리": ["processing", "process"],
        "분석": ["analysis", "analyze"],

        # Architecture terms
        "트랜스포머": ["transformer", "Transformer"],
        "어텐션": ["attention", "Attention"],
        "인코더": ["encoder", "Encoder"],
        "디코더": ["decoder", "Decoder"],

        # Training terms
        "손실": ["loss", "Loss"],
        "최적화": ["optimization", "optimize"],
        "배치": ["batch", "Batch"],
        "에포크": ["epoch", "Epoch"],

        # Specific models
        "버트": ["bert", "BERT"],
        "지피티": ["gpt", "GPT"],

        # Search terms
        "희소": ["sparse", "Sparse"],
        "밀집": ["dense", "Dense"],
        "랭킹": ["ranking", "rank"],
        "스코어": ["score", "Score"],

        # General
        "시스템": ["system", "System"],
        "기술": ["technology", "tech"],
        "방법": ["method", "approach"],
        "성능": ["performance", "perf"],
    }


def apply_bilingual_synonyms_to_idf(
    idf_dict: Dict[str, float],
    bilingual_dict: Dict[str, List[str]],
    tokenizer,
) -> Dict[str, float]:
    """
    Share IDF values across bilingual synonym pairs.

    If "모델" and "model" are synonyms, they should have similar IDF values.

    Args:
        idf_dict: Original IDF dictionary
        bilingual_dict: Bilingual synonym dictionary
        tokenizer: Tokenizer

    Returns:
        IDF dictionary with shared values across bilingual synonyms

    Example:
        >>> enhanced_idf = apply_bilingual_synonyms_to_idf(
        ...     idf_dict, bilingual_dict, tokenizer
        ... )
    """
    print("\n🔗 Applying Bilingual Synonyms to IDF")

    enhanced_idf = idf_dict.copy()
    updates = 0

    for term, synonyms in tqdm(bilingual_dict.items(), desc="Sharing IDF"):
        # Get IDF for main term
        if term not in enhanced_idf:
            continue

        main_idf = enhanced_idf[term]

        # Collect IDF values from all synonyms
        synonym_idfs = [main_idf]
        for syn in synonyms:
            if syn in enhanced_idf:
                synonym_idfs.append(enhanced_idf[syn])

        # Use maximum IDF (most discriminative)
        max_idf = max(synonym_idfs)

        # Apply to all synonyms
        enhanced_idf[term] = max_idf
        for syn in synonyms:
            if syn in enhanced_idf or syn.lower() in enhanced_idf:
                enhanced_idf[syn] = max_idf
                enhanced_idf[syn.lower()] = max_idf
                updates += 1

    print(f"✓ Updated {updates:,} IDF entries with bilingual synonyms")

    return enhanced_idf
