"""
Synthetic Data Generator using LLM

This module generates synthetic query-document pairs using LLM models
for training neural sparse retrieval models.

Features:
- Document → Query generation (reverse direction)
- Query augmentation (paraphrasing)
- Quality filtering
- Batch processing for efficiency

Requirements:
- src.llm_loader module
- transformers, torch
"""

from typing import List, Tuple, Optional, Dict, Any
import re
from tqdm import tqdm


# Prompt templates
DOC_TO_QUERY_PROMPT = """다음 문서를 읽고 사용자가 이 문서를 찾기 위해 검색할 만한 쿼리를 {num_queries}개 생성하세요.
각 쿼리는 짧고 구체적이어야 합니다 (5-15단어).
**중요: 쿼리는 반드시 한국어로 작성해야 합니다.**

문서: {document}

검색 쿼리 ({num_queries}개, 각 줄에 하나씩, 한국어로):"""

QUERY_AUGMENT_PROMPT = """다음 검색 쿼리와 같은 의미를 가지지만 표현이 다른 쿼리를 {num_variants}개 생성하세요.
**중요: 변형 쿼리는 반드시 한국어로 작성해야 합니다.**

원본 쿼리: {query}

변형 쿼리 ({num_variants}개, 각 줄에 하나씩, 한국어로):"""


def generate_queries_from_document(
    document: str,
    llm_model: Any,
    llm_tokenizer: Any,
    num_queries: int = 3,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    verbose: bool = False,
) -> List[str]:
    """
    Generate queries from a document using LLM.

    Args:
        document: Source document
        llm_model: Loaded LLM model
        llm_tokenizer: Loaded tokenizer
        num_queries: Number of queries to generate
        max_new_tokens: Max tokens in generation
        temperature: Sampling temperature
        verbose: Print detailed logs

    Returns:
        List of generated queries

    Example:
        >>> doc = "OpenSearch는 강력한 검색 엔진입니다."
        >>> queries = generate_queries_from_document(doc, model, tokenizer)
        >>> print(queries)  # ["OpenSearch 기능", "검색 엔진 비교", ...]
    """
    import time
    from src.llm_loader import generate_text

    # Truncate long documents
    doc_truncated = document[:500]  # First 500 chars

    if verbose:
        print(f"      🔹 Document length: {len(document)} chars (truncated to {len(doc_truncated)})")

    prompt = DOC_TO_QUERY_PROMPT.format(
        document=doc_truncated,
        num_queries=num_queries,
    )

    if verbose:
        print(f"      🔹 Prompt length: {len(prompt)} chars")
        print(f"      🔹 Sending to LLM (max_tokens={max_new_tokens}, temp={temperature})...")

    # Generate
    start_time = time.time()
    generated = generate_text(
        model=llm_model,
        tokenizer=llm_tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
    )
    gen_time = time.time() - start_time

    if verbose:
        print(f"      🔹 LLM generation completed in {gen_time:.2f}s")
        print(f"      🔹 Raw output length: {len(generated)} chars")

    # Parse queries (each line)
    queries = []
    for line in generated.split('\n'):
        line = line.strip()
        # Remove numbering (1., 2., -, etc.)
        line = re.sub(r'^[\d\-\*\•]+[\.\)]\s*', '', line)
        line = line.strip()

        if line and len(line) > 5:  # Min length
            queries.append(line)

    if verbose:
        print(f"      🔹 Parsed {len(queries)} queries from output")

    return queries[:num_queries]  # Limit to requested number


def augment_query(
    query: str,
    llm_model: Any,
    llm_tokenizer: Any,
    num_variants: int = 2,
    max_new_tokens: int = 100,
    temperature: float = 0.9,
) -> List[str]:
    """
    Generate query variations (paraphrasing).

    Args:
        query: Original query
        llm_model: Loaded LLM model
        llm_tokenizer: Loaded tokenizer
        num_variants: Number of variants to generate
        max_new_tokens: Max tokens in generation
        temperature: Sampling temperature

    Returns:
        List of query variants

    Example:
        >>> query = "한국어 검색 최적화"
        >>> variants = augment_query(query, model, tokenizer)
        >>> print(variants)  # ["한글 검색 개선", "코리안 검색 향상", ...]
    """
    from src.llm_loader import generate_text

    prompt = QUERY_AUGMENT_PROMPT.format(
        query=query,
        num_variants=num_variants,
    )

    generated = generate_text(
        model=llm_model,
        tokenizer=llm_tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
    )

    # Parse variants
    variants = []
    for line in generated.split('\n'):
        line = line.strip()
        line = re.sub(r'^[\d\-\*\•]+[\.\)]\s*', '', line)
        line = line.strip()

        if line and len(line) > 5:
            variants.append(line)

    return variants[:num_variants]


def filter_quality(
    query: str,
    document: str,
    min_query_length: int = 5,
    max_query_length: int = 100,
    min_doc_length: int = 20,
) -> bool:
    """
    Filter low-quality query-document pairs.

    Args:
        query: Query string
        document: Document string
        min_query_length: Minimum query length (chars)
        max_query_length: Maximum query length (chars)
        min_doc_length: Minimum document length (chars)

    Returns:
        True if passes quality check, False otherwise

    Example:
        >>> filter_quality("검색", "문서", min_query_length=5)
        False  # Query too short
    """
    # Length checks
    if len(query) < min_query_length:
        return False

    if len(query) > max_query_length:
        return False

    if len(document) < min_doc_length:
        return False

    # Check if query is not just document prefix
    if document.startswith(query):
        return False

    # Check for too much overlap (potential copying)
    query_words = set(query.split())
    doc_words = set(document.split())
    if len(query_words) > 0:
        overlap = len(query_words & doc_words) / len(query_words)
        if overlap < 0.3:  # Too little overlap (not relevant)
            return False
        if overlap > 0.95:  # Too much overlap (copying)
            return False

    return True


def generate_synthetic_qd_pairs(
    documents: List[str],
    llm_model: Any,
    llm_tokenizer: Any,
    num_queries_per_doc: int = 3,
    batch_size: int = 2,
    max_documents: Optional[int] = None,
    enable_filtering: bool = True,
    verbose: bool = True,
) -> List[Tuple[str, str, float]]:
    """
    Generate synthetic query-document pairs from documents.

    Args:
        documents: List of documents
        llm_model: Loaded LLM model
        llm_tokenizer: Loaded tokenizer
        num_queries_per_doc: Queries to generate per document
        batch_size: Batch size for processing (not used in current impl)
        max_documents: Maximum documents to process (None = all)
        enable_filtering: Whether to apply quality filtering
        verbose: Print detailed progress logs

    Returns:
        List of (query, document, relevance) tuples

    Example:
        >>> docs = ["OpenSearch는 검색 엔진입니다.", "Elasticsearch와 호환됩니다."]
        >>> pairs = generate_synthetic_qd_pairs(docs, model, tokenizer)
        >>> print(len(pairs))  # 6 (3 queries × 2 docs)
    """
    import time
    from datetime import datetime

    if max_documents is not None:
        documents = documents[:max_documents]

    print("\n" + "="*70)
    print("📝 Generating Synthetic Query-Document Pairs")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Documents: {len(documents)}")
    print(f"Queries per doc: {num_queries_per_doc}")
    print(f"Expected total queries: {len(documents) * num_queries_per_doc}")
    print(f"Quality filtering: {'ON' if enable_filtering else 'OFF'}")
    print(f"Verbose logging: {'ON' if verbose else 'OFF'}")
    print("="*70)

    synthetic_pairs = []
    failed_count = 0
    filtered_count = 0
    total_generation_time = 0

    # Progress tracking
    start_time = time.time()
    last_report_time = start_time

    for i, doc in enumerate(tqdm(documents, desc="Generating queries", ncols=100)):
        doc_start_time = time.time()

        if verbose and i % 10 == 0:
            # Report every 10 documents
            elapsed = time.time() - start_time
            avg_time_per_doc = elapsed / (i + 1) if i > 0 else 0
            eta = avg_time_per_doc * (len(documents) - i - 1)

            print(f"\n{'='*70}")
            print(f"📊 Progress Report - Document {i+1}/{len(documents)}")
            print(f"{'='*70}")
            print(f"⏱️  Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
            print(f"✅ Generated: {len(synthetic_pairs)} pairs")
            print(f"❌ Failed/Filtered: {failed_count + filtered_count}")
            print(f"⚡ Avg time/doc: {avg_time_per_doc:.2f}s")
            print(f"{'='*70}")

        try:
            if verbose:
                print(f"\n📄 Doc {i+1}: {doc[:80]}...")
                print(f"   🤖 Calling LLM...")

            # Generate queries
            query_gen_start = time.time()
            queries = generate_queries_from_document(
                document=doc,
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer,
                num_queries=num_queries_per_doc,
            )
            query_gen_time = time.time() - query_gen_start
            total_generation_time += query_gen_time

            if verbose:
                print(f"   ✅ LLM responded in {query_gen_time:.2f}s")
                print(f"   📝 Generated {len(queries)} queries:")
                for j, q in enumerate(queries, 1):
                    print(f"      {j}. {q}")

            # Filter and add queries
            added_for_this_doc = 0
            for query in queries:
                # Quality filtering
                if enable_filtering:
                    if not filter_quality(query, doc):
                        filtered_count += 1
                        if verbose:
                            print(f"      ⚠️  Filtered: {query[:50]}...")
                        continue

                # Add positive pair
                synthetic_pairs.append((query, doc, 1.0))
                added_for_this_doc += 1

            if verbose:
                print(f"   ➕ Added {added_for_this_doc} pairs")

        except Exception as e:
            failed_count += 1
            print(f"\n❌ Error on doc {i+1}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue

        doc_time = time.time() - doc_start_time
        if verbose:
            print(f"   ⏱️  Total time for this doc: {doc_time:.2f}s")

    # Final report
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("✅ Generation Complete")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Avg time per doc: {total_time/len(documents):.2f}s")
    print(f"Avg LLM generation time: {total_generation_time/len(documents):.2f}s")
    print()
    print(f"📊 Results:")
    print(f"  Total pairs generated: {len(synthetic_pairs):,}")
    print(f"  Failed: {failed_count:,}")
    print(f"  Filtered: {filtered_count:,}")
    print(f"  Success rate: {len(synthetic_pairs)/(len(documents)*num_queries_per_doc)*100:.1f}%")
    print(f"  Average queries per doc: {len(synthetic_pairs) / len(documents):.2f}")
    print("="*70)

    return synthetic_pairs


def generate_hard_negatives(
    query: str,
    positive_doc: str,
    candidate_docs: List[str],
    llm_model: Any,
    llm_tokenizer: Any,
    num_negatives: int = 2,
) -> List[str]:
    """
    Generate hard negative documents for a query.

    Hard negatives are documents that are semantically similar but not relevant.

    Args:
        query: Query string
        positive_doc: Positive (relevant) document
        candidate_docs: Pool of candidate documents
        llm_model: Loaded LLM model
        llm_tokenizer: Loaded tokenizer
        num_negatives: Number of hard negatives to generate

    Returns:
        List of hard negative documents

    Example:
        >>> hard_negs = generate_hard_negatives(
        ...     "OpenSearch 기능",
        ...     "OpenSearch는 검색 엔진입니다.",
        ...     candidate_docs,
        ...     model,
        ...     tokenizer
        ... )
    """
    # Placeholder: Simple random selection from candidates
    # TODO: Implement LLM-based hard negative generation
    import random
    return random.sample(candidate_docs, min(num_negatives, len(candidate_docs)))


def deduplicate_pairs(
    pairs: List[Tuple[str, str, float]]
) -> List[Tuple[str, str, float]]:
    """
    Remove duplicate query-document pairs.

    Args:
        pairs: List of (query, document, relevance) tuples

    Returns:
        Deduplicated list

    Example:
        >>> pairs = [("q1", "d1", 1.0), ("q1", "d1", 1.0), ("q2", "d2", 1.0)]
        >>> dedup = deduplicate_pairs(pairs)
        >>> len(dedup)  # 2
    """
    seen = set()
    unique_pairs = []

    for query, doc, relevance in pairs:
        key = (query, doc)
        if key not in seen:
            seen.add(key)
            unique_pairs.append((query, doc, relevance))

    return unique_pairs


if __name__ == "__main__":
    print("="*70)
    print("Synthetic Data Generator Module")
    print("="*70)
    print("\nUsage:")
    print("  from src.synthetic_data_generator import generate_synthetic_qd_pairs")
    print("  from src.llm_loader import load_qwen3_awq")
    print()
    print("  model, tokenizer = load_qwen3_awq()")
    print("  pairs = generate_synthetic_qd_pairs(documents, model, tokenizer)")
    print("="*70)
