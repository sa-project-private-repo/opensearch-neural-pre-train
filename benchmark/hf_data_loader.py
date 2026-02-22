"""
HuggingFace dataset loader for MTEB-style retrieval benchmarks.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class MTEBBenchmarkData:
    """Prepared MTEB-style benchmark data."""

    queries: List[str]
    query_ids: List[str]
    query_relevant_docs: Dict[str, List[str]]  # query_id -> [doc_ids]
    documents: Dict[str, str]  # doc_id -> content
    doc_titles: Dict[str, str]  # doc_id -> title
    dataset_name: str


AVAILABLE_DATASETS = {
    "ko-strategyqa": {
        "name": "mteb/Ko-StrategyQA",
        "description": "Korean multi-hop QA retrieval (592 queries, 9K docs)",
    },
    "miracl-ko": {
        "name": "miracl/miracl",
        "config": "ko",
        "description": "MIRACL Korean Wikipedia retrieval",
    },
    "mrtydi-ko": {
        "name": "castorini/mr-tydi",
        "config": "korean",
        "description": "Mr. TyDi Korean retrieval benchmark (test split)",
    },
}


def load_ko_strategyqa(
    max_queries: Optional[int] = None,
) -> MTEBBenchmarkData:
    """
    Load Ko-StrategyQA dataset from HuggingFace.

    Args:
        max_queries: Maximum number of queries to load

    Returns:
        MTEBBenchmarkData with queries, documents, and relevance judgments
    """
    logger.info("Loading Ko-StrategyQA dataset from HuggingFace...")

    # Load all splits
    corpus = load_dataset("mteb/Ko-StrategyQA", "corpus")["dev"]
    queries_ds = load_dataset("mteb/Ko-StrategyQA", "queries")["dev"]
    qrels = load_dataset("mteb/Ko-StrategyQA", "qrels")["dev"]

    # Build corpus
    documents: Dict[str, str] = {}
    doc_titles: Dict[str, str] = {}
    for doc in corpus:
        doc_id = doc["_id"]
        documents[doc_id] = doc["text"]
        doc_titles[doc_id] = doc.get("title", "")

    logger.info(f"Loaded {len(documents)} documents")

    # Build queries
    query_id_to_text: Dict[str, str] = {}
    for q in queries_ds:
        query_id_to_text[q["_id"]] = q["text"]

    # Build relevance judgments
    query_relevant_docs: Dict[str, List[str]] = {}
    for rel in qrels:
        qid = rel["query-id"]
        doc_id = rel["corpus-id"]
        score = rel["score"]
        if score > 0:  # Only positive relevance
            if qid not in query_relevant_docs:
                query_relevant_docs[qid] = []
            query_relevant_docs[qid].append(doc_id)

    # Filter to queries with relevance judgments
    valid_query_ids = list(query_relevant_docs.keys())
    if max_queries:
        valid_query_ids = valid_query_ids[:max_queries]

    queries = [query_id_to_text[qid] for qid in valid_query_ids]
    query_ids = valid_query_ids

    logger.info(
        f"Loaded {len(queries)} queries with relevance judgments, "
        f"{len(documents)} documents"
    )

    return MTEBBenchmarkData(
        queries=queries,
        query_ids=query_ids,
        query_relevant_docs=query_relevant_docs,
        documents=documents,
        doc_titles=doc_titles,
        dataset_name="Ko-StrategyQA",
    )


def load_miracl_ko(
    max_queries: Optional[int] = None,
    max_corpus: Optional[int] = 10000,
) -> MTEBBenchmarkData:
    """
    Load MIRACL Korean dataset from HuggingFace.

    Loads dev queries/qrels first, then builds a corpus containing
    all relevant/negative passages plus random distractors from the
    full corpus to reach max_corpus size.

    Args:
        max_queries: Maximum number of queries to load
        max_corpus: Maximum corpus size (default 10K).
            None for full corpus (~1.5M docs, very slow).

    Returns:
        MTEBBenchmarkData with queries, documents, and relevance judgments
    """
    import random

    # 1. Load queries and qrels first (small, fast)
    logger.info("Loading MIRACL Korean dev queries and qrels...")
    dev_ds = load_dataset(
        "miracl/miracl", "ko", split="dev", trust_remote_code=True
    )

    query_relevant_docs: Dict[str, List[str]] = {}
    query_id_to_text: Dict[str, str] = {}
    passage_docs: Dict[str, str] = {}  # from positive/negative passages
    passage_titles: Dict[str, str] = {}

    for row in dev_ds:
        qid = str(row["query_id"])
        query_id_to_text[qid] = row["query"]
        positive_passages = row.get("positive_passages", []) or []
        negative_passages = row.get("negative_passages", []) or []
        if positive_passages:
            query_relevant_docs[qid] = [p["docid"] for p in positive_passages]
        for p in positive_passages + negative_passages:
            doc_id = p["docid"]
            title = p.get("title", "") or ""
            text = p.get("text", "") or ""
            passage_docs[doc_id] = (title + "\n" + text).strip() if title else text
            passage_titles[doc_id] = title

    valid_query_ids = list(query_relevant_docs.keys())
    if max_queries:
        valid_query_ids = valid_query_ids[:max_queries]

    queries = [query_id_to_text[qid] for qid in valid_query_ids]
    query_ids = valid_query_ids

    logger.info(
        f"Loaded {len(queries)} queries, "
        f"{len(passage_docs)} passage documents from qrels"
    )

    # 2. Build corpus: passage docs + random distractors from full corpus
    documents: Dict[str, str] = dict(passage_docs)
    doc_titles: Dict[str, str] = dict(passage_titles)

    need_distractors = (
        max_corpus is not None and max_corpus > len(documents)
    )
    if need_distractors:
        num_distractors = max_corpus - len(documents)
        logger.info(
            f"Loading MIRACL corpus for {num_distractors} random distractors..."
        )
        corpus_ds = load_dataset(
            "miracl/miracl-corpus", "ko", split="train",
            trust_remote_code=True,
        )
        # Sample random indices
        existing_ids = set(documents.keys())
        all_indices = list(range(len(corpus_ds)))
        random.seed(42)
        random.shuffle(all_indices)

        added = 0
        for idx in all_indices:
            if added >= num_distractors:
                break
            doc = corpus_ds[idx]
            doc_id = doc["docid"]
            if doc_id in existing_ids:
                continue
            title = doc.get("title", "") or ""
            text = doc.get("text", "") or ""
            documents[doc_id] = (
                (title + "\n" + text).strip() if title else text
            )
            doc_titles[doc_id] = title
            added += 1

        logger.info(f"Added {added} distractor documents")
    elif max_corpus is None:
        # Load full corpus
        logger.info(
            "Loading full MIRACL Korean corpus (this may take a while)..."
        )
        corpus_ds = load_dataset(
            "miracl/miracl-corpus", "ko", split="train",
            trust_remote_code=True,
        )
        for doc in corpus_ds:
            doc_id = doc["docid"]
            if doc_id not in documents:
                title = doc.get("title", "") or ""
                text = doc.get("text", "") or ""
                documents[doc_id] = (
                    (title + "\n" + text).strip() if title else text
                )
                doc_titles[doc_id] = title

    logger.info(
        f"Final corpus: {len(documents)} documents, "
        f"{len(queries)} queries"
    )

    return MTEBBenchmarkData(
        queries=queries,
        query_ids=query_ids,
        query_relevant_docs=query_relevant_docs,
        documents=documents,
        doc_titles=doc_titles,
        dataset_name="MIRACL-ko",
    )


def load_mrtydi_ko(
    max_queries: Optional[int] = None,
    max_corpus: Optional[int] = 10000,
) -> MTEBBenchmarkData:
    """
    Load Mr. TyDi Korean dataset from HuggingFace.

    Loads test queries/qrels first, then builds a corpus containing
    all relevant/negative passages plus random distractors from the
    full corpus to reach max_corpus size.

    Args:
        max_queries: Maximum number of queries to load
        max_corpus: Maximum corpus size (default 10K).
            None for full corpus.

    Returns:
        MTEBBenchmarkData with queries, documents, and relevance judgments
    """
    import random

    # 1. Load queries and qrels first (small, fast)
    logger.info("Loading Mr. TyDi Korean test queries and qrels...")
    test_ds = load_dataset(
        "castorini/mr-tydi", "korean", split="test", trust_remote_code=True
    )

    query_relevant_docs: Dict[str, List[str]] = {}
    query_id_to_text: Dict[str, str] = {}
    passage_docs: Dict[str, str] = {}  # from positive/negative passages
    passage_titles: Dict[str, str] = {}

    for row in test_ds:
        qid = str(row["query_id"])
        query_id_to_text[qid] = row["query"]
        positive_passages = row.get("positive_passages", []) or []
        negative_passages = row.get("negative_passages", []) or []
        if positive_passages:
            query_relevant_docs[qid] = [p["docid"] for p in positive_passages]
        for p in positive_passages + negative_passages:
            doc_id = p["docid"]
            title = p.get("title", "") or ""
            text = p.get("text", "") or ""
            passage_docs[doc_id] = (title + "\n" + text).strip() if title else text
            passage_titles[doc_id] = title

    valid_query_ids = list(query_relevant_docs.keys())
    if max_queries:
        valid_query_ids = valid_query_ids[:max_queries]

    queries = [query_id_to_text[qid] for qid in valid_query_ids]
    query_ids = valid_query_ids

    logger.info(
        f"Loaded {len(queries)} queries, "
        f"{len(passage_docs)} passage documents from qrels"
    )

    # 2. Load corpus (always needed: passages may have empty text)
    logger.info("Loading Mr. TyDi Korean corpus...")
    corpus_ds = load_dataset(
        "castorini/mr-tydi-corpus", "korean", split="train",
        trust_remote_code=True,
    )

    # Build corpus lookup for resolving empty passages
    corpus_lookup: Dict[str, Dict[str, str]] = {}
    for doc in corpus_ds:
        doc_id = doc["docid"]
        corpus_lookup[doc_id] = {
            "title": doc.get("title", "") or "",
            "text": doc.get("text", "") or "",
        }

    # Resolve empty passage text from corpus
    resolved = 0
    for doc_id in list(passage_docs.keys()):
        if not passage_docs[doc_id] and doc_id in corpus_lookup:
            entry = corpus_lookup[doc_id]
            title = entry["title"]
            text = entry["text"]
            passage_docs[doc_id] = (
                (title + "\n" + text).strip() if title else text
            )
            passage_titles[doc_id] = title
            resolved += 1
    if resolved:
        logger.info(f"Resolved {resolved} empty passages from corpus")

    documents: Dict[str, str] = dict(passage_docs)
    doc_titles: Dict[str, str] = dict(passage_titles)

    # Add distractors from corpus
    need_distractors = (
        max_corpus is not None and max_corpus > len(documents)
    )
    if need_distractors:
        num_distractors = max_corpus - len(documents)
        logger.info(
            f"Adding {num_distractors} random distractors..."
        )
        existing_ids = set(documents.keys())
        all_corpus_ids = list(corpus_lookup.keys())
        random.seed(42)
        random.shuffle(all_corpus_ids)

        added = 0
        for doc_id in all_corpus_ids:
            if added >= num_distractors:
                break
            if doc_id in existing_ids:
                continue
            entry = corpus_lookup[doc_id]
            title = entry["title"]
            text = entry["text"]
            documents[doc_id] = (
                (title + "\n" + text).strip() if title else text
            )
            doc_titles[doc_id] = title
            added += 1

        logger.info(f"Added {added} distractor documents")
    elif max_corpus is None:
        for doc_id, entry in corpus_lookup.items():
            if doc_id not in documents:
                title = entry["title"]
                text = entry["text"]
                documents[doc_id] = (
                    (title + "\n" + text).strip()
                    if title
                    else text
                )
                doc_titles[doc_id] = title

    logger.info(
        f"Final corpus: {len(documents)} documents, "
        f"{len(queries)} queries"
    )

    return MTEBBenchmarkData(
        queries=queries,
        query_ids=query_ids,
        query_relevant_docs=query_relevant_docs,
        documents=documents,
        doc_titles=doc_titles,
        dataset_name="mrtydi-ko",
    )


def load_hf_dataset(
    dataset_name: str,
    max_queries: Optional[int] = None,
) -> MTEBBenchmarkData:
    """
    Load a HuggingFace retrieval dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "ko-strategyqa", "miracl-ko")
        max_queries: Maximum number of queries

    Returns:
        MTEBBenchmarkData
    """
    if dataset_name == "ko-strategyqa":
        return load_ko_strategyqa(max_queries)
    elif dataset_name == "miracl-ko":
        return load_miracl_ko(max_queries)
    elif dataset_name == "mrtydi-ko":
        return load_mrtydi_ko(max_queries)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(AVAILABLE_DATASETS.keys())}"
        )


def list_available_datasets() -> Dict[str, str]:
    """List available HuggingFace datasets."""
    return {k: v["description"] for k, v in AVAILABLE_DATASETS.items()}
