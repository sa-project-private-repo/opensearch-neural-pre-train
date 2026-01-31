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


def load_hf_dataset(
    dataset_name: str,
    max_queries: Optional[int] = None,
) -> MTEBBenchmarkData:
    """
    Load a HuggingFace retrieval dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "ko-strategyqa")
        max_queries: Maximum number of queries

    Returns:
        MTEBBenchmarkData
    """
    if dataset_name == "ko-strategyqa":
        return load_ko_strategyqa(max_queries)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(AVAILABLE_DATASETS.keys())}"
        )


def list_available_datasets() -> Dict[str, str]:
    """List available HuggingFace datasets."""
    return {k: v["description"] for k, v in AVAILABLE_DATASETS.items()}
