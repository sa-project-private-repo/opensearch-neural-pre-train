"""
7-way OpenSearch retrieval benchmark.

Methods: BM25, Dense, Sparse, BM25+Dense, BM25+Sparse, Dense+Sparse, BM25+Dense+Sparse
Dense: BAAI/bge-m3 (1024-dim)
Sparse: sewoong/korean-neural-sparse-encoder-base-klue-large (klue/roberta-large SPLADE)
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from benchmark.config import BenchmarkConfig
from benchmark.encoders import BgeM3Encoder, NeuralSparseEncoder
from benchmark.hf_data_loader import load_hf_dataset
from benchmark.index_manager import IndexManager
from benchmark.indexer import EncodedDocument, index_documents
from benchmark.metrics import BenchmarkMetrics, QueryResult, compute_metrics
from benchmark.searchers import (
    BM25Searcher,
    SemanticSearcher,
    NeuralSparseSearcher,
)
from benchmark.hybrid_searcher import (
    HybridBM25SemanticSearcher,
    HybridBM25SparseSearcher,
    HybridSparseSemanticSearcher,
    HybridTripleSearcher,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SPARSE_MODEL = "sewoong/korean-neural-sparse-encoder-base-klue-large"
DENSE_MODEL = "BAAI/bge-m3"

SEVEN_METHODS = [
    "bm25",
    "dense",
    "sparse",
    "bm25_dense_rrf",
    "bm25_sparse_rrf",
    "dense_sparse_rrf",
    "triple_rrf",
]


def create_7way_searchers(
    client,
    config: BenchmarkConfig,
    dense_encoder: BgeM3Encoder,
    sparse_encoder: NeuralSparseEncoder,
) -> dict:
    """Create all 7 searchers."""
    return {
        "bm25": BM25Searcher(
            client=client,
            index_name=config.bm25_index,
            top_k=config.top_k,
        ),
        "dense": SemanticSearcher(
            client=client,
            index_name=config.dense_index,
            encoder=dense_encoder,
            top_k=config.top_k,
        ),
        "sparse": NeuralSparseSearcher(
            client=client,
            index_name=config.sparse_index,
            encoder=sparse_encoder,
            top_k=config.top_k,
        ),
        "bm25_dense_rrf": HybridBM25SemanticSearcher(
            client=client,
            bm25_index=config.bm25_index,
            dense_index=config.dense_index,
            dense_encoder=dense_encoder,
            fusion_method="rrf",
            top_k=config.top_k,
            retrieval_k=100,
            k=60,
        ),
        "bm25_sparse_rrf": HybridBM25SparseSearcher(
            client=client,
            bm25_index=config.bm25_index,
            sparse_index=config.sparse_index,
            sparse_encoder=sparse_encoder,
            fusion_method="rrf",
            top_k=config.top_k,
            retrieval_k=100,
            k=60,
        ),
        "dense_sparse_rrf": HybridSparseSemanticSearcher(
            client=client,
            sparse_index=config.sparse_index,
            dense_index=config.dense_index,
            sparse_encoder=sparse_encoder,
            dense_encoder=dense_encoder,
            fusion_method="rrf",
            top_k=config.top_k,
            retrieval_k=100,
            k=60,
        ),
        "triple_rrf": HybridTripleSearcher(
            client=client,
            bm25_index=config.bm25_index,
            dense_index=config.dense_index,
            sparse_index=config.sparse_index,
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            top_k=config.top_k,
            retrieval_k=100,
            k=60,
        ),
    }


def run_single_method(
    method_name: str,
    searcher,
    queries: List[str],
    query_ids: List[str],
    query_relevant_docs: dict,
) -> tuple[List[QueryResult], BenchmarkMetrics]:
    """Run benchmark for a single search method."""
    results = []

    for i, query in enumerate(tqdm(queries, desc=method_name)):
        qid = query_ids[i]
        relevant = query_relevant_docs.get(qid, [])

        try:
            response = searcher.search(query)
            retrieved = [r.doc_id for r in response.results]

            hit_rank = None
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in set(relevant):
                    hit_rank = rank
                    break

            results.append(QueryResult(
                query=query,
                target_doc_id=relevant[0] if relevant else "",
                retrieved_doc_ids=retrieved,
                latency_ms=response.latency_ms,
                hit_rank=hit_rank,
            ))
        except Exception as e:
            logger.warning(f"Search failed: {e}")
            results.append(QueryResult(
                query=query,
                target_doc_id=relevant[0] if relevant else "",
                retrieved_doc_ids=[],
                latency_ms=0,
                hit_rank=None,
            ))

    metrics = compute_metrics(method_name, results)
    return results, metrics


def main():
    parser = argparse.ArgumentParser(description="7-way OpenSearch benchmark")
    parser.add_argument(
        "--dataset", type=str, default="ko-strategyqa",
        choices=["ko-strategyqa", "miracl-ko", "mrtydi-ko"],
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument(
        "--index-suffix", type=str, default="klue7",
        help="Suffix for index names",
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", default=None,
        help="Specific methods to run (default: all 7)",
    )
    args = parser.parse_args()

    # Config
    config = BenchmarkConfig()
    config.bm25_index = f"benchmark-bm25-{args.index_suffix}"
    config.dense_index = f"benchmark-dense-{args.index_suffix}"
    config.sparse_index = f"benchmark-sparse-{args.index_suffix}"
    config.hybrid_index = f"benchmark-hybrid-{args.index_suffix}"

    # Index manager
    index_manager = IndexManager(config)

    # Load encoders
    logger.info(f"Loading dense encoder: {DENSE_MODEL}")
    dense_encoder = BgeM3Encoder(model_name=DENSE_MODEL, device=config.device)

    logger.info(f"Loading sparse encoder: {SPARSE_MODEL}")
    sparse_encoder = NeuralSparseEncoder(
        model_path=SPARSE_MODEL,
        device=config.device,
        query_max_length=64,
        doc_max_length=256,
    )

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    data = load_hf_dataset(args.dataset, args.max_queries)
    logger.info(f"Queries: {len(data.queries)}, Docs: {len(data.documents)}")

    # Setup indices
    if not args.skip_setup:
        logger.info("=== INDEX SETUP ===")

        # Delete existing indices
        for idx in config.index_names:
            index_manager.delete_index(idx)

        index_manager.create_all_indices()

        # Encode documents
        logger.info("Encoding documents...")
        doc_ids = list(data.documents.keys())
        doc_texts = [data.documents[d] for d in doc_ids]
        batch_size = config.batch_size

        encoded_docs = []
        for i in tqdm(range(0, len(doc_texts), batch_size), desc="Encoding"):
            batch_ids = doc_ids[i:i + batch_size]
            batch_texts = doc_texts[i:i + batch_size]

            dense_vecs = dense_encoder.encode(batch_texts)
            sparse_vecs = sparse_encoder.encode(batch_texts)

            for j, doc_id in enumerate(batch_ids):
                encoded_docs.append(EncodedDocument(
                    doc_id=doc_id,
                    content=batch_texts[j],
                    dense_embedding=dense_vecs[j].tolist(),
                    sparse_embedding=sparse_vecs[j],
                ))

        # Index
        tokenizer = sparse_encoder.tokenizer
        counts = index_documents(
            index_manager, encoded_docs, config, tokenizer=tokenizer,
        )
        logger.info(f"Indexed: {counts}")

    # Create searchers
    methods_to_run = args.methods or SEVEN_METHODS
    all_searchers = create_7way_searchers(
        client=index_manager.client,
        config=config,
        dense_encoder=dense_encoder,
        sparse_encoder=sparse_encoder,
    )
    searchers = {k: v for k, v in all_searchers.items() if k in methods_to_run}

    # Run benchmark
    logger.info(f"=== BENCHMARK: {args.dataset} ===")
    all_metrics: Dict[str, BenchmarkMetrics] = {}

    for method_name, searcher in searchers.items():
        logger.info(f"Running {method_name}...")
        _, metrics = run_single_method(
            method_name, searcher,
            data.queries, data.query_ids,
            data.query_relevant_docs,
        )
        all_metrics[method_name] = metrics
        logger.info(
            f"  R@1={metrics.recall_at_1:.1%} "
            f"R@5={metrics.recall_at_5:.1%} "
            f"R@10={metrics.recall_at_10:.1%} "
            f"MRR={metrics.mrr:.4f} "
            f"P50={metrics.latency_p50_ms:.0f}ms"
        )

    # Print results table
    print(f"\n{'='*90}")
    print(f"7-WAY BENCHMARK: {args.dataset}")
    print(f"Queries: {len(data.queries)}, Documents: {len(data.documents)}")
    print(f"Dense: {DENSE_MODEL}, Sparse: {SPARSE_MODEL}")
    print(f"{'='*90}")
    print(f"{'Method':<22} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR@10':>8} {'P50ms':>8}")
    print(f"{'-'*90}")

    for method_name in SEVEN_METHODS:
        if method_name not in all_metrics:
            continue
        m = all_metrics[method_name]
        print(
            f"{method_name:<22} "
            f"{m.recall_at_1:>7.1%} "
            f"{m.recall_at_5:>7.1%} "
            f"{m.recall_at_10:>7.1%} "
            f"{m.mrr:>7.4f} "
            f"{m.latency_p50_ms:>7.0f}"
        )

    # Save results
    output_dir = Path(f"outputs/benchmark_7way/{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dict = {
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "dense_model": DENSE_MODEL,
        "sparse_model": SPARSE_MODEL,
        "num_queries": len(data.queries),
        "num_documents": len(data.documents),
        "metrics": {
            name: {
                "recall_at_1": m.recall_at_1,
                "recall_at_5": m.recall_at_5,
                "recall_at_10": m.recall_at_10,
                "mrr": m.mrr,
                "ndcg_at_10": m.ndcg_at_10,
                "latency_p50_ms": m.latency_p50_ms,
                "latency_p95_ms": m.latency_p95_ms,
            }
            for name, m in all_metrics.items()
        },
    }

    results_path = output_dir / "metrics.json"
    results_path.write_text(json.dumps(results_dict, indent=2))
    logger.info(f"Results saved to {results_path}")

    # Cleanup
    if args.cleanup:
        for idx in config.index_names:
            index_manager.delete_index(idx)

    return all_metrics


if __name__ == "__main__":
    main()
