"""
Threshold grid search for V28 NeuralSparseEncoderV28.

For each threshold in THRESHOLDS:
  - Create a dedicated OpenSearch sparse index
  - Encode documents with that threshold
  - Index documents
  - Run neural_sparse queries
  - Compute Recall@1/5/10, MRR, nDCG@10
  - Clean up the index

Prints a comparison table and saves results to outputs/threshold_search/results.json.
"""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from opensearchpy.helpers import bulk
from tqdm import tqdm

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.config import BenchmarkConfig
from benchmark.encoders import NeuralSparseEncoderV28
from benchmark.hf_data_loader import MTEBBenchmarkData, load_ko_strategyqa
from benchmark.index_manager import IndexManager
from benchmark.metrics import QueryResult, compute_metrics
from benchmark.searchers import NeuralSparseSearcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "outputs/train_v28_ddp/checkpoint_epoch25_step41850/model.pt"
THRESHOLDS: List[float] = [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
OUTPUT_PATH = Path("outputs/threshold_search/results.json")
TOP_K = 10
BATCH_SIZE = 32
NUM_WORKERS = 4


def safe_index_name(threshold: float) -> str:
    """Generate a valid OpenSearch index name for a threshold value."""
    # e.g. 1e-05 -> "1e-05", replace '.' and '-' for safety
    s = f"{threshold:.0e}".replace(".", "p").replace("-", "m").replace("+", "")
    return f"benchmark-sparse-thresh-{s}"


def create_sparse_index(index_manager: IndexManager, index_name: str) -> None:
    """Create a rank_features sparse index with two-phase pipeline."""
    client = index_manager.client
    if client.indices.exists(index=index_name):
        logger.info(f"Index {index_name} already exists, deleting first...")
        client.indices.delete(index=index_name)

    body = {
        "settings": {
            "number_of_shards": 3,
            "number_of_replicas": 1,
        },
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "content": {"type": "text"},
                "sparse_embedding": {"type": "rank_features"},
            }
        },
    }
    client.indices.create(index=index_name, body=body)
    logger.info(f"Created sparse index: {index_name}")

    # Apply two-phase pipeline (best-effort)
    pipeline_id = "neural_sparse_two_phase"
    try:
        client.indices.put_settings(
            index=index_name,
            body={"index.search.default_pipeline": pipeline_id},
        )
    except Exception as e:
        logger.warning(f"Could not apply two-phase pipeline: {e}")


def index_sparse_docs(
    index_manager: IndexManager,
    index_name: str,
    doc_ids: List[str],
    doc_texts: List[str],
    sparse_vecs: List[Dict[str, float]],
    chunk_size: int = 100,
) -> int:
    """Bulk-index sparse embeddings and return number of indexed docs."""
    client = index_manager.client

    actions = [
        {
            "_index": index_name,
            "_id": doc_id,
            "_source": {
                "doc_id": doc_id,
                "content": doc_texts[i],
                "sparse_embedding": sparse_vecs[i],
            },
        }
        for i, doc_id in enumerate(doc_ids)
    ]

    success, _ = bulk(client, actions, chunk_size=chunk_size)
    client.indices.refresh(index=index_name)
    logger.info(f"Indexed {success} documents into {index_name}")
    return success


def run_neural_sparse_queries(
    index_manager: IndexManager,
    index_name: str,
    encoder: NeuralSparseEncoderV28,
    data: MTEBBenchmarkData,
    top_k: int = TOP_K,
) -> List[QueryResult]:
    """Run all queries and collect results."""
    searcher = NeuralSparseSearcher(
        client=index_manager.client,
        index_name=index_name,
        encoder=encoder,  # type: ignore[arg-type]
        top_k=top_k,
    )

    results: List[QueryResult] = []
    for i, query in enumerate(tqdm(data.queries, desc="Searching", leave=False)):
        query_id = data.query_ids[i]
        relevant_docs = set(data.query_relevant_docs.get(query_id, []))

        try:
            response = searcher.search(query)
            retrieved_ids = [r.doc_id for r in response.results]
        except Exception as e:
            logger.warning(f"Search error for query {query_id}: {e}")
            retrieved_ids = []
            response = type("R", (), {"latency_ms": 0})()  # type: ignore[assignment]

        hit_rank = None
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_docs:
                hit_rank = rank
                break

        results.append(
            QueryResult(
                query=query,
                target_doc_id=next(iter(relevant_docs), ""),
                retrieved_doc_ids=retrieved_ids,
                latency_ms=response.latency_ms,
                hit_rank=hit_rank,
            )
        )

    return results


def metrics_to_dict(m: Any) -> Dict[str, float]:
    """Convert BenchmarkMetrics to plain dict."""
    return {
        "recall_at_1": round(m.recall_at_1, 4),
        "recall_at_5": round(m.recall_at_5, 4),
        "recall_at_10": round(m.recall_at_10, 4),
        "mrr": round(m.mrr, 4),
        "ndcg_at_10": round(m.ndcg_at_10, 4),
        "latency_p50_ms": round(m.latency_p50_ms, 2),
        "latency_p95_ms": round(m.latency_p95_ms, 2),
    }


def print_table(all_results: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table."""
    header = (
        f"{'Threshold':>12} | {'R@1':>6} | {'R@5':>6} | {'R@10':>6} "
        f"| {'MRR':>6} | {'nDCG@10':>8} | {'Lat p50':>8} | {'Tokens/doc':>10}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for row in all_results:
        m = row["metrics"]
        print(
            f"{row['threshold']:>12.2e} | "
            f"{m['recall_at_1']:>6.4f} | "
            f"{m['recall_at_5']:>6.4f} | "
            f"{m['recall_at_10']:>6.4f} | "
            f"{m['mrr']:>6.4f} | "
            f"{m['ndcg_at_10']:>8.4f} | "
            f"{m['latency_p50_ms']:>8.1f} | "
            f"{row.get('avg_tokens_per_doc', 0):>10.1f}"
        )
    print(sep + "\n")


def main() -> None:
    """Entry point for threshold grid search."""
    logger.info("=== Threshold Grid Search for V28 NeuralSparseEncoder ===")

    config = BenchmarkConfig()
    index_manager = IndexManager(config)

    # Load dataset once
    logger.info("Loading Ko-StrategyQA dataset...")
    data = load_ko_strategyqa()
    doc_ids = list(data.documents.keys())
    doc_texts = [data.documents[doc_id] for doc_id in doc_ids]
    logger.info(f"Dataset: {len(data.queries)} queries, {len(doc_ids)} documents")

    all_results: List[Dict[str, Any]] = []

    for threshold in THRESHOLDS:
        index_name = safe_index_name(threshold)
        logger.info(f"\n{'='*60}")
        logger.info(f"Threshold: {threshold:.2e}  ->  index: {index_name}")
        logger.info("="*60)

        # 1. Load encoder with this threshold (model is loaded fresh each time
        #    to avoid any state leakage, though threshold is the only difference)
        t0 = time.time()
        encoder = NeuralSparseEncoderV28(
            checkpoint_path=PROJECT_ROOT / CHECKPOINT_PATH,
            device=config.device,
            threshold=threshold,
        )
        logger.info(f"Encoder loaded in {time.time() - t0:.1f}s")

        # 2. Encode documents
        logger.info(f"Encoding {len(doc_texts)} documents (threshold={threshold:.2e})...")
        t0 = time.time()
        sparse_vecs = encoder.encode(
            doc_texts,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
        encode_secs = time.time() - t0
        logger.info(f"Encoding done in {encode_secs:.1f}s")

        # Stats: average non-zero tokens per document
        avg_tokens = sum(len(v) for v in sparse_vecs) / max(len(sparse_vecs), 1)
        logger.info(f"Avg tokens/doc: {avg_tokens:.1f}")

        # 3. Create index and index documents
        create_sparse_index(index_manager, index_name)
        index_sparse_docs(
            index_manager, index_name, doc_ids, doc_texts, sparse_vecs
        )

        # 4. Run queries and compute metrics
        logger.info("Running queries...")
        query_results = run_neural_sparse_queries(
            index_manager, index_name, encoder, data, top_k=TOP_K
        )
        metrics = compute_metrics(f"thresh_{threshold:.2e}", query_results)
        m_dict = metrics_to_dict(metrics)

        logger.info(
            f"Results: R@1={m_dict['recall_at_1']:.4f}, "
            f"R@10={m_dict['recall_at_10']:.4f}, "
            f"MRR={m_dict['mrr']:.4f}, "
            f"nDCG@10={m_dict['ndcg_at_10']:.4f}"
        )

        all_results.append({
            "threshold": threshold,
            "index_name": index_name,
            "avg_tokens_per_doc": round(avg_tokens, 2),
            "encode_secs": round(encode_secs, 1),
            "metrics": m_dict,
        })

        # 5. Clean up index to save storage
        logger.info(f"Cleaning up index: {index_name}")
        index_manager.delete_index(index_name)

        # Free GPU memory before next iteration
        import gc
        import torch
        del encoder
        torch.cuda.empty_cache()
        gc.collect()

    # Print comparison table
    print_table(all_results)

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset": "Ko-StrategyQA",
        "checkpoint": CHECKPOINT_PATH,
        "num_queries": len(data.queries),
        "num_documents": len(doc_ids),
        "thresholds": THRESHOLDS,
        "results": all_results,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    logger.info(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
