"""
Benchmark runner for HuggingFace MTEB-style retrieval datasets.
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
from benchmark.encoders import create_encoders, create_encoders_v28, create_encoders_v29, create_encoders_v30, create_encoders_v33
from benchmark.hf_data_loader import MTEBBenchmarkData, load_hf_dataset
from benchmark.index_manager import IndexManager
from benchmark.metrics import BenchmarkMetrics, QueryResult, compute_metrics, paired_t_test
from benchmark.report import generate_report
from benchmark.searchers import create_searchers
from benchmark.hybrid_searcher import create_hybrid_searchers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class HFBenchmarkRunner:
    """Runs benchmark on HuggingFace MTEB-style datasets."""

    def __init__(self, config: BenchmarkConfig, dataset_name: str, checkpoint_path: Optional[str] = None):
        self.config = config
        self.dataset_name = dataset_name
        self.checkpoint_path = checkpoint_path
        self.results: Dict[str, List[QueryResult]] = {}
        self.metrics: Dict[str, BenchmarkMetrics] = {}

    def run_setup(self, max_queries: Optional[int] = None) -> None:
        """Set up indices and load data."""
        logger.info("=== SETUP PHASE ===")

        # Create index manager
        logger.info("Creating index manager...")
        self.index_manager = IndexManager(self.config)

        # Create indices
        logger.info("Creating indices...")
        self.index_manager.create_all_indices()

        # Load encoders
        ckpt = self.checkpoint_path
        if ckpt and ("v33" in ckpt or "train_v33" in ckpt):
            logger.info(f"Loading V33 encoders (checkpoint: {ckpt})...")
            self.dense_encoder, self.sparse_encoder = create_encoders_v33(
                self.config,
                checkpoint_path=ckpt,
            )
        elif ckpt and ("v30" in ckpt or "train_v30" in ckpt):
            logger.info(f"Loading V30 encoders (checkpoint: {ckpt})...")
            self.dense_encoder, self.sparse_encoder = create_encoders_v30(
                self.config,
                checkpoint_path=ckpt,
            )
        else:
            ckpt = ckpt or "outputs/train_v28_ddp/checkpoint_epoch30_step63180/model.pt"
            logger.info(f"Loading V28 encoders (checkpoint: {ckpt})...")
            self.dense_encoder, self.sparse_encoder = create_encoders_v28(
                self.config,
                checkpoint_path=ckpt,
            )

        # Load HuggingFace dataset
        logger.info(f"Loading {self.dataset_name} dataset...")
        self.data = load_hf_dataset(self.dataset_name, max_queries)

        # Encode and index documents
        logger.info(f"Encoding and indexing {len(self.data.documents)} documents...")
        self._index_documents()

    def _index_documents(self) -> None:
        """Encode and index documents from MTEB dataset."""
        from benchmark.indexer import EncodedDocument

        batch_size = self.config.batch_size
        doc_ids = list(self.data.documents.keys())
        doc_texts = [self.data.documents[doc_id] for doc_id in doc_ids]

        encoded_docs = []

        for i in tqdm(range(0, len(doc_texts), batch_size), desc="Encoding"):
            batch_ids = doc_ids[i:i + batch_size]
            batch_texts = doc_texts[i:i + batch_size]

            # Dense encoding
            dense_vecs = self.dense_encoder.encode(batch_texts)

            # Sparse encoding
            sparse_vecs = self.sparse_encoder.encode(batch_texts)

            for j, doc_id in enumerate(batch_ids):
                encoded_docs.append(
                    EncodedDocument(
                        doc_id=doc_id,
                        content=batch_texts[j],
                        dense_embedding=dense_vecs[j].tolist(),
                        sparse_embedding=sparse_vecs[j],
                    )
                )

        # Index all documents (pass tokenizer for sparse_vector int key conversion)
        from benchmark.indexer import index_documents
        tokenizer = getattr(self.sparse_encoder, 'tokenizer', None)
        counts = index_documents(
            self.index_manager, encoded_docs, self.config, tokenizer=tokenizer
        )
        logger.info(f"Indexed documents: {counts}")

    def run_benchmark(
        self,
        include_hybrid: bool = True,
        methods: Optional[List[str]] = None,
    ) -> None:
        """Run benchmark for all search methods."""
        logger.info("=== BENCHMARK PHASE ===")

        # Create searchers
        searchers = create_searchers(
            client=self.index_manager.client,
            config=self.config,
            dense_encoder=self.dense_encoder,
            sparse_encoder=self.sparse_encoder,
        )

        if include_hybrid:
            hybrid_searchers = create_hybrid_searchers(
                client=self.index_manager.client,
                config=self.config,
                dense_encoder=self.dense_encoder,
                sparse_encoder=self.sparse_encoder,
            )
            searchers.update(hybrid_searchers)

        if methods:
            searchers = {k: v for k, v in searchers.items() if k in methods}

        # Run benchmark
        for method_name, searcher in searchers.items():
            logger.info(f"Running {method_name} benchmark...")
            results = self._run_method(method_name, searcher)
            self.results[method_name] = results
            self.metrics[method_name] = self._compute_mteb_metrics(method_name, results)
            logger.info(
                f"{method_name}: Recall@1={self.metrics[method_name].recall_at_1:.4f}, "
                f"MRR={self.metrics[method_name].mrr:.4f}"
            )

    def _run_method(self, method_name: str, searcher) -> List[QueryResult]:
        """Run benchmark for a single method."""
        results = []

        for i, query in enumerate(tqdm(self.data.queries, desc=method_name)):
            query_id = self.data.query_ids[i]
            relevant_docs = self.data.query_relevant_docs.get(query_id, [])

            try:
                response = searcher.search(query)
                retrieved_ids = [r.doc_id for r in response.results]

                # Find rank of first relevant document
                hit_rank = None
                for rank, doc_id in enumerate(retrieved_ids, 1):
                    if doc_id in relevant_docs:
                        hit_rank = rank
                        break

                results.append(
                    QueryResult(
                        query=query,
                        target_doc_id=relevant_docs[0] if relevant_docs else "",
                        retrieved_doc_ids=retrieved_ids,
                        latency_ms=response.latency_ms,
                        hit_rank=hit_rank,
                    )
                )
            except Exception as e:
                logger.warning(f"Search failed for query '{query[:50]}...': {e}")
                results.append(
                    QueryResult(
                        query=query,
                        target_doc_id=relevant_docs[0] if relevant_docs else "",
                        retrieved_doc_ids=[],
                        latency_ms=0,
                        hit_rank=None,
                    )
                )

        return results

    def _compute_mteb_metrics(
        self, method_name: str, results: List[QueryResult]
    ) -> BenchmarkMetrics:
        """Compute metrics for MTEB-style evaluation."""
        # Use any relevant doc as hit (not just first)
        adjusted_results = []

        for i, result in enumerate(results):
            query_id = self.data.query_ids[i]
            relevant_docs = set(self.data.query_relevant_docs.get(query_id, []))

            # Recalculate hit rank considering all relevant docs
            hit_rank = None
            for rank, doc_id in enumerate(result.retrieved_doc_ids, 1):
                if doc_id in relevant_docs:
                    hit_rank = rank
                    break

            adjusted_results.append(
                QueryResult(
                    query=result.query,
                    target_doc_id=result.target_doc_id,
                    retrieved_doc_ids=result.retrieved_doc_ids,
                    latency_ms=result.latency_ms,
                    hit_rank=hit_rank,
                )
            )

        return compute_metrics(method_name, adjusted_results)

    def run_statistical_tests(self) -> Dict:
        """Run statistical significance tests."""
        logger.info("=== STATISTICAL TESTS ===")

        tests = {}
        methods = list(self.results.keys())

        for i, method_a in enumerate(methods):
            for method_b in methods[i + 1:]:
                key = f"{method_a}_vs_{method_b}"
                try:
                    tests[key] = paired_t_test(
                        self.results[method_a],
                        self.results[method_b],
                    )
                    sig = "***" if tests[key]["significant"] else ""
                    logger.info(f"{key}: p={tests[key]['p_value']:.4f} {sig}")
                except Exception as e:
                    logger.warning(f"Could not run t-test for {key}: {e}")

        return tests

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate benchmark report."""
        logger.info("=== GENERATING REPORT ===")

        tests = self.run_statistical_tests()
        report = generate_report(self.metrics, tests)

        # Add dataset info header
        header = f"""# {self.data.dataset_name} Benchmark Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset**: {self.data.dataset_name}
**Queries**: {len(self.data.queries)}
**Documents**: {len(self.data.documents)}

---

"""
        report = header + report

        if output_path:
            output_path.write_text(report)
            logger.info(f"Report saved to {output_path}")

        return report

    def save_results(self, output_dir: Path) -> None:
        """Save detailed results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = output_dir / "metrics.json"
        metrics_dict = {
            "dataset": self.data.dataset_name,
            "num_queries": len(self.data.queries),
            "num_documents": len(self.data.documents),
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
                for name, m in self.metrics.items()
            },
        }
        metrics_path.write_text(json.dumps(metrics_dict, indent=2))
        logger.info(f"Metrics saved to {metrics_path}")

    def cleanup(self) -> None:
        """Delete benchmark indices."""
        logger.info("=== CLEANUP ===")
        self.index_manager.delete_all_indices()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run benchmark on HuggingFace retrieval datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ko-strategyqa",
        choices=["ko-strategyqa", "miracl-ko", "mrtydi-ko"],
        help="Dataset to benchmark",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of queries (None = all)",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip index creation and data loading",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete indices after benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/benchmark_hf",
        help="Output directory for results",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Specific methods to run",
    )
    parser.add_argument(
        "--index-suffix",
        type=str,
        default="hf",
        help="Suffix for index names",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: hardcoded V28 path)",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        choices=["v28", "v29", "v30", "v33"],
        help="Model version to use (auto-detected from checkpoint path if not specified)",
    )
    args = parser.parse_args()

    # Create config with custom index names
    config = BenchmarkConfig()
    config.bm25_index = f"benchmark-bm25-{args.index_suffix}"
    config.dense_index = f"benchmark-dense-{args.index_suffix}"
    config.sparse_index = f"benchmark-sparse-{args.index_suffix}"
    config.hybrid_index = f"benchmark-hybrid-{args.index_suffix}"

    # Create runner
    runner = HFBenchmarkRunner(config, args.dataset, checkpoint_path=args.checkpoint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not args.skip_setup:
            runner.run_setup(args.max_queries)
        else:
            logger.info("Loading encoders (skip-setup mode)...")
            runner.index_manager = IndexManager(config)
            runner.dense_encoder, runner.sparse_encoder = create_encoders_v29(config)
            runner.data = load_hf_dataset(args.dataset, args.max_queries)

        runner.run_benchmark(include_hybrid=True, methods=args.methods)
        runner.save_results(output_dir)
        report = runner.generate_report(output_dir / "report.md")
        print("\n" + report)

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        if args.cleanup:
            runner.cleanup()


if __name__ == "__main__":
    main()
