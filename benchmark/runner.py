"""
Benchmark runner - main execution script.
"""
import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from benchmark.config import BenchmarkConfig
from benchmark.data_loader import BenchmarkData, load_benchmark_data
from benchmark.encoders import create_encoders, create_encoders_v29, create_encoders_v30
from benchmark.index_manager import IndexManager
from benchmark.indexer import encode_documents, index_documents
from benchmark.metrics import (
    BenchmarkMetrics,
    QueryResult,
    compute_metrics,
    paired_t_test,
)
from benchmark.report import generate_report
from benchmark.searchers import BaseSearcher, create_searchers
from benchmark.hybrid_searcher import create_hybrid_searchers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs the complete benchmark pipeline."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize runner with configuration."""
        self.config = config
        self.results: Dict[str, List[QueryResult]] = {}
        self.metrics: Dict[str, BenchmarkMetrics] = {}

    def run_setup(self) -> None:
        """Set up indices and load data."""
        logger.info("=== SETUP PHASE ===")

        # Create index manager
        logger.info("Creating index manager...")
        self.index_manager = IndexManager(self.config)

        # Create indices
        logger.info("Creating indices...")
        self.index_manager.create_all_indices()

        # Load encoders
        logger.info("Loading encoders...")
        self.dense_encoder, self.sparse_encoder = create_encoders_v29(self.config)

        # Load data
        logger.info("Loading benchmark data...")
        self.data = load_benchmark_data(self.config)

        # Encode documents
        logger.info("Encoding documents...")
        self.encoded_docs = encode_documents(
            self.data,
            self.dense_encoder,
            self.sparse_encoder,
            batch_size=self.config.batch_size,
        )

        # Index documents
        logger.info("Indexing documents...")
        counts = index_documents(
            self.index_manager,
            self.encoded_docs,
            self.config,
        )
        logger.info(f"Indexed documents: {counts}")

    def run_benchmark(
        self,
        parallel: bool = False,
        include_hybrid: bool = False,
        methods: Optional[List[str]] = None,
    ) -> None:
        """
        Run benchmark for all search methods.

        Args:
            parallel: Run methods in parallel
            include_hybrid: Include hybrid Sparse+Dense fusion methods
            methods: Specific methods to run (None = all available)
        """
        logger.info("=== BENCHMARK PHASE ===")

        # Create base searchers
        searchers = create_searchers(
            client=self.index_manager.client,
            config=self.config,
            dense_encoder=self.dense_encoder,
            sparse_encoder=self.sparse_encoder,
        )

        # Add hybrid searchers if requested
        if include_hybrid:
            logger.info("Adding hybrid Sparse+Semantic searchers...")
            hybrid_searchers = create_hybrid_searchers(
                client=self.index_manager.client,
                config=self.config,
                dense_encoder=self.dense_encoder,
                sparse_encoder=self.sparse_encoder,
            )
            searchers.update(hybrid_searchers)

        # Filter by requested methods if specified
        if methods:
            searchers = {k: v for k, v in searchers.items() if k in methods}

        # Prepare queries
        queries = self.data.queries
        query_positive_ids = self.data.query_positive_ids

        if parallel:
            self._run_benchmark_parallel(searchers, queries, query_positive_ids)
        else:
            self._run_benchmark_sequential(searchers, queries, query_positive_ids)

    def _run_benchmark_sequential(
        self,
        searchers: Dict[str, BaseSearcher],
        queries: List[str],
        query_positive_ids: Dict[str, str],
    ) -> None:
        """Run benchmark sequentially for all search methods."""
        for method_name, searcher in searchers.items():
            logger.info(f"Running {method_name} benchmark...")
            results = self._run_method(
                method_name,
                searcher,
                queries,
                query_positive_ids,
            )
            self.results[method_name] = results
            self.metrics[method_name] = compute_metrics(method_name, results)
            logger.info(
                f"{method_name}: Recall@1={self.metrics[method_name].recall_at_1:.4f}, "
                f"MRR={self.metrics[method_name].mrr:.4f}"
            )

    def _run_benchmark_parallel(
        self,
        searchers: Dict[str, BaseSearcher],
        queries: List[str],
        query_positive_ids: Dict[str, str],
    ) -> None:
        """Run benchmark in parallel for all search methods."""
        logger.info(f"Running {len(searchers)} search methods in parallel...")

        # Create progress bars for each method
        progress_bars = {
            name: tqdm(total=len(queries), desc=name, position=i)
            for i, name in enumerate(searchers.keys())
        }
        progress_lock = Lock()

        def run_single_query(
            method_name: str,
            searcher: BaseSearcher,
            query: str,
            target_doc_id: str,
        ) -> Tuple[str, QueryResult]:
            """Run a single query for a method."""
            try:
                response = searcher.search(query)
                retrieved_ids = [r.doc_id for r in response.results]
                result = QueryResult(
                    query=query,
                    target_doc_id=target_doc_id,
                    retrieved_doc_ids=retrieved_ids,
                    latency_ms=response.latency_ms,
                )
            except Exception as e:
                logger.warning(f"Search failed for {method_name} query '{query}': {e}")
                result = QueryResult(
                    query=query,
                    target_doc_id=target_doc_id,
                    retrieved_doc_ids=[],
                    latency_ms=0,
                    hit_rank=None,
                )

            with progress_lock:
                progress_bars[method_name].update(1)

            return method_name, result

        # Initialize results
        for method_name in searchers.keys():
            self.results[method_name] = []

        # Submit all tasks
        with ThreadPoolExecutor(max_workers=len(searchers)) as executor:
            futures = []
            for query in queries:
                target_doc_id = query_positive_ids[query]
                for method_name, searcher in searchers.items():
                    future = executor.submit(
                        run_single_query,
                        method_name,
                        searcher,
                        query,
                        target_doc_id,
                    )
                    futures.append(future)

            # Collect results
            for future in as_completed(futures):
                method_name, result = future.result()
                self.results[method_name].append(result)

        # Close progress bars
        for pbar in progress_bars.values():
            pbar.close()

        # Compute metrics
        for method_name in searchers.keys():
            self.metrics[method_name] = compute_metrics(
                method_name, self.results[method_name]
            )
            logger.info(
                f"{method_name}: Recall@1={self.metrics[method_name].recall_at_1:.4f}, "
                f"MRR={self.metrics[method_name].mrr:.4f}"
            )

    def _run_method(
        self,
        method_name: str,
        searcher: BaseSearcher,
        queries: List[str],
        query_positive_ids: Dict[str, str],
    ) -> List[QueryResult]:
        """Run benchmark for a single method."""
        results = []

        for query in tqdm(queries, desc=method_name):
            target_doc_id = query_positive_ids[query]

            try:
                response = searcher.search(query)
                retrieved_ids = [r.doc_id for r in response.results]

                results.append(
                    QueryResult(
                        query=query,
                        target_doc_id=target_doc_id,
                        retrieved_doc_ids=retrieved_ids,
                        latency_ms=response.latency_ms,
                    )
                )
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                results.append(
                    QueryResult(
                        query=query,
                        target_doc_id=target_doc_id,
                        retrieved_doc_ids=[],
                        latency_ms=0,
                        hit_rank=None,
                    )
                )

        return results

    def run_statistical_tests(self) -> Dict:
        """Run statistical significance tests."""
        logger.info("=== STATISTICAL TESTS ===")

        tests = {}
        methods = list(self.results.keys())

        # Compare all pairs
        for i, method_a in enumerate(methods):
            for method_b in methods[i + 1:]:
                key = f"{method_a}_vs_{method_b}"
                try:
                    tests[key] = paired_t_test(
                        self.results[method_a],
                        self.results[method_b],
                    )
                    sig = "***" if tests[key]["significant"] else ""
                    logger.info(
                        f"{key}: p={tests[key]['p_value']:.4f} {sig}"
                    )
                except Exception as e:
                    logger.warning(f"Could not run t-test for {key}: {e}")

        return tests

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate benchmark report."""
        logger.info("=== GENERATING REPORT ===")

        tests = self.run_statistical_tests()
        report = generate_report(self.metrics, tests)

        if output_path:
            output_path.write_text(report)
            logger.info(f"Report saved to {output_path}")

        return report

    def save_results(self, output_dir: Path) -> None:
        """Save detailed results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_path = output_dir / "metrics.json"
        metrics_dict = {
            name: {
                "method": m.method,
                "num_queries": m.num_queries,
                "recall_at_1": m.recall_at_1,
                "recall_at_5": m.recall_at_5,
                "recall_at_10": m.recall_at_10,
                "mrr": m.mrr,
                "ndcg_at_10": m.ndcg_at_10,
                "latency_p50_ms": m.latency_p50_ms,
                "latency_p95_ms": m.latency_p95_ms,
                "latency_p99_ms": m.latency_p99_ms,
                "latency_mean_ms": m.latency_mean_ms,
            }
            for name, m in self.metrics.items()
        }
        metrics_path.write_text(json.dumps(metrics_dict, indent=2))
        logger.info(f"Metrics saved to {metrics_path}")

    def save_benchmark_data(self, output_dir: Path) -> None:
        """Save benchmark data mapping for skip-setup mode."""
        output_dir.mkdir(parents=True, exist_ok=True)
        data_path = output_dir / "benchmark_data.json"
        data_dict = {
            "queries": self.data.queries,
            "query_positive_ids": self.data.query_positive_ids,
            "doc_ids": self.data.doc_ids,
            "sample_size": self.config.sample_size,
        }
        data_path.write_text(json.dumps(data_dict, indent=2, ensure_ascii=False))
        logger.info(f"Benchmark data saved to {data_path}")

    def load_benchmark_data_from_file(self, data_path: Path) -> BenchmarkData:
        """Load benchmark data from saved file."""
        data_dict = json.loads(data_path.read_text())
        logger.info(f"Loaded benchmark data from {data_path}")
        logger.info(f"  Queries: {len(data_dict['queries'])}")
        logger.info(f"  Documents: {len(data_dict['doc_ids'])}")
        # Note: documents content not needed for searching, only doc_ids matter
        return BenchmarkData(
            queries=data_dict["queries"],
            query_positive_ids=data_dict["query_positive_ids"],
            documents={},  # Not needed for benchmark, only for indexing
            doc_ids=data_dict["doc_ids"],
        )

    def cleanup(self) -> None:
        """Delete benchmark indices."""
        logger.info("=== CLEANUP ===")
        self.index_manager.delete_all_indices()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run OpenSearch benchmark")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Number of queries to sample",
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
        "--parallel",
        action="store_true",
        help="Run all search methods in parallel",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--include-hybrid",
        action="store_true",
        help="Include hybrid Sparse+Semantic fusion methods",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Specific methods to run (e.g., bm25 semantic neural_sparse hybrid_rrf)",
    )
    args = parser.parse_args()

    # Create config
    config = BenchmarkConfig(sample_size=args.sample_size)

    # Create runner
    runner = BenchmarkRunner(config)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_data_path = output_dir / "benchmark_data.json"

    try:
        # Setup
        if not args.skip_setup:
            runner.run_setup()
            # Save benchmark data for future skip-setup runs
            runner.save_benchmark_data(output_dir)
        else:
            # Load from saved benchmark data if available
            logger.info("Loading encoders (skip-setup mode)...")
            runner.index_manager = IndexManager(config)
            runner.dense_encoder, runner.sparse_encoder = create_encoders_v30(config)

            if benchmark_data_path.exists():
                logger.info(f"Loading saved benchmark data from {benchmark_data_path}")
                runner.data = runner.load_benchmark_data_from_file(benchmark_data_path)
            else:
                logger.warning(
                    f"No saved benchmark data found at {benchmark_data_path}. "
                    "Loading fresh data - doc IDs may not match indexed data!"
                )
                runner.data = load_benchmark_data(config)

        # Run benchmark
        runner.run_benchmark(
            parallel=args.parallel,
            include_hybrid=args.include_hybrid,
            methods=args.methods,
        )

        # Generate report
        report = runner.generate_report(output_dir / "report.md")
        print("\n" + report)

        runner.save_results(output_dir)

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        if args.cleanup:
            runner.cleanup()


if __name__ == "__main__":
    main()
