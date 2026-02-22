"""
Benchmark report generation.
"""
from datetime import datetime
from typing import Dict, List, Optional

from benchmark.metrics import BenchmarkMetrics


def generate_report(
    metrics: Dict[str, BenchmarkMetrics],
    statistical_tests: Optional[Dict] = None,
) -> str:
    """
    Generate markdown report from benchmark metrics.

    Args:
        metrics: Dict mapping method name to BenchmarkMetrics
        statistical_tests: Optional dict of statistical test results

    Returns:
        Markdown formatted report
    """
    lines = []

    # Header
    lines.append("# OpenSearch Retrieval Benchmark Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary table
    lines.append("## Performance Summary")
    lines.append("")
    lines.append(
        "| Method | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | "
        "P50 (ms) | P95 (ms) |"
    )
    lines.append(
        "|--------|----------|----------|-----------|-----|---------|"
        "----------|----------|"
    )

    # Sort by Recall@1 descending
    sorted_methods = sorted(
        metrics.items(),
        key=lambda x: x[1].recall_at_1,
        reverse=True,
    )

    best_recall = sorted_methods[0][1].recall_at_1 if sorted_methods else 0

    for method, m in sorted_methods:
        is_best = m.recall_at_1 == best_recall
        best_marker = " **" if is_best else ""
        end_marker = "**" if is_best else ""

        lines.append(
            f"| {best_marker}{method}{end_marker} | "
            f"{m.recall_at_1:.1%} | {m.recall_at_5:.1%} | {m.recall_at_10:.1%} | "
            f"{m.mrr:.4f} | {m.ndcg_at_10:.4f} | "
            f"{m.latency_p50_ms:.1f} | {m.latency_p95_ms:.1f} |"
        )

    lines.append("")

    # Detailed metrics
    lines.append("## Detailed Metrics")
    lines.append("")

    for method, m in sorted_methods:
        lines.append(f"### {method}")
        lines.append("")
        lines.append(f"- **Queries**: {m.num_queries}")
        lines.append(f"- **Recall@1**: {m.recall_at_1:.4f} ({m.recall_at_1:.1%})")
        lines.append(f"- **Recall@5**: {m.recall_at_5:.4f} ({m.recall_at_5:.1%})")
        lines.append(f"- **Recall@10**: {m.recall_at_10:.4f} ({m.recall_at_10:.1%})")
        lines.append(f"- **MRR**: {m.mrr:.4f}")
        lines.append(f"- **nDCG@10**: {m.ndcg_at_10:.4f}")
        lines.append(f"- **Latency P50**: {m.latency_p50_ms:.2f} ms")
        lines.append(f"- **Latency P95**: {m.latency_p95_ms:.2f} ms")
        lines.append(f"- **Latency P99**: {m.latency_p99_ms:.2f} ms")
        lines.append(f"- **Latency Mean**: {m.latency_mean_ms:.2f} ms")
        lines.append("")

    # Statistical tests
    if statistical_tests:
        lines.append("## Statistical Significance")
        lines.append("")
        lines.append("Paired t-test on reciprocal ranks (p < 0.05 is significant):")
        lines.append("")

        for comparison, result in statistical_tests.items():
            sig = "✓ significant" if result["significant"] else "✗ not significant"
            lines.append(
                f"- **{comparison}**: p = {result['p_value']:.4f} ({sig})"
            )

        lines.append("")

    # Analysis
    lines.append("## Analysis")
    lines.append("")

    # Find best method
    if sorted_methods:
        best_method, best_metrics = sorted_methods[0]
        lines.append(f"**Best performing method**: {best_method}")
        lines.append("")

        # Compare to BM25 baseline
        if "bm25" in metrics and best_method != "bm25":
            bm25_recall = metrics["bm25"].recall_at_1
            if bm25_recall > 0:
                improvement = (
                    (best_metrics.recall_at_1 - bm25_recall)
                    / bm25_recall
                    * 100
                )
                lines.append(
                    f"- Recall@1 improvement over BM25: "
                    f"{improvement:+.1f}%"
                )
            else:
                lines.append(
                    "- BM25 Recall@1 is 0 (no results)"
                )

        # Neural sparse vs semantic
        if "neural_sparse" in metrics and "semantic" in metrics:
            ns_recall = metrics["neural_sparse"].recall_at_1
            sem_recall = metrics["semantic"].recall_at_1
            if ns_recall > sem_recall:
                lines.append(
                    f"- Neural Sparse outperforms Semantic by "
                    f"{(ns_recall - sem_recall) * 100:.1f}pp on Recall@1"
                )
            else:
                lines.append(
                    f"- Semantic outperforms Neural Sparse by "
                    f"{(sem_recall - ns_recall) * 100:.1f}pp on Recall@1"
                )

        # Latency comparison
        if "neural_sparse" in metrics:
            ns_latency = metrics["neural_sparse"].latency_p50_ms
            other_latencies = [
                (name, m.latency_p50_ms)
                for name, m in metrics.items()
                if name != "neural_sparse"
            ]
            if other_latencies:
                avg_other = sum(l for _, l in other_latencies) / len(other_latencies)
                if ns_latency < avg_other:
                    lines.append(
                        f"- Neural Sparse is {avg_other / ns_latency:.1f}x faster "
                        f"than average on P50 latency"
                    )

    lines.append("")
    lines.append("---")
    lines.append("*Report generated by OpenSearch Benchmark v1.0*")

    return "\n".join(lines)


def format_metrics_table(metrics: Dict[str, BenchmarkMetrics]) -> str:
    """Generate a simple ASCII table of metrics."""
    header = (
        f"{'Method':<15} {'R@1':<8} {'R@5':<8} {'R@10':<8} "
        f"{'MRR':<8} {'nDCG@10':<8} {'P50':<8}"
    )
    separator = "-" * len(header)

    lines = [header, separator]

    for method, m in sorted(
        metrics.items(),
        key=lambda x: x[1].recall_at_1,
        reverse=True,
    ):
        lines.append(
            f"{method:<15} {m.recall_at_1:<8.1%} {m.recall_at_5:<8.1%} "
            f"{m.recall_at_10:<8.1%} {m.mrr:<8.4f} {m.ndcg_at_10:<8.4f} "
            f"{m.latency_p50_ms:<8.1f}"
        )

    return "\n".join(lines)
