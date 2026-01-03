"""
OpenSearch Retrieval Benchmark Module.

Compares 4 search methods:
- BM25: Lexical search baseline
- Semantic: Dense vector search (bge-m3)
- Neural Sparse: Korean sparse encoder (v21.4)
- Hybrid: BM25 + Semantic combination
"""

from benchmark.config import BenchmarkConfig
from benchmark.runner import BenchmarkRunner

__all__ = ["BenchmarkConfig", "BenchmarkRunner"]
