"""
Search implementations for benchmark.
"""
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from opensearchpy import OpenSearch

from benchmark.config import BenchmarkConfig
from benchmark.encoders import BgeM3Encoder, NeuralSparseEncoder

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result."""

    doc_id: str
    score: float
    rank: int


@dataclass
class SearchResponse:
    """Search response with results and timing."""

    results: List[SearchResult]
    latency_ms: float
    total_hits: int


class BaseSearcher(ABC):
    """Abstract base class for searchers."""

    def __init__(
        self,
        client: OpenSearch,
        index_name: str,
        top_k: int = 10,
    ):
        """Initialize searcher."""
        self.client = client
        self.index_name = index_name
        self.top_k = top_k

    @abstractmethod
    def search(self, query: str) -> SearchResponse:
        """Execute search and return results."""
        pass

    def _execute_search(self, body: Dict) -> SearchResponse:
        """Execute search with timing."""
        start = time.perf_counter()
        response = self.client.search(index=self.index_name, body=body)
        latency = (time.perf_counter() - start) * 1000  # ms

        hits = response["hits"]["hits"]
        results = [
            SearchResult(
                doc_id=hit["_id"],
                score=hit["_score"],
                rank=i + 1,
            )
            for i, hit in enumerate(hits)
        ]

        return SearchResponse(
            results=results,
            latency_ms=latency,
            total_hits=response["hits"]["total"]["value"],
        )


class BM25Searcher(BaseSearcher):
    """BM25 lexical search."""

    def search(self, query: str) -> SearchResponse:
        """Search using BM25."""
        body = {
            "size": self.top_k,
            "query": {
                "match": {
                    "content": {
                        "query": query,
                        "analyzer": "korean_analyzer",
                    }
                }
            },
        }
        return self._execute_search(body)


class SemanticSearcher(BaseSearcher):
    """Dense vector semantic search."""

    def __init__(
        self,
        client: OpenSearch,
        index_name: str,
        encoder: BgeM3Encoder,
        top_k: int = 10,
    ):
        """Initialize semantic searcher with encoder."""
        super().__init__(client, index_name, top_k)
        self.encoder = encoder

    def search(self, query: str) -> SearchResponse:
        """Search using k-NN."""
        # Encode query
        query_vector = self.encoder.encode_single(query)

        body = {
            "size": self.top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": self.top_k,
                    }
                }
            },
        }
        return self._execute_search(body)


class NeuralSparseSearcher(BaseSearcher):
    """Neural sparse search using sparse_vector ANN (SEISMIC)."""

    def __init__(
        self,
        client: OpenSearch,
        index_name: str,
        encoder: NeuralSparseEncoder,
        top_k: int = 10,
    ):
        """Initialize neural sparse searcher with encoder."""
        super().__init__(client, index_name, top_k)
        self.encoder = encoder

    def search(self, query: str) -> SearchResponse:
        """Search using neural sparse vectors with native neural_sparse query."""
        # Encode query (top_k=64 for good coverage)
        query_sparse = self.encoder.encode_for_query(query, top_k=64)

        if not query_sparse:
            # Empty query, return empty results
            return SearchResponse(results=[], latency_ms=0, total_hits=0)

        # Use OpenSearch native neural_sparse query with pre-encoded tokens
        body = {
            "size": self.top_k,
            "query": {
                "neural_sparse": {
                    "sparse_embedding": {
                        "query_tokens": query_sparse,
                    }
                }
            },
        }
        return self._execute_search(body)


class HybridSearcher(BaseSearcher):
    """Hybrid search combining BM25 and k-NN."""

    def __init__(
        self,
        client: OpenSearch,
        index_name: str,
        encoder: BgeM3Encoder,
        top_k: int = 10,
        bm25_weight: float = 0.3,
        knn_weight: float = 0.7,
    ):
        """Initialize hybrid searcher."""
        super().__init__(client, index_name, top_k)
        self.encoder = encoder
        self.bm25_weight = bm25_weight
        self.knn_weight = knn_weight

    def search(self, query: str) -> SearchResponse:
        """Search using hybrid query."""
        query_vector = self.encoder.encode_single(query)

        # OpenSearch 2.10+ hybrid query
        body = {
            "size": self.top_k,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "match": {
                                "content": {
                                    "query": query,
                                    "analyzer": "korean_analyzer",
                                }
                            }
                        },
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": self.top_k,
                                }
                            }
                        },
                    ]
                }
            },
        }
        return self._execute_search(body)


def create_searchers(
    client: OpenSearch,
    config: BenchmarkConfig,
    dense_encoder: BgeM3Encoder,
    sparse_encoder: NeuralSparseEncoder,
) -> Dict[str, BaseSearcher]:
    """
    Create all searchers from config.

    Returns:
        Dict mapping search method name to searcher instance
    """
    return {
        "bm25": BM25Searcher(
            client=client,
            index_name=config.bm25_index,
            top_k=config.top_k,
        ),
        "semantic": SemanticSearcher(
            client=client,
            index_name=config.dense_index,
            encoder=dense_encoder,
            top_k=config.top_k,
        ),
        "neural_sparse": NeuralSparseSearcher(
            client=client,
            index_name=config.sparse_index,
            encoder=sparse_encoder,
            top_k=config.top_k,
        ),
        "hybrid": HybridSearcher(
            client=client,
            index_name=config.hybrid_index,
            encoder=dense_encoder,
            top_k=config.top_k,
        ),
    }
