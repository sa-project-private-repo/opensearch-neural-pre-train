"""
Hybrid searcher combining Neural Sparse and Dense (Semantic) search.

Supports multiple fusion strategies:
- RRF (Reciprocal Rank Fusion) - recommended default
- Linear score combination
- Weighted RRF
"""
import logging
import time
from typing import Dict, List, Optional

from opensearchpy import OpenSearch

from benchmark.config import BenchmarkConfig
from benchmark.encoders import BgeM3Encoder, NeuralSparseEncoder
from benchmark.score_fusion import (
    RankedResult,
    ScoreFusion,
    RRFFusion,
    LinearFusion,
    WeightedRRFFusion,
    create_fusion_method,
)
from benchmark.searchers import (
    BaseSearcher,
    SearchResponse,
    SearchResult,
    SemanticSearcher,
    NeuralSparseSearcher,
)

logger = logging.getLogger(__name__)


class HybridSparseSemanticSearcher(BaseSearcher):
    """
    Hybrid search combining Neural Sparse and Dense Semantic search.

    Uses late fusion to combine results from both search methods,
    taking advantage of sparse's lexical matching and dense's
    semantic understanding.

    Expected improvement: +15-18pp Recall@1 over sparse alone.
    """

    def __init__(
        self,
        client: OpenSearch,
        sparse_index: str,
        dense_index: str,
        sparse_encoder: NeuralSparseEncoder,
        dense_encoder: BgeM3Encoder,
        fusion_method: str = "rrf",
        top_k: int = 10,
        retrieval_k: int = 100,
        **fusion_kwargs,
    ):
        """
        Initialize hybrid searcher.

        Args:
            client: OpenSearch client
            sparse_index: Index name for sparse vectors
            dense_index: Index name for dense vectors
            sparse_encoder: Neural sparse encoder
            dense_encoder: BGE-M3 encoder
            fusion_method: Fusion strategy ("rrf", "linear", "weighted_rrf")
            top_k: Number of final results to return
            retrieval_k: Number of results to retrieve from each method
                        before fusion (should be > top_k)
            **fusion_kwargs: Additional parameters for fusion method
        """
        # Note: index_name is not directly used since we query two indices
        super().__init__(client, sparse_index, top_k)

        self.sparse_index = sparse_index
        self.dense_index = dense_index
        self.sparse_encoder = sparse_encoder
        self.dense_encoder = dense_encoder
        self.retrieval_k = retrieval_k

        # Create fusion method
        self.fusion: ScoreFusion = create_fusion_method(
            fusion_method, **fusion_kwargs
        )
        self.fusion_method = fusion_method

        # Create sub-searchers
        self._sparse_searcher = NeuralSparseSearcher(
            client=client,
            index_name=sparse_index,
            encoder=sparse_encoder,
            top_k=retrieval_k,
        )
        self._semantic_searcher = SemanticSearcher(
            client=client,
            index_name=dense_index,
            encoder=dense_encoder,
            top_k=retrieval_k,
        )

    def search(self, query: str) -> SearchResponse:
        """
        Execute hybrid search with late fusion.

        Args:
            query: Search query string

        Returns:
            Fused search results
        """
        start = time.perf_counter()

        # Execute both searches
        sparse_response = self._sparse_searcher.search(query)
        dense_response = self._semantic_searcher.search(query)

        # Convert to RankedResult for fusion
        sparse_ranked = [
            RankedResult(doc_id=r.doc_id, score=r.score, rank=r.rank)
            for r in sparse_response.results
        ]
        dense_ranked = [
            RankedResult(doc_id=r.doc_id, score=r.score, rank=r.rank)
            for r in dense_response.results
        ]

        # Fuse results
        fused_results = self.fusion.fuse(sparse_ranked, dense_ranked)

        # Take top_k and convert back to SearchResult
        final_results = [
            SearchResult(
                doc_id=r.doc_id,
                score=r.score,
                rank=i + 1,  # Re-rank after fusion
            )
            for i, r in enumerate(fused_results[: self.top_k])
        ]

        total_latency = (time.perf_counter() - start) * 1000  # ms

        return SearchResponse(
            results=final_results,
            latency_ms=total_latency,
            total_hits=len(fused_results),
        )

    def get_component_results(
        self, query: str
    ) -> Dict[str, SearchResponse]:
        """
        Get individual results from each component.

        Useful for debugging and analysis.

        Args:
            query: Search query string

        Returns:
            Dict with "sparse" and "dense" SearchResponse objects
        """
        return {
            "sparse": self._sparse_searcher.search(query),
            "dense": self._semantic_searcher.search(query),
        }


class HybridNativeSearcher(BaseSearcher):
    """
    Hybrid search using OpenSearch native hybrid query.

    Uses OpenSearch 2.10+ hybrid query with normalization pipeline.
    Combines neural_sparse and k-NN in a single query.

    This is more efficient than late fusion but requires
    proper index setup with search pipelines.
    """

    def __init__(
        self,
        client: OpenSearch,
        index_name: str,
        sparse_encoder: NeuralSparseEncoder,
        dense_encoder: BgeM3Encoder,
        top_k: int = 10,
        sparse_weight: float = 0.4,
        dense_weight: float = 0.6,
        normalization: str = "min_max",
        combination: str = "arithmetic_mean",
    ):
        """
        Initialize native hybrid searcher.

        Args:
            client: OpenSearch client
            index_name: Index with both sparse and dense fields
            sparse_encoder: Neural sparse encoder
            dense_encoder: BGE-M3 encoder
            top_k: Number of results to return
            sparse_weight: Weight for sparse component
            dense_weight: Weight for dense component
            normalization: Score normalization ("min_max", "l2")
            combination: Score combination ("arithmetic_mean", "geometric_mean", "harmonic_mean")
        """
        super().__init__(client, index_name, top_k)
        self.sparse_encoder = sparse_encoder
        self.dense_encoder = dense_encoder
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.normalization = normalization
        self.combination = combination

    def search(self, query: str) -> SearchResponse:
        """Execute native hybrid search."""
        start = time.perf_counter()

        # Encode query for both methods
        query_sparse = self.sparse_encoder.encode_for_query(query, top_k=64)
        query_dense = self.dense_encoder.encode_single(query)

        if not query_sparse:
            # Fallback to dense only
            body = {
                "size": self.top_k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_dense,
                            "k": self.top_k,
                        }
                    }
                },
            }
        else:
            # Native hybrid query
            body = {
                "size": self.top_k,
                "query": {
                    "hybrid": {
                        "queries": [
                            {
                                "neural_sparse": {
                                    "sparse_embedding": {
                                        "query_tokens": query_sparse,
                                    }
                                }
                            },
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": query_dense,
                                        "k": self.top_k * 2,
                                    }
                                }
                            },
                        ]
                    }
                },
            }

        try:
            response = self.client.search(
                index=self.index_name,
                body=body,
                params={
                    "search_pipeline": f"hybrid-pipeline-{self.normalization}-{self.combination}"
                },
            )
        except Exception as e:
            # Fallback without pipeline if not configured
            logger.warning(f"Hybrid pipeline not available: {e}. Using default.")
            response = self.client.search(index=self.index_name, body=body)

        latency = (time.perf_counter() - start) * 1000

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


def create_hybrid_searchers(
    client: OpenSearch,
    config: BenchmarkConfig,
    dense_encoder: BgeM3Encoder,
    sparse_encoder: NeuralSparseEncoder,
) -> Dict[str, BaseSearcher]:
    """
    Create all hybrid searcher variants for benchmarking.

    Returns:
        Dict mapping hybrid method name to searcher instance
    """
    return {
        "hybrid_rrf": HybridSparseSemanticSearcher(
            client=client,
            sparse_index=config.sparse_index,
            dense_index=config.dense_index,
            sparse_encoder=sparse_encoder,
            dense_encoder=dense_encoder,
            fusion_method="rrf",
            top_k=config.top_k,
            retrieval_k=100,
            k=60,  # RRF constant
        ),
        "hybrid_linear_0.3": HybridSparseSemanticSearcher(
            client=client,
            sparse_index=config.sparse_index,
            dense_index=config.dense_index,
            sparse_encoder=sparse_encoder,
            dense_encoder=dense_encoder,
            fusion_method="linear",
            top_k=config.top_k,
            retrieval_k=100,
            alpha=0.3,  # 30% sparse, 70% dense
        ),
        "hybrid_linear_0.4": HybridSparseSemanticSearcher(
            client=client,
            sparse_index=config.sparse_index,
            dense_index=config.dense_index,
            sparse_encoder=sparse_encoder,
            dense_encoder=dense_encoder,
            fusion_method="linear",
            top_k=config.top_k,
            retrieval_k=100,
            alpha=0.4,  # 40% sparse, 60% dense
        ),
        "hybrid_linear_0.5": HybridSparseSemanticSearcher(
            client=client,
            sparse_index=config.sparse_index,
            dense_index=config.dense_index,
            sparse_encoder=sparse_encoder,
            dense_encoder=dense_encoder,
            fusion_method="linear",
            top_k=config.top_k,
            retrieval_k=100,
            alpha=0.5,  # 50% sparse, 50% dense
        ),
        "hybrid_weighted_rrf": HybridSparseSemanticSearcher(
            client=client,
            sparse_index=config.sparse_index,
            dense_index=config.dense_index,
            sparse_encoder=sparse_encoder,
            dense_encoder=dense_encoder,
            fusion_method="weighted_rrf",
            top_k=config.top_k,
            retrieval_k=100,
            k=60,
            sparse_weight=0.4,
            dense_weight=0.6,
        ),
    }
