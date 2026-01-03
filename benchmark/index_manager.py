"""
OpenSearch index management for benchmark.
"""
import logging
from typing import Dict, Any, Optional

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

from benchmark.config import BenchmarkConfig

logger = logging.getLogger(__name__)


class IndexManager:
    """Manages OpenSearch indices for benchmark."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize index manager with configuration."""
        self.config = config
        self.client = self._create_client()

    def _create_client(self) -> OpenSearch:
        """Create OpenSearch client with AWS authentication."""
        credentials = boto3.Session().get_credentials()
        aws_auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            self.config.opensearch_region,
            "es",
            session_token=credentials.token,
        )

        return OpenSearch(
            hosts=[{
                "host": self.config.opensearch_host,
                "port": self.config.opensearch_port,
            }],
            http_auth=aws_auth,
            use_ssl=self.config.use_ssl,
            verify_certs=self.config.verify_certs,
            connection_class=RequestsHttpConnection,
        )

    def create_bm25_index(self) -> None:
        """Create BM25 index with Korean analyzer."""
        index_name = self.config.bm25_index
        if self.client.indices.exists(index=index_name):
            logger.info(f"Index {index_name} already exists, skipping creation")
            return

        body = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "korean_analyzer": {
                            "type": "custom",
                            "tokenizer": "nori_tokenizer",
                        }
                    }
                },
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "korean_analyzer",
                    },
                }
            },
        }

        self.client.indices.create(index=index_name, body=body)
        logger.info(f"Created index: {index_name}")

    def create_dense_index(self) -> None:
        """Create dense vector index for k-NN search."""
        index_name = self.config.dense_index
        if self.client.indices.exists(index=index_name):
            logger.info(f"Index {index_name} already exists, skipping creation")
            return

        body = {
            "settings": {
                "index": {"knn": True},
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "method": {
                            "name": "hnsw",
                            "engine": "faiss",
                            "space_type": "innerproduct",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 16,
                            },
                        },
                    },
                }
            },
        }

        self.client.indices.create(index=index_name, body=body)
        logger.info(f"Created index: {index_name}")

    def create_sparse_index(self) -> None:
        """Create sparse vector index with rank_features."""
        index_name = self.config.sparse_index
        if self.client.indices.exists(index=index_name):
            logger.info(f"Index {index_name} already exists, skipping creation")
            return

        body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "content": {"type": "text"},
                    "sparse_embedding": {"type": "rank_features"},
                }
            },
        }

        self.client.indices.create(index=index_name, body=body)
        logger.info(f"Created index: {index_name}")

    def create_hybrid_index(self) -> None:
        """Create hybrid index with both text and dense vector."""
        index_name = self.config.hybrid_index
        if self.client.indices.exists(index=index_name):
            logger.info(f"Index {index_name} already exists, skipping creation")
            return

        body = {
            "settings": {
                "index": {"knn": True},
                "analysis": {
                    "analyzer": {
                        "korean_analyzer": {
                            "type": "custom",
                            "tokenizer": "nori_tokenizer",
                        }
                    }
                },
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "korean_analyzer",
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "method": {
                            "name": "hnsw",
                            "engine": "faiss",
                            "space_type": "innerproduct",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 16,
                            },
                        },
                    },
                }
            },
        }

        self.client.indices.create(index=index_name, body=body)
        logger.info(f"Created index: {index_name}")

    def create_all_indices(self) -> None:
        """Create all benchmark indices."""
        self.create_bm25_index()
        self.create_dense_index()
        self.create_sparse_index()
        self.create_hybrid_index()
        logger.info("All indices created successfully")

    def delete_index(self, index_name: str) -> None:
        """Delete an index if it exists."""
        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)
            logger.info(f"Deleted index: {index_name}")

    def delete_all_indices(self) -> None:
        """Delete all benchmark indices."""
        for index_name in self.config.index_names:
            self.delete_index(index_name)
        logger.info("All indices deleted")

    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics for an index."""
        if not self.client.indices.exists(index=index_name):
            return {"error": f"Index {index_name} does not exist"}
        return self.client.indices.stats(index=index_name)

    def refresh_index(self, index_name: str) -> None:
        """Refresh an index to make documents searchable."""
        self.client.indices.refresh(index=index_name)
        logger.info(f"Refreshed index: {index_name}")
