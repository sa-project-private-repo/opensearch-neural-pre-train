"""
Document indexing for benchmark.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

from tqdm import tqdm

from benchmark.config import BenchmarkConfig
from benchmark.data_loader import BenchmarkData
from benchmark.encoders import BgeM3Encoder, NeuralSparseEncoder
from benchmark.index_manager import IndexManager

logger = logging.getLogger(__name__)


@dataclass
class EncodedDocument:
    """Document with all encoded representations."""

    doc_id: str
    content: str
    dense_embedding: List[float]
    sparse_embedding: Dict[str, float]


def encode_documents(
    data: BenchmarkData,
    dense_encoder: BgeM3Encoder,
    sparse_encoder: NeuralSparseEncoder,
    batch_size: int = 32,
    num_workers: int = 4,
) -> List[EncodedDocument]:
    """
    Encode all documents with both encoders.

    Args:
        data: Benchmark data with documents
        dense_encoder: Dense encoder (bge-m3)
        sparse_encoder: Sparse encoder (v21.4)
        batch_size: Batch size for encoding
        num_workers: Number of parallel workers for sparse encoding

    Returns:
        List of encoded documents
    """
    doc_ids = list(data.documents.keys())
    contents = [data.documents[doc_id] for doc_id in doc_ids]

    logger.info(f"Encoding {len(contents)} documents...")

    # Dense encoding
    logger.info("Generating dense embeddings...")
    dense_embeddings = dense_encoder.encode(contents, batch_size=batch_size)

    # Sparse encoding with parallel workers
    logger.info("Generating sparse embeddings...")
    sparse_embeddings = sparse_encoder.encode(
        contents, batch_size=batch_size, num_workers=num_workers
    )

    # Combine
    encoded_docs = []
    for i, doc_id in enumerate(doc_ids):
        encoded_docs.append(
            EncodedDocument(
                doc_id=doc_id,
                content=contents[i],
                dense_embedding=dense_embeddings[i].tolist(),
                sparse_embedding=sparse_embeddings[i],
            )
        )

    logger.info(f"Encoded {len(encoded_docs)} documents")
    return encoded_docs


def bulk_actions_bm25(
    docs: List[EncodedDocument],
    index_name: str,
) -> Iterator[Dict]:
    """Generate bulk index actions for BM25 index."""
    for doc in docs:
        yield {
            "_index": index_name,
            "_id": doc.doc_id,
            "_source": {
                "doc_id": doc.doc_id,
                "content": doc.content,
            },
        }


def bulk_actions_dense(
    docs: List[EncodedDocument],
    index_name: str,
) -> Iterator[Dict]:
    """Generate bulk index actions for dense index."""
    for doc in docs:
        yield {
            "_index": index_name,
            "_id": doc.doc_id,
            "_source": {
                "doc_id": doc.doc_id,
                "content": doc.content,
                "embedding": doc.dense_embedding,
            },
        }


def bulk_actions_sparse(
    docs: List[EncodedDocument],
    index_name: str,
    tokenizer=None,
) -> Iterator[Dict]:
    """Generate bulk index actions for sparse index.

    sparse_vector type requires integer token IDs as keys.
    Converts token strings to IDs using the tokenizer vocab.
    """
    # Build token-to-id lookup if tokenizer provided
    token_to_id: Optional[Dict[str, int]] = None
    if tokenizer is not None:
        token_to_id = tokenizer.get_vocab()

    for doc in docs:
        sparse = doc.sparse_embedding
        if token_to_id is not None:
            # Convert string token keys to integer IDs
            sparse = {}
            for token, weight in doc.sparse_embedding.items():
                tid = token_to_id.get(token)
                if tid is not None:
                    sparse[str(tid)] = weight
        yield {
            "_index": index_name,
            "_id": doc.doc_id,
            "_source": {
                "doc_id": doc.doc_id,
                "content": doc.content,
                "sparse_embedding": sparse,
            },
        }


def bulk_actions_hybrid(
    docs: List[EncodedDocument],
    index_name: str,
) -> Iterator[Dict]:
    """Generate bulk index actions for hybrid index."""
    for doc in docs:
        yield {
            "_index": index_name,
            "_id": doc.doc_id,
            "_source": {
                "doc_id": doc.doc_id,
                "content": doc.content,
                "embedding": doc.dense_embedding,
            },
        }


def index_documents(
    index_manager: IndexManager,
    encoded_docs: List[EncodedDocument],
    config: BenchmarkConfig,
    chunk_size: int = 100,
    tokenizer=None,
) -> Dict[str, int]:
    """
    Index documents to all benchmark indices.

    Args:
        index_manager: OpenSearch index manager
        encoded_docs: Encoded documents
        config: Benchmark configuration
        chunk_size: Bulk indexing chunk size
        tokenizer: Tokenizer for converting token strings to IDs (sparse_vector)

    Returns:
        Dict with document counts per index
    """
    from opensearchpy.helpers import bulk

    client = index_manager.client
    counts = {}

    # BM25 index
    logger.info(f"Indexing to {config.bm25_index}...")
    actions = list(bulk_actions_bm25(encoded_docs, config.bm25_index))
    success, _ = bulk(client, actions, chunk_size=chunk_size)
    counts[config.bm25_index] = success
    index_manager.refresh_index(config.bm25_index)

    # Dense index
    logger.info(f"Indexing to {config.dense_index}...")
    actions = list(bulk_actions_dense(encoded_docs, config.dense_index))
    success, _ = bulk(client, actions, chunk_size=chunk_size)
    counts[config.dense_index] = success
    index_manager.refresh_index(config.dense_index)

    # Sparse index (token strings -> integer IDs for sparse_vector)
    logger.info(f"Indexing to {config.sparse_index}...")
    actions = list(bulk_actions_sparse(
        encoded_docs, config.sparse_index, tokenizer=tokenizer
    ))
    success, _ = bulk(client, actions, chunk_size=chunk_size)
    counts[config.sparse_index] = success
    index_manager.refresh_index(config.sparse_index)

    # Hybrid index
    logger.info(f"Indexing to {config.hybrid_index}...")
    actions = list(bulk_actions_hybrid(encoded_docs, config.hybrid_index))
    success, _ = bulk(client, actions, chunk_size=chunk_size)
    counts[config.hybrid_index] = success
    index_manager.refresh_index(config.hybrid_index)

    logger.info(f"Indexing complete: {counts}")
    return counts
