"""SPLADE-v3 model wrapper for sparse retrieval.

This module provides a wrapper for the naver/splade-v3 model from Hugging Face.
SPLADE-v3 is a state-of-the-art sparse retrieval model that produces sparse
vectors for efficient retrieval.

Reference:
- Model: https://huggingface.co/naver/splade-v3
- Paper: SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer


@dataclass
class SparseEmbedding:
    """Sparse embedding representation."""

    indices: List[int]
    values: List[float]
    dimension: int

    def to_dict(self) -> Dict[int, float]:
        """Convert to dictionary format."""
        return {idx: val for idx, val in zip(self.indices, self.values)}

    def to_dense(self) -> torch.Tensor:
        """Convert to dense tensor."""
        dense = torch.zeros(self.dimension)
        for idx, val in zip(self.indices, self.values):
            dense[idx] = val
        return dense


class SPLADEv3:
    """
    Wrapper for SPLADE sparse encoder models.

    Supports:
    - prithivida/Splade_PP_en_v2 (public, recommended)
    - naver/splade-v3 (gated, requires access)
    - naver/splade-cocondenser-ensembledistil

    Features:
    - Sparse vector output (30,522 dimensions for BERT vocab)
    - Separate query and document encoding
    - Efficient sparse-to-sparse similarity

    Example:
        >>> model = SPLADEv3()
        >>> query_emb = model.encode_query("검색 엔진이란 무엇인가?")
        >>> doc_emb = model.encode_document("검색 엔진은 인터넷에서 정보를 찾는 도구입니다.")
        >>> score = model.similarity(query_emb, doc_emb)
    """

    def __init__(
        self,
        model_name: str = "naver/splade-v3",
        device: Optional[str] = None,
        max_length: int = 256,
    ):
        """
        Initialize SPLADE-v3 model.

        Args:
            model_name: Hugging Face model name
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length
        """
        from sentence_transformers import SparseEncoder

        self.model_name = model_name
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load model
        print(f"Loading SPLADE-v3 from {model_name}...")
        self.model = SparseEncoder(model_name)
        print(f"Model loaded on {device}")

        # Load tokenizer for token decoding
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased"  # SPLADE-v3 uses BERT tokenizer
        )
        self.vocab_size = self.tokenizer.vocab_size

    def encode_query(
        self,
        queries: Union[str, List[str]],
        return_sparse: bool = True,
    ) -> Union[torch.Tensor, List[SparseEmbedding]]:
        """
        Encode queries to sparse vectors.

        Args:
            queries: Single query or list of queries
            return_sparse: If True, return SparseEmbedding objects

        Returns:
            Sparse embeddings (tensor or list of SparseEmbedding)
        """
        if isinstance(queries, str):
            queries = [queries]

        embeddings = self.model.encode_query(queries)

        if return_sparse:
            return self._to_sparse_embeddings(embeddings)
        return embeddings

    def encode_document(
        self,
        documents: Union[str, List[str]],
        return_sparse: bool = True,
    ) -> Union[torch.Tensor, List[SparseEmbedding]]:
        """
        Encode documents to sparse vectors.

        Args:
            documents: Single document or list of documents
            return_sparse: If True, return SparseEmbedding objects

        Returns:
            Sparse embeddings (tensor or list of SparseEmbedding)
        """
        if isinstance(documents, str):
            documents = [documents]

        embeddings = self.model.encode_document(documents)

        if return_sparse:
            return self._to_sparse_embeddings(embeddings)
        return embeddings

    def _to_sparse_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> List[SparseEmbedding]:
        """Convert tensor to list of SparseEmbedding objects."""
        result = []
        for emb in embeddings:
            # Convert sparse tensor to dense CPU tensor
            emb = self._ensure_dense_cpu(emb)

            nonzero = emb.nonzero().squeeze(-1)
            indices = nonzero.tolist()
            values = emb[nonzero].tolist()
            result.append(
                SparseEmbedding(
                    indices=indices,
                    values=values,
                    dimension=emb.shape[0],
                )
            )
        return result

    def similarity(
        self,
        query_emb: Union[torch.Tensor, List[SparseEmbedding]],
        doc_emb: Union[torch.Tensor, List[SparseEmbedding]],
    ) -> torch.Tensor:
        """
        Compute similarity between query and document embeddings.

        Args:
            query_emb: Query embeddings
            doc_emb: Document embeddings

        Returns:
            Similarity scores [num_queries, num_docs]
        """
        # Convert SparseEmbedding to tensor if needed
        if isinstance(query_emb, list) and isinstance(query_emb[0], SparseEmbedding):
            query_emb = torch.stack([e.to_dense() for e in query_emb])
        if isinstance(doc_emb, list) and isinstance(doc_emb[0], SparseEmbedding):
            doc_emb = torch.stack([e.to_dense() for e in doc_emb])

        # Convert sparse tensors to dense (handles SparseCUDA format)
        if query_emb.is_sparse or query_emb.layout != torch.strided:
            query_emb = query_emb.to_dense()
        if doc_emb.is_sparse or doc_emb.layout != torch.strided:
            doc_emb = doc_emb.to_dense()

        return self.model.similarity(query_emb, doc_emb)

    def _ensure_dense_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to dense CPU tensor safely."""
        # Check for sparse tensor (covers both is_sparse and sparse layouts)
        if tensor.is_sparse or tensor.layout != torch.strided:
            tensor = tensor.to_dense()
        return tensor.cpu()

    def get_top_tokens(
        self,
        embedding: Union[torch.Tensor, SparseEmbedding],
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Get top-k tokens with highest weights.

        Args:
            embedding: Sparse embedding (tensor or SparseEmbedding)
            top_k: Number of top tokens to return

        Returns:
            List of (token, weight) tuples sorted by weight
        """
        if isinstance(embedding, SparseEmbedding):
            dense = embedding.to_dense()
        else:
            dense = self._ensure_dense_cpu(embedding)

        # Get top-k
        values, indices = torch.topk(dense, k=min(top_k, dense.shape[0]))

        result = []
        for idx, val in zip(indices.tolist(), values.tolist()):
            if val > 0:
                token = self.tokenizer.decode([idx]).strip()
                result.append((token, val))

        return result

    def analyze_text(
        self,
        text: str,
        is_query: bool = True,
        top_k: int = 20,
    ) -> Dict:
        """
        Analyze a text and return detailed sparse representation.

        Args:
            text: Input text
            is_query: If True, encode as query; otherwise as document
            top_k: Number of top tokens to show

        Returns:
            Dictionary with analysis results
        """
        if is_query:
            emb = self.encode_query(text, return_sparse=False)
        else:
            emb = self.encode_document(text, return_sparse=False)

        emb = emb[0]  # Get first (only) embedding

        # Convert sparse tensor to dense CPU tensor
        emb = self._ensure_dense_cpu(emb)

        # Get top tokens
        top_tokens = self.get_top_tokens(emb, top_k)

        # Count non-zero
        nonzero_count = (emb > 0).sum().item()

        return {
            "text": text,
            "type": "query" if is_query else "document",
            "nonzero_count": nonzero_count,
            "sparsity": 1.0 - (nonzero_count / emb.shape[0]),
            "top_tokens": top_tokens,
            "max_weight": emb.max().item(),
            "mean_weight": emb[emb > 0].mean().item() if nonzero_count > 0 else 0,
        }


def load_splade_v3(
    model_name: str = "naver/splade-v3",
    device: Optional[str] = None,
) -> SPLADEv3:
    """
    Load SPLADE-v3 model.

    Args:
        model_name: Hugging Face model name
        device: Device to use

    Returns:
        SPLADEv3 instance
    """
    return SPLADEv3(model_name=model_name, device=device)
