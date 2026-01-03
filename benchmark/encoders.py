"""
Encoders for benchmark: dense (bge-m3) and sparse (v21.4).
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer

from benchmark.config import BenchmarkConfig

logger = logging.getLogger(__name__)


class BgeM3Encoder:
    """Dense encoder using bge-m3 model."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
    ):
        """Initialize bge-m3 encoder."""
        self.device = device
        logger.info(f"Loading bge-m3 model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"bge-m3 loaded, dimension: {self.dimension}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to dense vectors.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            normalize: Whether to L2 normalize vectors

        Returns:
            Dense embeddings as numpy array
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings

    def encode_single(self, text: str) -> List[float]:
        """Encode single text and return as list."""
        embedding = self.encode(text, batch_size=1, normalize=True)
        return embedding[0].tolist()


class NeuralSparseEncoder:
    """Sparse encoder using trained v21.4 model."""

    def __init__(
        self,
        model_path: Union[str, Path] = "huggingface/v21.4",
        device: str = "cuda",
        max_length: int = 64,
    ):
        """
        Initialize neural sparse encoder.

        Args:
            model_path: Path to trained model directory
            device: Device to run model on
            max_length: Maximum sequence length
        """
        self.device = device
        self.max_length = max_length
        model_path = Path(model_path)

        logger.info(f"Loading neural sparse model from: {model_path}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

        self.vocab_size = self.tokenizer.vocab_size
        self.relu = nn.ReLU()

        # Special token IDs to exclude
        self.special_token_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
        }
        # Remove None values
        self.special_token_ids = {
            tid for tid in self.special_token_ids if tid is not None
        }

        # Pre-build token lookup table for fast decoding
        self._token_lookup = self.tokenizer.convert_ids_to_tokens(
            list(range(self.vocab_size))
        )

        logger.info(
            f"Neural sparse model loaded, vocab_size: {self.vocab_size}"
        )
        logger.info(f"Built token lookup table with {len(self._token_lookup)} entries")

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """
        Encode texts to sparse vectors.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            top_k: Only keep top-k activations (None for all)

        Returns:
            List of sparse vectors as {token: weight} dicts
        """
        if isinstance(texts, str):
            texts = [texts]

        all_sparse_vectors = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            outputs = self.model(**inputs)
            logits = outputs.logits

            # Apply SPLADE transformation: log(1 + ReLU(logits))
            token_scores = torch.log1p(self.relu(logits))

            # Apply attention mask
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            masked_scores = token_scores * mask

            # Max pooling over sequence dimension
            sparse_repr, _ = masked_scores.max(dim=1)

            # Convert to sparse dictionaries
            for j in range(sparse_repr.size(0)):
                vec = sparse_repr[j]
                nonzero_indices = vec.nonzero(as_tuple=True)[0]

                sparse_dict = {}
                for idx in nonzero_indices.tolist():
                    if idx in self.special_token_ids:
                        continue
                    weight = vec[idx].item()
                    if weight > 0:
                        token = self._token_lookup[idx]
                        # Skip special tokens (prefixed with [ or <)
                        if token and not token.startswith(("[", "<")):
                            sparse_dict[token] = weight

                # Apply top-k filtering if specified
                if top_k is not None and len(sparse_dict) > top_k:
                    sorted_items = sorted(
                        sparse_dict.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    sparse_dict = dict(sorted_items[:top_k])

                all_sparse_vectors.append(sparse_dict)

        return all_sparse_vectors

    def encode_single(self, text: str, top_k: Optional[int] = None) -> Dict[str, float]:
        """Encode single text and return sparse vector."""
        result = self.encode([text], batch_size=1, top_k=top_k)
        return result[0]

    def encode_for_query(
        self,
        text: str,
        top_k: int = 100,
    ) -> Dict[str, float]:
        """
        Encode query text for searching.

        Args:
            text: Query text
            top_k: Number of top activations to keep

        Returns:
            Sparse vector as {token: weight} dict
        """
        return self.encode_single(text, top_k=top_k)


def create_encoders(config: BenchmarkConfig) -> tuple:
    """
    Create both encoders from config.

    Returns:
        Tuple of (dense_encoder, sparse_encoder)
    """
    dense_encoder = BgeM3Encoder(
        model_name=config.bge_m3_model,
        device=config.device,
    )
    sparse_encoder = NeuralSparseEncoder(
        model_path=config.neural_sparse_path,
        device=config.device,
    )
    return dense_encoder, sparse_encoder
