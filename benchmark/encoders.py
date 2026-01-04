"""
Encoders for benchmark: dense (bge-m3) and sparse (v21.4).
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import gc

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from benchmark.config import BenchmarkConfig
from benchmark.dataset import TextDataset

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

    def _create_collate_fn(self):
        """Create collate function for DataLoader."""

        def collate_fn(batch_texts: List[str]):
            return self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

        return collate_fn

    @torch.no_grad()
    def _encode_batch(
        self,
        inputs: Dict[str, torch.Tensor],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """
        Encode a single batch to sparse vectors.

        Args:
            inputs: Tokenized inputs on device
            top_k: Only keep top-k activations

        Returns:
            List of sparse vectors for this batch
        """
        outputs = self.model(**inputs)
        logits = outputs.logits

        # Apply SPLADE transformation: log(1 + ReLU(logits))
        token_scores = torch.log1p(self.relu(logits))

        # Apply attention mask
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        masked_scores = token_scores * mask

        # Max pooling over sequence dimension
        sparse_repr, _ = masked_scores.max(dim=1)

        # Convert to sparse dictionaries using pre-built lookup
        batch_vectors = []
        for j in range(sparse_repr.size(0)):
            vec = sparse_repr[j].cpu()  # Move to CPU for dict building
            nonzero_mask = vec > 0
            nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]

            sparse_dict = {}
            for idx in nonzero_indices.tolist():
                if idx in self.special_token_ids:
                    continue
                weight = vec[idx].item()
                token = self._token_lookup[idx]
                # Skip special tokens (prefixed with [ or <)
                if token and not token.startswith(("[", "<")):
                    sparse_dict[token] = weight

            if top_k is not None and len(sparse_dict) > top_k:
                sorted_items = sorted(
                    sparse_dict.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                sparse_dict = dict(sorted_items[:top_k])

            batch_vectors.append(sparse_dict)

        return batch_vectors

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        num_workers: int = 4,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """
        Encode texts to sparse vectors using parallel DataLoader.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            num_workers: Number of parallel workers for data loading
            top_k: Only keep top-k activations (None for all)

        Returns:
            List of sparse vectors as {token: weight} dicts
        """
        if isinstance(texts, str):
            texts = [texts]

        dataset = TextDataset(texts)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._create_collate_fn(),
            pin_memory=True if self.device != "cpu" else False,
        )

        all_sparse_vectors = []
        show_progress = len(texts) > 100

        for batch_idx, inputs in enumerate(
            tqdm(dataloader, desc="Sparse encoding", disable=not show_progress)
        ):
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Encode batch
            batch_vectors = self._encode_batch(inputs, top_k)
            all_sparse_vectors.extend(batch_vectors)

            # Periodic memory cleanup
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        return all_sparse_vectors

    def encode_single(self, text: str, top_k: Optional[int] = None) -> Dict[str, float]:
        """Encode single text and return sparse vector."""
        result = self.encode([text], batch_size=1, top_k=top_k)
        return result[0]

    @torch.no_grad()
    def encode_for_query(
        self,
        text: str,
        top_k: int = 100,
    ) -> Dict[str, float]:
        """
        Encode query text for searching (optimized for single query).

        This method bypasses DataLoader overhead for fast single-query encoding.

        Args:
            text: Query text
            top_k: Number of top activations to keep

        Returns:
            Sparse vector as {token: weight} dict
        """
        # Direct tokenization without DataLoader
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Encode and return
        return self._encode_batch(inputs, top_k)[0]


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
