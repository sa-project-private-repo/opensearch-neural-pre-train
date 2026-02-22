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
from src.model.splade_xlmr import SPLADEDocContextGated

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


class NeuralSparseEncoderV28:
    """
    V28 Sparse encoder using SPLADEDocContextGated model.

    V28 uses Context Gate architecture that requires loading from
    PyTorch checkpoint (not HuggingFace format) to preserve
    the context gate weights.

    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "outputs/train_v28_ddp/checkpoint_epoch25_step41850/model.pt",
        device: str = "cuda",
        max_length: int = 192,
    ):
        """
        Initialize V28 neural sparse encoder.

        Args:
            checkpoint_path: Path to PyTorch checkpoint file
            device: Device to run model on
            max_length: Maximum sequence length
        """
        self.device = device
        self.max_length = max_length
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading V28 neural sparse model from: {checkpoint_path}")

        # Load tokenizer from base model
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

        # Create SPLADEDocContextGated model
        self.model = SPLADEDocContextGated(
            model_name="xlm-roberta-base",
            use_mlm_head=True,
            gate_hidden=256,
            gate_heads=4,
        )

        # Load checkpoint weights
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        self.vocab_size = self.tokenizer.vocab_size

        # Special token IDs to exclude
        self.special_token_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
        }
        self.special_token_ids = {
            tid for tid in self.special_token_ids if tid is not None
        }

        # Pre-build token lookup table
        self._token_lookup = self.tokenizer.convert_ids_to_tokens(
            list(range(self.vocab_size))
        )

        logger.info(
            f"V28 neural sparse model loaded with Context Gate, vocab_size: {self.vocab_size}"
        )

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
        Encode a single batch to sparse vectors using Context Gate.

        Args:
            inputs: Tokenized inputs on device
            top_k: Only keep top-k activations

        Returns:
            List of sparse vectors for this batch
        """
        # V28 model forward pass includes Context Gate
        sparse_repr, _ = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # Convert to sparse dictionaries
        batch_vectors = []
        for j in range(sparse_repr.size(0)):
            vec = sparse_repr[j].cpu()
            # Min threshold: OpenSearch requires positive normal
            # floats (>= 1.17549435e-38). Use 1e-3 to also
            # filter noise from context gate.
            nonzero_mask = vec > 1e-3
            nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]

            sparse_dict = {}
            for idx in nonzero_indices.tolist():
                if idx in self.special_token_ids:
                    continue
                weight = vec[idx].item()
                token = self._token_lookup[idx]
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
            tqdm(dataloader, desc="V28 Sparse encoding", disable=not show_progress)
        ):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            batch_vectors = self._encode_batch(inputs, top_k)
            all_sparse_vectors.extend(batch_vectors)

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
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self._encode_batch(inputs, top_k)[0]


def create_encoders(config: BenchmarkConfig) -> tuple:
    """
    Create both encoders from config.

    Auto-detects V28 context gate checkpoint and uses
    NeuralSparseEncoderV28 when available.

    Returns:
        Tuple of (dense_encoder, sparse_encoder)
    """
    dense_encoder = BgeM3Encoder(
        model_name=config.bge_m3_model,
        device=config.device,
    )
    max_length = getattr(config, "neural_sparse_max_length", 192)

    # Check for V28 context gate checkpoint
    context_gate_path = Path(config.neural_sparse_path) / "context_gate.pt"
    if context_gate_path.exists():
        # V28: load full model with context gate from DDP checkpoint
        checkpoint_path = getattr(
            config,
            "v28_checkpoint_path",
            "outputs/train_v28_ddp/checkpoint_epoch25_step41850/model.pt",
        )
        logger.info(
            f"Context gate detected, using NeuralSparseEncoderV28"
        )
        sparse_encoder = NeuralSparseEncoderV28(
            checkpoint_path=checkpoint_path,
            device=config.device,
            max_length=max_length,
        )
    else:
        sparse_encoder = NeuralSparseEncoder(
            model_path=config.neural_sparse_path,
            device=config.device,
            max_length=max_length,
        )
    return dense_encoder, sparse_encoder


def create_encoders_v28(
    config: BenchmarkConfig,
    checkpoint_path: str = "outputs/train_v28/checkpoint_epoch25_step280825/model.pt",
) -> tuple:
    """
    Create encoders with V28 sparse encoder.

    V28 uses SPLADEDocContextGated architecture that requires loading
    from PyTorch checkpoint to preserve Context Gate weights.

    Args:
        config: Benchmark configuration
        checkpoint_path: Path to V28 PyTorch checkpoint

    Returns:
        Tuple of (dense_encoder, sparse_encoder_v28)
    """
    dense_encoder = BgeM3Encoder(
        model_name=config.bge_m3_model,
        device=config.device,
    )
    max_length = getattr(config, "neural_sparse_max_length", 192)
    sparse_encoder = NeuralSparseEncoderV28(
        checkpoint_path=checkpoint_path,
        device=config.device,
        max_length=max_length,
    )
    return dense_encoder, sparse_encoder
