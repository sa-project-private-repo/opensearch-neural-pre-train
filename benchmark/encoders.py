"""
Encoders for benchmark: dense (bge-m3) and sparse (V33 SPLADEModernBERT).
"""
import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from benchmark.config import BenchmarkConfig
from benchmark.dataset import TextDataset
from src.model.splade_modern import SPLADEModernBERT

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
            numpy array of shape (n_texts, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=normalize,
        )
        return embeddings


class NeuralSparseEncoder:
    """Sparse encoder using HuggingFace AutoModelForMaskedLM format."""

    def __init__(
        self,
        model_path: Union[str, Path] = "huggingface/v33",
        device: str = "cuda",
        max_length: int = 64,
        query_max_length: Optional[int] = None,
        doc_max_length: Optional[int] = None,
    ):
        """
        Initialize neural sparse encoder.

        Args:
            model_path: Path to trained model directory
            device: Device to run model on
            max_length: Maximum sequence length (fallback)
            query_max_length: Max length for queries (None=max_length)
            doc_max_length: Max length for documents (None=max_length)
        """
        self.device = device
        self.max_length = max_length
        self.query_max_length = query_max_length or max_length
        self.doc_max_length = doc_max_length or max_length
        model_path = Path(model_path)

        logger.info(f"Loading neural sparse model from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

        self.vocab_size = self.tokenizer.vocab_size
        self.relu = nn.ReLU()

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

        self._token_lookup = self.tokenizer.convert_ids_to_tokens(
            list(range(self.vocab_size))
        )

        logger.info(
            f"Neural sparse model loaded, vocab_size: {self.vocab_size}"
        )

    def _create_collate_fn(self, max_length: Optional[int] = None):
        """Create collate function for DataLoader."""
        effective_length = max_length or self.doc_max_length

        def collate_fn(batch_texts: List[str]):
            return self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=effective_length,
            )

        return collate_fn

    @torch.no_grad()
    def _encode_batch(
        self,
        inputs: Dict[str, torch.Tensor],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Encode a single batch to sparse vectors."""
        outputs = self.model(**inputs)
        logits = outputs.logits

        token_scores = torch.log1p(self.relu(logits))
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        masked_scores = token_scores * mask
        sparse_repr, _ = masked_scores.max(dim=1)

        batch_vectors = []
        for j in range(sparse_repr.size(0)):
            vec = sparse_repr[j].cpu()
            nonzero_mask = vec > 0
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
        """Encode texts to sparse vectors."""
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
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            batch_vectors = self._encode_batch(inputs, top_k)
            all_sparse_vectors.extend(batch_vectors)

            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        return all_sparse_vectors

    def encode_single(
        self, text: str, top_k: Optional[int] = None
    ) -> Dict[str, float]:
        """Encode single text and return sparse vector."""
        result = self.encode([text], batch_size=1, top_k=top_k)
        return result[0]

    @torch.no_grad()
    def encode_for_query(
        self,
        text: str,
        top_k: int = 100,
    ) -> Dict[str, float]:
        """Encode query text for searching."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self._encode_batch(inputs, top_k)[0]


class NeuralSparseEncoderV33:
    """V33 Sparse encoder using SPLADEModernBERT (skt/A.X-Encoder-base, 50K vocab)."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "outputs/train_v33/final_model/model.pt",
        device: str = "cuda",
        query_max_length: int = 64,
        doc_max_length: int = 256,
    ):
        self.device = device
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        checkpoint_path = Path(checkpoint_path)

        logger.info(f"Loading V33 neural sparse model from: {checkpoint_path}")

        self.tokenizer = AutoTokenizer.from_pretrained("skt/A.X-Encoder-base")

        self.model = SPLADEModernBERT(model_name="skt/A.X-Encoder-base")

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        self.vocab_size = self.tokenizer.vocab_size

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

        self._token_lookup = self.tokenizer.convert_ids_to_tokens(
            list(range(self.vocab_size))
        )

        logger.info(f"V33 neural sparse model loaded, vocab_size: {self.vocab_size}")

    def _create_collate_fn(self, max_length: Optional[int] = None):
        effective_length = max_length or self.doc_max_length

        def collate_fn(batch_texts: List[str]):
            return self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=effective_length,
            )

        return collate_fn

    @torch.no_grad()
    def _encode_batch(
        self,
        inputs: Dict[str, torch.Tensor],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        sparse_repr, _ = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        batch_vectors = []
        for j in range(sparse_repr.size(0)):
            vec = sparse_repr[j].cpu()
            nonzero_mask = vec > 0
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
            tqdm(dataloader, desc="V33 Sparse encoding", disable=not show_progress)
        ):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            batch_vectors = self._encode_batch(inputs, top_k)
            all_sparse_vectors.extend(batch_vectors)

            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        return all_sparse_vectors

    def encode_single(self, text: str, top_k: Optional[int] = None) -> Dict[str, float]:
        result = self.encode([text], batch_size=1, top_k=top_k)
        return result[0]

    @torch.no_grad()
    def encode_for_query(
        self,
        text: str,
        top_k: int = 100,
    ) -> Dict[str, float]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self._encode_batch(inputs, top_k)[0]


def create_encoders_v33(
    config: BenchmarkConfig,
    checkpoint_path: str = "outputs/train_v33/final_model/model.pt",
) -> tuple:
    """Create V33 dense + sparse encoder pair."""
    dense_encoder = BgeM3Encoder(
        model_name=config.bge_m3_model,
        device=config.device,
    )
    q_len = getattr(config, "query_max_length", 64)
    d_len = getattr(config, "doc_max_length", 256)
    sparse_encoder = NeuralSparseEncoderV33(
        checkpoint_path=checkpoint_path,
        device=config.device,
        query_max_length=q_len,
        doc_max_length=d_len,
    )
    return dense_encoder, sparse_encoder
