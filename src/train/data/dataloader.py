"""DataLoader utilities for SPLADE training."""

import logging
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class TripletCollator:
    """
    Collator for triplet batches.

    Tokenizes and pads query, positive, and negative texts.
    Supports in-batch negatives where negatives from other samples
    in the batch are used as additional negatives.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        query_max_length: Optional[int] = None,
        doc_max_length: Optional[int] = None,
        use_in_batch_negatives: bool = True,
    ):
        """
        Initialize collator.

        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length (fallback)
            query_max_length: Max length for queries (None=max_length)
            doc_max_length: Max length for documents (None=max_length)
            use_in_batch_negatives: Whether to use in-batch negatives
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.query_max_length = query_max_length or max_length
        self.doc_max_length = doc_max_length or max_length
        self.use_in_batch_negatives = use_in_batch_negatives

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of triplets.

        Supports both single-negative and multi-negative formats:
        - Single: {"query": ..., "positive": ..., "negative": ...}
        - Multi:  {"query": ..., "positive": ..., "negatives": [...]}

        For multi-negatives, all negatives are flattened to [batch*k, seq]
        for efficient tokenization. The `num_negatives` field indicates k.

        Args:
            batch: List of triplet dictionaries

        Returns:
            Dictionary with tokenized tensors
        """
        queries = [item["query"] for item in batch]
        positives = [item["positive"] for item in batch]

        # Detect multi-negative format
        has_multi_neg = (
            "negatives" in batch[0]
            and isinstance(batch[0]["negatives"], list)
        )

        if has_multi_neg:
            # Multi-hard-negatives: flatten to [batch*k] texts
            num_negatives = len(batch[0]["negatives"])
            all_negatives = []
            for item in batch:
                negs = item.get("negatives", [])
                # Pad to num_negatives if short
                while len(negs) < num_negatives:
                    negs.append(
                        negs[-1] if negs else item["positive"]
                    )
                all_negatives.extend(negs[:num_negatives])
        else:
            # Single negative
            num_negatives = 1
            all_negatives = []
            for item in batch:
                neg = item.get("negative")
                if neg is None:
                    neg = item["positive"]
                all_negatives.append(neg)

        # Tokenize all texts (asymmetric lengths)
        query_encoding = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
            return_tensors="pt",
        )

        positive_encoding = self.tokenizer(
            positives,
            padding=True,
            truncation=True,
            max_length=self.doc_max_length,
            return_tensors="pt",
        )

        # [batch*k, seq_len] for multi-neg, [batch, seq_len] for single
        negative_encoding = self.tokenizer(
            all_negatives,
            padding=True,
            truncation=True,
            max_length=self.doc_max_length,
            return_tensors="pt",
        )

        result = {
            "query_input_ids": query_encoding["input_ids"],
            "query_attention_mask": query_encoding["attention_mask"],
            "positive_input_ids": positive_encoding["input_ids"],
            "positive_attention_mask": positive_encoding["attention_mask"],
            "negative_input_ids": negative_encoding["input_ids"],
            "negative_attention_mask": negative_encoding["attention_mask"],
            "num_negatives": num_negatives,
            # Raw texts for teacher model KD
            "query_texts": queries,
            "positive_texts": positives,
        }

        # Teacher scores
        if "teacher_pos_score" in batch[0]:
            result["teacher_pos_scores"] = torch.tensor(
                [item["teacher_pos_score"] for item in batch],
                dtype=torch.float32,
            )

        if has_multi_neg and "teacher_neg_scores" in batch[0]:
            # Multi-neg teacher scores: [batch, k]
            result["teacher_neg_scores"] = torch.tensor(
                [item["teacher_neg_scores"] for item in batch],
                dtype=torch.float32,
            )
        elif "teacher_neg_score" in batch[0]:
            # Single neg teacher score: [batch]
            result["teacher_neg_scores"] = torch.tensor(
                [item.get("teacher_neg_score", 0.0) for item in batch],
                dtype=torch.float32,
            )

        # Add metadata
        if "pair_type" in batch[0]:
            result["pair_types"] = [
                item.get("pair_type", "unknown") for item in batch
            ]

        if "difficulty" in batch[0]:
            result["difficulties"] = [
                item.get("difficulty", "medium") for item in batch
            ]

        return result


def create_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 32,
    max_length: int = 256,
    query_max_length: Optional[int] = None,
    doc_max_length: Optional[int] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    use_in_batch_negatives: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for training.

    Args:
        dataset: Dataset to load from
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size (per GPU if distributed)
        max_length: Maximum sequence length (fallback)
        query_max_length: Max length for queries (None=max_length)
        doc_max_length: Max length for documents (None=max_length)
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        use_in_batch_negatives: Whether to use in-batch negatives
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop the last incomplete batch
        distributed: Use DistributedSampler for DDP
        world_size: Number of processes (DDP)
        rank: Current process rank (DDP)

    Returns:
        DataLoader instance
    """
    collator = TripletCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        query_max_length=query_max_length,
        doc_max_length=doc_max_length,
        use_in_batch_negatives=use_in_batch_negatives,
    )

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False  # sampler handles shuffling

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
    )

    mode = "distributed" if distributed else "standard"
    logger.info(
        f"Created DataLoader ({mode}): batch_size={batch_size}, "
        f"num_batches={len(dataloader)}, shuffle={shuffle}"
    )

    return dataloader
