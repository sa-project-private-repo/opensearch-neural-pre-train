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
        use_in_batch_negatives: bool = True,
    ):
        """
        Initialize collator.

        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            use_in_batch_negatives: Whether to use in-batch negatives
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_in_batch_negatives = use_in_batch_negatives

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of triplets.

        Args:
            batch: List of triplet dictionaries

        Returns:
            Dictionary with tokenized tensors
        """
        queries = [item["query"] for item in batch]
        positives = [item["positive"] for item in batch]

        # Handle None negatives
        negatives = []
        for item in batch:
            neg = item.get("negative")
            if neg is None:
                # Use positive as fallback (will be filtered by loss function)
                neg = item["positive"]
            negatives.append(neg)

        # Tokenize all texts
        query_encoding = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        positive_encoding = self.tokenizer(
            positives,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        negative_encoding = self.tokenizer(
            negatives,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        result = {
            "query_input_ids": query_encoding["input_ids"],
            "query_attention_mask": query_encoding["attention_mask"],
            "positive_input_ids": positive_encoding["input_ids"],
            "positive_attention_mask": positive_encoding["attention_mask"],
            "negative_input_ids": negative_encoding["input_ids"],
            "negative_attention_mask": negative_encoding["attention_mask"],
        }

        # Add metadata
        if "pair_type" in batch[0]:
            result["pair_types"] = [item.get("pair_type", "unknown") for item in batch]

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
        max_length: Maximum sequence length
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
