"""Dataset loader for paired training data."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PairedDataset(Dataset):
    """
    Dataset for (Query, Document) paired data.

    Loads data from JSONL files and tokenizes on-the-fly.
    """

    def __init__(
        self,
        data_files: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        num_negatives: int = 1,
        negative_sampling: str = "random",  # "random" or "hard"
    ):
        """
        Initialize dataset.

        Args:
            data_files: List of JSONL file paths
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            num_negatives: Number of negative samples per query
            negative_sampling: Negative sampling strategy
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.negative_sampling = negative_sampling

        # Load data
        self.data = []
        for file_path in data_files:
            self.data.extend(self._load_jsonl(file_path))

        print(f"✓ Loaded {len(self.data):,} training pairs")

    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training example."""
        item = self.data[idx]

        # Query
        query = item['query']
        query_encoded = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Positive document - support multiple formats
        if 'document' in item:
            pos_doc = item['document']
        elif 'positive_doc' in item:
            pos_doc = item['positive_doc']
        elif 'pos' in item:
            pos_doc = item['pos']
        else:
            raise KeyError(f"No document field found in item at index {idx}. Available keys: {list(item.keys())}")

        pos_doc_encoded = self.tokenizer(
            pos_doc,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Negative documents (sample from other documents)
        negative_docs = self._sample_negatives(idx)
        neg_docs_encoded = self.tokenizer(
            negative_docs,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'query_input_ids': query_encoded['input_ids'].squeeze(0),
            'query_attention_mask': query_encoded['attention_mask'].squeeze(0),
            'pos_doc_input_ids': pos_doc_encoded['input_ids'].squeeze(0),
            'pos_doc_attention_mask': pos_doc_encoded['attention_mask'].squeeze(0),
            'neg_doc_input_ids': neg_docs_encoded['input_ids'],  # [num_neg, seq_len]
            'neg_doc_attention_mask': neg_docs_encoded['attention_mask'],
        }

    def _sample_negatives(self, exclude_idx: int) -> List[str]:
        """Sample negative documents."""
        negatives = []

        # Random sampling from other documents
        indices = list(range(len(self.data)))
        indices.remove(exclude_idx)

        sampled_indices = random.sample(indices, min(self.num_negatives, len(indices)))

        for idx in sampled_indices:
            # Support multiple data formats
            item = self.data[idx]
            if 'document' in item:
                negatives.append(item['document'])
            elif 'positive_doc' in item:
                negatives.append(item['positive_doc'])
            elif 'pos' in item:
                negatives.append(item['pos'])
            else:
                raise KeyError(f"No document field found in item at index {idx}. Available keys: {list(item.keys())}")

        return negatives


class HardNegativesDataset(Dataset):
    """
    Dataset with pre-mined hard negatives.

    Supports two formats:
    1. {"query": "...", "positive_doc": "...", "hard_negatives": [...]}
    2. {"query": "...", "pos": "...", "negs": [...]}  # GitHub sample format
    """

    def __init__(
        self,
        data_files: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
    ):
        """
        Initialize dataset with hard negatives.

        Args:
            data_files: List of JSONL files with hard negatives
            tokenizer: Tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.data = []
        for file_path in data_files:
            self.data.extend(self._load_jsonl(file_path))

        print(f"✓ Loaded {len(self.data):,} samples with hard negatives")

    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training example with hard negatives."""
        item = self.data[idx]

        # Query
        query = item['query']
        query_encoded = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Positive document (support both formats)
        if 'pos' in item:
            pos_doc = item['pos']  # GitHub sample format
        elif 'positive_doc' in item:
            pos_doc = item['positive_doc']  # Our format
        else:
            raise KeyError(f"Missing positive document field in item: {item.keys()}")

        pos_doc_encoded = self.tokenizer(
            pos_doc,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Hard negatives (support both formats)
        if 'negs' in item:
            hard_negatives = item['negs']  # GitHub sample format
        elif 'hard_negatives' in item:
            hard_negatives = item['hard_negatives']  # Our format
        else:
            raise KeyError(f"Missing negatives field in item: {item.keys()}")

        neg_docs_encoded = self.tokenizer(
            hard_negatives,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'query_input_ids': query_encoded['input_ids'].squeeze(0),
            'query_attention_mask': query_encoded['attention_mask'].squeeze(0),
            'pos_doc_input_ids': pos_doc_encoded['input_ids'].squeeze(0),
            'pos_doc_attention_mask': pos_doc_encoded['attention_mask'].squeeze(0),
            'neg_doc_input_ids': neg_docs_encoded['input_ids'],
            'neg_doc_attention_mask': neg_docs_encoded['attention_mask'],
        }


def create_dataloaders(
    train_files: List[str],
    val_files: Optional[List[str]],
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_length: int = 256,
    num_negatives: int = 7,
    num_workers: int = 4,
    use_hard_negatives: bool = False,
) -> tuple:
    """
    Create train and validation dataloaders.

    Args:
        train_files: Training data files
        val_files: Validation data files
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_negatives: Number of negative samples
        num_workers: Number of dataloader workers
        use_hard_negatives: Use hard negatives dataset

    Returns:
        (train_loader, val_loader)
    """
    # Choose dataset class
    DatasetClass = HardNegativesDataset if use_hard_negatives else PairedDataset

    # Training dataset
    if use_hard_negatives:
        train_dataset = DatasetClass(train_files, tokenizer, max_length)
    else:
        train_dataset = DatasetClass(
            train_files, tokenizer, max_length, num_negatives
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Validation dataset
    val_loader = None
    if val_files:
        if use_hard_negatives:
            val_dataset = DatasetClass(val_files, tokenizer, max_length)
        else:
            val_dataset = DatasetClass(
                val_files, tokenizer, max_length, num_negatives
            )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader
