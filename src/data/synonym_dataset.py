"""Cross-lingual synonym dataset for KD training."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class SynonymDataset(Dataset):
    """
    Dataset for cross-lingual synonym pairs.

    Loads Korean-English synonym pairs for cross-lingual alignment training.
    """

    def __init__(
        self,
        data_path: str,
        max_en_terms: int = 3,
    ):
        """
        Initialize synonym dataset.

        Args:
            data_path: Path to JSONL file with synonym pairs
            max_en_terms: Maximum English terms to sample per entry
        """
        self.data_path = Path(data_path)
        self.max_en_terms = max_en_terms
        self.entries: List[Dict] = []

        self._load_data()

    def _load_data(self) -> None:
        """Load synonym data from JSONL file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Synonym data not found: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    # Validate entry format
                    if self._validate_entry(entry):
                        self.entries.append(entry)

        print(f"Loaded {len(self.entries)} synonym pairs from {self.data_path}")

    def _validate_entry(self, entry: Dict) -> bool:
        """Validate entry has required fields."""
        # Format 1: ko_term + en_terms (list)
        if "ko_term" in entry and "en_terms" in entry:
            return bool(entry["ko_term"] and entry["en_terms"])
        # Format 2: ko_term + en_term (single) - large-scale dataset format
        if "ko_term" in entry and "en_term" in entry:
            return bool(entry["ko_term"] and entry["en_term"])
        # Format 3: ko + en_primary (generation format)
        if "ko" in entry and "en_primary" in entry:
            return bool(entry["ko"] and entry["en_primary"])
        return False

    def _normalize_entry(self, entry: Dict) -> Dict:
        """Normalize entry to standard format."""
        # Format 1: Already has ko_term + en_terms (list)
        if "ko_term" in entry and "en_terms" in entry:
            return entry
        # Format 2: ko_term + en_term (single) - large-scale dataset format
        if "ko_term" in entry and "en_term" in entry:
            return {
                "ko_term": entry["ko_term"],
                "en_terms": [entry["en_term"]],
                "category": entry.get("source", "unknown"),
            }
        # Format 3: Convert from generation format (ko + en_primary)
        en_terms = [entry["en_primary"]]
        if entry.get("en_alternatives"):
            en_terms.extend(entry["en_alternatives"])
        if entry.get("abbreviation"):
            en_terms.append(entry["abbreviation"])
        return {
            "ko_term": entry["ko"],
            "en_terms": [t for t in en_terms if t],
            "category": entry.get("category", "unknown"),
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        Get a synonym pair.

        Returns:
            Dictionary with:
                - ko_term: Korean term
                - en_term: Randomly sampled English term
                - category: Category if available
        """
        entry = self._normalize_entry(self.entries[idx])

        # Sample one English term
        en_terms = entry["en_terms"][: self.max_en_terms]
        en_term = random.choice(en_terms) if en_terms else ""

        return {
            "ko_term": entry["ko_term"],
            "en_term": en_term,
            "category": entry.get("category", "unknown"),
        }

    def get_all_pairs(self) -> List[Tuple[str, str]]:
        """Get all Korean-English pairs for evaluation."""
        pairs = []
        for entry in self.entries:
            entry = self._normalize_entry(entry)
            for en_term in entry["en_terms"]:
                pairs.append((entry["ko_term"], en_term))
        return pairs


class ParallelSentenceDataset(Dataset):
    """
    Dataset for Korean-English parallel sentences.

    Used for additional cross-lingual training signal.
    """

    def __init__(self, data_path: str):
        """
        Initialize parallel sentence dataset.

        Args:
            data_path: Path to JSONL file with parallel sentences
        """
        self.data_path = Path(data_path)
        self.entries: List[Dict] = []

        self._load_data()

    def _load_data(self) -> None:
        """Load parallel sentences from JSONL file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Parallel data not found: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if entry.get("ko") and entry.get("en"):
                        self.entries.append(entry)

        print(f"Loaded {len(self.entries)} parallel sentences from {self.data_path}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        Get a parallel sentence pair.

        Returns:
            Dictionary with:
                - ko: Korean sentence
                - en: English sentence
                - term_ko: Optional source term (Korean)
                - term_en: Optional source term (English)
        """
        entry = self.entries[idx]
        return {
            "ko": entry["ko"],
            "en": entry["en"],
            "term_ko": entry.get("term_ko", ""),
            "term_en": entry.get("term_en", ""),
        }


class SynonymCollator:
    """
    Collator for synonym batches.

    Tokenizes Korean and English terms for the sparse model.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 64,
    ):
        """
        Initialize collator.

        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate synonym pairs into batch.

        Args:
            features: List of synonym entries

        Returns:
            Batch dictionary with tokenized inputs
        """
        ko_terms = [f["ko_term"] for f in features]
        en_terms = [f["en_term"] for f in features]

        # Tokenize Korean terms
        ko_encoding = self.tokenizer(
            ko_terms,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize English terms
        en_encoding = self.tokenizer(
            en_terms,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "ko_input_ids": ko_encoding["input_ids"],
            "ko_attention_mask": ko_encoding["attention_mask"],
            "en_input_ids": en_encoding["input_ids"],
            "en_attention_mask": en_encoding["attention_mask"],
            "ko_terms": ko_terms,
            "en_terms": en_terms,
        }


class MixedBatchSampler:
    """
    Sampler that creates mixed batches with main data and synonym pairs.

    For each main batch, samples a proportion of synonym pairs to include.
    """

    def __init__(
        self,
        main_dataset_size: int,
        synonym_dataset_size: int,
        batch_size: int,
        synonym_ratio: float = 0.2,
    ):
        """
        Initialize mixed batch sampler.

        Args:
            main_dataset_size: Size of main training dataset
            synonym_dataset_size: Size of synonym dataset
            batch_size: Total batch size
            synonym_ratio: Fraction of batch from synonym dataset
        """
        self.main_size = main_dataset_size
        self.synonym_size = synonym_dataset_size
        self.batch_size = batch_size
        self.synonym_ratio = synonym_ratio

        # Calculate split
        self.synonym_per_batch = max(1, int(batch_size * synonym_ratio))
        self.main_per_batch = batch_size - self.synonym_per_batch

    def get_batch_indices(
        self,
        batch_idx: int,
    ) -> Tuple[List[int], List[int]]:
        """
        Get indices for a mixed batch.

        Args:
            batch_idx: Current batch index

        Returns:
            Tuple of (main_indices, synonym_indices)
        """
        # Sample main indices
        main_start = (batch_idx * self.main_per_batch) % self.main_size
        main_indices = [
            (main_start + i) % self.main_size for i in range(self.main_per_batch)
        ]

        # Sample synonym indices (random)
        synonym_indices = random.sample(
            range(self.synonym_size),
            min(self.synonym_per_batch, self.synonym_size),
        )

        return main_indices, synonym_indices


if __name__ == "__main__":
    # Test dataset loading
    print("Testing synonym dataset...")

    # Create test data
    test_data = [
        {"ko_term": "머신러닝", "en_terms": ["machine learning", "ML"], "category": "ML"},
        {"ko_term": "딥러닝", "en_terms": ["deep learning", "DL"], "category": "ML"},
        {"ko_term": "자연어처리", "en_terms": ["NLP", "natural language processing"], "category": "NLP"},
    ]

    # Save test data
    test_path = Path("/tmp/test_synonyms.jsonl")
    with open(test_path, "w", encoding="utf-8") as f:
        for entry in test_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Load and test
    dataset = SynonymDataset(str(test_path))
    print(f"Dataset size: {len(dataset)}")

    for i in range(len(dataset)):
        item = dataset[i]
        print(f"  {item['ko_term']} → {item['en_term']}")

    # Test get_all_pairs
    pairs = dataset.get_all_pairs()
    print(f"\nAll pairs ({len(pairs)}):")
    for ko, en in pairs:
        print(f"  {ko} → {en}")

    # Cleanup
    test_path.unlink()
    print("\nTest completed!")
