"""Training data builder for Neural Sparse pre-training."""

import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class NeuralSparseDataset(Dataset):
    """
    Dataset for Neural Sparse training.

    Combines query-document pairs with hard negatives and synonym pairs.
    """

    def __init__(
        self,
        qd_pairs: List[Dict],
        documents: Optional[Dict[str, str]] = None,
        synonyms: Optional[List[Dict]] = None,
        num_negatives: int = 10,
        synonym_sample_prob: float = 0.3,
    ):
        """
        Initialize dataset.

        Args:
            qd_pairs: List of query-document pairs
            documents: Dict mapping doc_id to document text
            synonyms: List of synonym dicts with 'korean' and 'english' keys
            num_negatives: Number of negative samples per query
            synonym_sample_prob: Probability of including synonym pair in batch
        """
        self.qd_pairs = qd_pairs
        self.documents = documents or {}
        self.synonyms = synonyms or []
        self.num_negatives = num_negatives
        self.synonym_sample_prob = synonym_sample_prob

        # Build document index if documents provided
        if documents:
            self.doc_ids = list(documents.keys())
        else:
            self.doc_ids = []

        print(f"Initialized NeuralSparseDataset:")
        print(f"  QD pairs: {len(self.qd_pairs)}")
        print(f"  Documents: {len(self.documents)}")
        print(f"  Synonyms: {len(self.synonyms)}")
        print(f"  Num negatives: {num_negatives}")

    def __len__(self) -> int:
        return len(self.qd_pairs)

    def __getitem__(self, idx: int) -> Dict:
        """Get training sample."""
        pair = self.qd_pairs[idx]

        query = pair["query"]
        pos_doc = pair["positive_doc"]

        # Get negative documents
        negative_docs = pair.get("negative_docs", [])

        # If not enough negatives, sample random documents
        if len(negative_docs) < self.num_negatives and self.doc_ids:
            additional_needed = self.num_negatives - len(negative_docs)
            random_neg = random.sample(self.doc_ids, min(additional_needed, len(self.doc_ids)))
            negative_docs.extend([self.documents[doc_id] for doc_id in random_neg])

        # Create sample
        sample = {
            "query": query,
            "positive_doc": pos_doc,
            "negative_docs": negative_docs[: self.num_negatives],
        }

        # Optionally add synonym pair
        if self.synonyms and random.random() < self.synonym_sample_prob:
            synonym = random.choice(self.synonyms)
            sample["korean_term"] = synonym["korean"]
            sample["english_term"] = synonym["english"]

        return sample


class TrainingDataBuilder:
    """Build training data from existing datasets."""

    def __init__(self):
        """Initialize data builder."""
        pass

    def load_qd_pairs(self, path: str) -> List[Dict]:
        """
        Load query-document pairs from pickle file.

        Args:
            path: Path to QD pairs pickle

        Returns:
            List of QD pair dicts
        """
        with open(path, "rb") as f:
            qd_pairs = pickle.load(f)

        print(f"Loaded {len(qd_pairs)} QD pairs from {path}")
        return qd_pairs

    def load_documents(self, path: str) -> Dict[str, str]:
        """
        Load documents from JSON file.

        Args:
            path: Path to documents JSON

        Returns:
            Dict mapping doc_id to text
        """
        with open(path, "r", encoding="utf-8") as f:
            documents = json.load(f)

        # Handle different formats
        if isinstance(documents, list):
            # List of dicts with 'id' and 'text'
            doc_dict = {}
            for doc in documents:
                if isinstance(doc, dict):
                    doc_id = doc.get("id", str(len(doc_dict)))
                    text = doc.get("text", doc.get("content", ""))
                    doc_dict[doc_id] = text
                else:
                    doc_dict[str(len(doc_dict))] = str(doc)
            documents = doc_dict

        print(f"Loaded {len(documents)} documents from {path}")
        return documents

    def load_synonyms(self, path: str) -> List[Dict]:
        """
        Load synonym pairs from JSON file.

        Args:
            path: Path to synonyms JSON

        Returns:
            List of synonym dicts
        """
        with open(path, "r", encoding="utf-8") as f:
            synonyms = json.load(f)

        # Ensure standard format
        if isinstance(synonyms, list):
            # Already in list format
            pass
        elif isinstance(synonyms, dict):
            # Convert dict format to list
            synonym_list = []
            for korean, english_list in synonyms.items():
                if isinstance(english_list, list):
                    for english in english_list:
                        synonym_list.append({"korean": korean, "english": english})
                else:
                    synonym_list.append({"korean": korean, "english": english_list})
            synonyms = synonym_list

        print(f"Loaded {len(synonyms)} synonym pairs from {path}")
        return synonyms

    def convert_qd_pairs_to_standard_format(
        self,
        qd_pairs: List,
    ) -> List[Dict]:
        """
        Convert QD pairs to standard format.

        Args:
            qd_pairs: Raw QD pairs (can be various formats)

        Returns:
            Standardized QD pairs
        """
        standardized = []

        for pair in qd_pairs:
            if isinstance(pair, dict):
                # Already dict format
                if "query" in pair and ("positive_doc" in pair or "document" in pair):
                    standardized.append(
                        {
                            "query": pair["query"],
                            "positive_doc": pair.get("positive_doc", pair.get("document", "")),
                            "negative_docs": pair.get("negative_docs", []),
                        }
                    )
            elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
                # (query, doc) tuple format
                standardized.append(
                    {
                        "query": pair[0],
                        "positive_doc": pair[1],
                        "negative_docs": [],
                    }
                )

        print(f"Converted {len(standardized)} QD pairs to standard format")
        return standardized

    def build_training_dataset(
        self,
        qd_pairs_path: str,
        documents_path: Optional[str] = None,
        synonyms_path: Optional[str] = None,
        num_negatives: int = 10,
        train_split: float = 0.9,
    ) -> Tuple[NeuralSparseDataset, NeuralSparseDataset]:
        """
        Build train and validation datasets.

        Args:
            qd_pairs_path: Path to QD pairs
            documents_path: Optional path to documents
            synonyms_path: Optional path to synonyms
            num_negatives: Number of negative samples
            train_split: Train/val split ratio

        Returns:
            (train_dataset, val_dataset)
        """
        # Load data
        qd_pairs = self.load_qd_pairs(qd_pairs_path)
        qd_pairs = self.convert_qd_pairs_to_standard_format(qd_pairs)

        documents = None
        if documents_path:
            documents = self.load_documents(documents_path)

        synonyms = None
        if synonyms_path:
            synonyms = self.load_synonyms(synonyms_path)

        # Split train/val
        random.shuffle(qd_pairs)
        split_idx = int(len(qd_pairs) * train_split)
        train_pairs = qd_pairs[:split_idx]
        val_pairs = qd_pairs[split_idx:]

        print(f"\nDataset split:")
        print(f"  Train: {len(train_pairs)} pairs")
        print(f"  Val: {len(val_pairs)} pairs")

        # Create datasets
        train_dataset = NeuralSparseDataset(
            qd_pairs=train_pairs,
            documents=documents,
            synonyms=synonyms,
            num_negatives=num_negatives,
            synonym_sample_prob=0.3,
        )

        val_dataset = NeuralSparseDataset(
            qd_pairs=val_pairs,
            documents=documents,
            synonyms=synonyms,
            num_negatives=num_negatives,
            synonym_sample_prob=0.0,  # No synonyms in validation
        )

        return train_dataset, val_dataset


if __name__ == "__main__":
    # Test training data builder
    print("Testing TrainingDataBuilder...")

    builder = TrainingDataBuilder()

    # Test paths
    qd_pairs_path = "dataset/base_model/qd_pairs_base.pkl"
    documents_path = "dataset/base_model/documents.json"
    synonyms_path = "dataset/synonyms/combined_synonyms.json"

    # Build datasets
    try:
        train_dataset, val_dataset = builder.build_training_dataset(
            qd_pairs_path=qd_pairs_path,
            documents_path=documents_path,
            synonyms_path=synonyms_path,
            num_negatives=5,
            train_split=0.9,
        )

        print(f"\nTrain dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")

        # Test getting a sample
        sample = train_dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Query: {sample['query'][:50]}...")
        print(f"Positive doc: {sample['positive_doc'][:50]}...")
        print(f"Num negatives: {len(sample['negative_docs'])}")
        if "korean_term" in sample:
            print(f"Synonym: {sample['korean_term']} -> {sample['english_term']}")

    except FileNotFoundError as e:
        print(f"\nNote: Test files not found: {e}")
        print("This is expected if running outside the project directory.")
