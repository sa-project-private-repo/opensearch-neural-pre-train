"""Data collator for Neural Sparse training."""

from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizer


class NeuralSparseDataCollator:
    """
    Data collator for Neural Sparse training.

    Handles batching of query-document pairs with hard negatives
    and optional cross-lingual synonym pairs.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        query_max_length: int = 64,
        doc_max_length: int = 256,
        num_negatives: int = 10,
    ):
        """
        Initialize data collator.

        Args:
            tokenizer: HuggingFace tokenizer
            query_max_length: Maximum length for queries
            doc_max_length: Maximum length for documents
            num_negatives: Number of negative samples per query
        """
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        self.num_negatives = num_negatives

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of features.

        Args:
            features: List of feature dicts containing:
                - query: Query text
                - positive_doc: Positive document text
                - negative_docs: List of negative document texts
                - korean_term (optional): Korean synonym
                - english_term (optional): English synonym

        Returns:
            Batch dict with tokenized inputs AND raw text for teacher models
        """
        batch = {}

        # Extract queries, positive docs, and negative docs
        queries = [f["query"] for f in features]
        pos_docs = [f["positive_doc"] for f in features]

        # Store raw text for teacher model (knowledge distillation)
        batch["queries"] = queries
        batch["positive_docs"] = pos_docs
        batch["negative_docs"] = [f["negative_docs"] for f in features]

        # Tokenize queries
        query_encoded = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
            return_tensors="pt",
        )
        batch["query_input_ids"] = query_encoded["input_ids"]
        batch["query_attention_mask"] = query_encoded["attention_mask"]

        # Tokenize positive documents
        pos_doc_encoded = self.tokenizer(
            pos_docs,
            padding=True,
            truncation=True,
            max_length=self.doc_max_length,
            return_tensors="pt",
        )
        batch["pos_doc_input_ids"] = pos_doc_encoded["input_ids"]
        batch["pos_doc_attention_mask"] = pos_doc_encoded["attention_mask"]

        # Tokenize negative documents
        # Flatten negatives for batch tokenization
        all_neg_docs = []
        for f in features:
            neg_docs = f.get("negative_docs", [])
            # Pad or truncate to num_negatives
            if len(neg_docs) < self.num_negatives:
                # Repeat last negative if not enough
                neg_docs = neg_docs + [neg_docs[-1]] * (
                    self.num_negatives - len(neg_docs)
                )
            else:
                neg_docs = neg_docs[: self.num_negatives]
            all_neg_docs.extend(neg_docs)

        neg_doc_encoded = self.tokenizer(
            all_neg_docs,
            padding=True,
            truncation=True,
            max_length=self.doc_max_length,
            return_tensors="pt",
        )

        # Reshape to [batch_size, num_negatives, seq_len]
        batch_size = len(features)
        seq_len = neg_doc_encoded["input_ids"].shape[1]

        batch["neg_doc_input_ids"] = neg_doc_encoded["input_ids"].view(
            batch_size, self.num_negatives, seq_len
        )
        batch["neg_doc_attention_mask"] = neg_doc_encoded["attention_mask"].view(
            batch_size, self.num_negatives, seq_len
        )

        # Handle synonym pairs if present
        korean_terms = [f.get("korean_term") for f in features if "korean_term" in f]
        english_terms = [
            f.get("english_term") for f in features if "english_term" in f
        ]

        if korean_terms and english_terms:
            # Tokenize Korean terms
            korean_encoded = self.tokenizer(
                korean_terms,
                padding=True,
                truncation=True,
                max_length=self.query_max_length,
                return_tensors="pt",
            )
            batch["korean_input_ids"] = korean_encoded["input_ids"]
            batch["korean_attention_mask"] = korean_encoded["attention_mask"]

            # Tokenize English terms
            english_encoded = self.tokenizer(
                english_terms,
                padding=True,
                truncation=True,
                max_length=self.query_max_length,
                return_tensors="pt",
            )
            batch["english_input_ids"] = english_encoded["input_ids"]
            batch["english_attention_mask"] = english_encoded["attention_mask"]

        return batch


class SimpleSynonymCollator:
    """
    Simple collator for synonym pairs only (for cross-lingual pre-training).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 64,
    ):
        """
        Initialize synonym collator.

        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum length for terms
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of synonym pairs.

        Args:
            features: List of dicts with 'korean' and 'english' keys

        Returns:
            Batch dict with tokenized inputs
        """
        korean_terms = [f["korean"] for f in features]
        english_terms = [f["english"] for f in features]

        # Tokenize Korean terms
        korean_encoded = self.tokenizer(
            korean_terms,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Tokenize English terms
        english_encoded = self.tokenizer(
            english_terms,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "korean_input_ids": korean_encoded["input_ids"],
            "korean_attention_mask": korean_encoded["attention_mask"],
            "english_input_ids": english_encoded["input_ids"],
            "english_attention_mask": english_encoded["attention_mask"],
        }


if __name__ == "__main__":
    # Test data collator
    from transformers import AutoTokenizer

    print("Testing NeuralSparseDataCollator...")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    # Create collator
    collator = NeuralSparseDataCollator(
        tokenizer=tokenizer,
        query_max_length=64,
        doc_max_length=256,
        num_negatives=3,
    )

    # Create dummy features
    features = [
        {
            "query": "인공지능 모델 학습",
            "positive_doc": "인공지능과 머신러닝 모델을 학습하는 방법에 대한 설명입니다.",
            "negative_docs": [
                "날씨가 좋은 날입니다.",
                "음식을 만드는 레시피입니다.",
                "여행 일정을 계획합니다.",
            ],
            "korean_term": "모델",
            "english_term": "model",
        },
        {
            "query": "검색 시스템 개발",
            "positive_doc": "검색 시스템을 개발하고 최적화하는 과정을 설명합니다.",
            "negative_docs": [
                "스포츠 경기 결과입니다.",
                "영화 리뷰입니다.",
                "책 추천 목록입니다.",
            ],
            "korean_term": "검색",
            "english_term": "search",
        },
    ]

    # Collate batch
    batch = collator(features)

    print("\nBatch keys:", batch.keys())
    print("\nQuery input_ids shape:", batch["query_input_ids"].shape)
    print("Positive doc input_ids shape:", batch["pos_doc_input_ids"].shape)
    print("Negative doc input_ids shape:", batch["neg_doc_input_ids"].shape)

    if "korean_input_ids" in batch:
        print("Korean term input_ids shape:", batch["korean_input_ids"].shape)
        print("English term input_ids shape:", batch["english_input_ids"].shape)

    print("\nTest SimpleSynonymCollator...")

    synonym_collator = SimpleSynonymCollator(tokenizer=tokenizer, max_length=64)

    synonym_features = [
        {"korean": "모델", "english": "model"},
        {"korean": "검색", "english": "search"},
        {"korean": "데이터", "english": "data"},
    ]

    synonym_batch = synonym_collator(synonym_features)

    print("\nSynonym batch keys:", synonym_batch.keys())
    print("Korean input_ids shape:", synonym_batch["korean_input_ids"].shape)
    print("English input_ids shape:", synonym_batch["english_input_ids"].shape)

    print("\nAll tests passed!")
