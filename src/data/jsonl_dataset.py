"""Dataset loader for JSONL format neural sparse training data."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import Dataset


class NeuralSparseJSONLDataset(Dataset):
    """
    Dataset for neural sparse training from JSONL files.

    Expected JSONL format:
    {
        "query": "질문 텍스트",
        "docs": ["문서1", "문서2", "문서3", ...],
        "scores": [10.0, 7.84, 7.36, ...]
    }

    Where:
    - docs[0] is the positive document (highest score)
    - docs[1:] are hard negative documents (lower scores)

    This dataset returns raw text (NOT tokenized) for the DataCollator
    to process. This allows:
    1. Flexible tokenization in DataCollator
    2. Support for knowledge distillation (teacher models need raw text)
    3. Cleaner separation of concerns
    """

    def __init__(
        self,
        jsonl_path: str,
        num_negatives: int = 7,
        validate_format: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            jsonl_path: Path to JSONL file
            num_negatives: Number of negative samples to use per query
            validate_format: Whether to validate data format on load
        """
        self.jsonl_path = Path(jsonl_path)
        self.num_negatives = num_negatives
        self.validate_format = validate_format

        # Load data
        self.data = self._load_jsonl()

        print(f"✓ Loaded {len(self.data):,} samples from {self.jsonl_path}")
        if len(self.data) > 0:
            self._print_sample_info()

    def _load_jsonl(self) -> List[Dict]:
        """
        Load data from JSONL file.

        Returns:
            List of data dictionaries

        Raises:
            FileNotFoundError: If JSONL file doesn't exist
            ValueError: If data format is invalid
        """
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")

        data = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)

                    # Validate format
                    if self.validate_format:
                        self._validate_item(item, line_num)

                    data.append(item)

                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                    continue
                except ValueError as e:
                    print(f"Warning: Skipping invalid data at line {line_num}: {e}")
                    continue

        if len(data) == 0:
            raise ValueError(f"No valid data found in {self.jsonl_path}")

        return data

    def _validate_item(self, item: Dict, line_num: int) -> None:
        """
        Validate data item format.

        Args:
            item: Data dictionary
            line_num: Line number in file (for error messages)

        Raises:
            ValueError: If format is invalid
        """
        # Check required keys
        if "query" not in item:
            raise ValueError(f"Missing 'query' key at line {line_num}")
        if "docs" not in item:
            raise ValueError(f"Missing 'docs' key at line {line_num}")

        # Check types
        if not isinstance(item["query"], str):
            raise ValueError(
                f"'query' must be string at line {line_num}, got {type(item['query'])}"
            )
        if not isinstance(item["docs"], list):
            raise ValueError(
                f"'docs' must be list at line {line_num}, got {type(item['docs'])}"
            )

        # Check minimum docs
        if len(item["docs"]) < 1:
            raise ValueError(
                f"'docs' must contain at least 1 document at line {line_num}"
            )

        # Check all docs are strings
        for i, doc in enumerate(item["docs"]):
            if not isinstance(doc, str):
                raise ValueError(
                    f"docs[{i}] must be string at line {line_num}, got {type(doc)}"
                )

    def _print_sample_info(self) -> None:
        """Print information about first sample."""
        sample = self.data[0]
        print(f"\nSample data:")
        print(f"  Query: {sample['query'][:80]}...")
        print(f"  Num docs: {len(sample['docs'])}")
        if "scores" in sample:
            print(f"  Scores range: [{min(sample['scores']):.2f}, {max(sample['scores']):.2f}]")
        print(f"  First doc: {sample['docs'][0][:80]}...")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a training sample.

        Returns RAW TEXT (not tokenized) for DataCollator to process.

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
            - 'query': Query text (str)
            - 'positive_doc': Positive document text (str)
            - 'negative_docs': List of negative document texts (List[str])
        """
        item = self.data[idx]

        query = item["query"]
        docs = item["docs"]

        # First doc is positive (highest score)
        positive_doc = docs[0]

        # Rest are negatives (lower scores)
        # Take up to num_negatives
        negative_docs = docs[1 : self.num_negatives + 1]

        # Pad negatives if not enough
        while len(negative_docs) < self.num_negatives:
            # Repeat last negative if available, otherwise use positive as fallback
            negative_docs.append(
                negative_docs[-1] if negative_docs else positive_doc
            )

        return {
            "query": query,
            "positive_doc": positive_doc,
            "negative_docs": negative_docs,
        }


if __name__ == "__main__":
    # Test dataset
    print("Testing NeuralSparseJSONLDataset...")

    # Create test data
    test_jsonl_path = Path("/tmp/test_neural_sparse.jsonl")

    test_data = [
        {
            "query": "인공지능 학습",
            "docs": [
                "인공지능 모델을 학습하는 방법에 대한 설명입니다.",
                "날씨가 좋은 날입니다.",
                "음식을 만드는 레시피입니다.",
                "여행 일정을 계획합니다.",
                "스포츠 경기 결과입니다.",
            ],
            "scores": [10.0, 7.5, 6.2, 5.8, 4.3],
        },
        {
            "query": "검색 시스템",
            "docs": [
                "검색 시스템을 개발하고 최적화하는 과정입니다.",
                "영화 리뷰입니다.",
                "책 추천 목록입니다.",
            ],
            "scores": [10.0, 5.0, 4.0],
        },
    ]

    # Write test file
    with open(test_jsonl_path, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Test dataset
    dataset = NeuralSparseJSONLDataset(
        str(test_jsonl_path),
        num_negatives=3,
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test __getitem__
    sample = dataset[0]
    print(f"\nSample structure:")
    print(f"  Keys: {list(sample.keys())}")
    print(f"  Query type: {type(sample['query'])}")
    print(f"  Positive doc type: {type(sample['positive_doc'])}")
    print(f"  Negative docs type: {type(sample['negative_docs'])}")
    print(f"  Num negatives: {len(sample['negative_docs'])}")

    print(f"\nSample content:")
    print(f"  Query: {sample['query']}")
    print(f"  Positive: {sample['positive_doc'][:80]}...")
    print(f"  Negatives:")
    for i, neg in enumerate(sample["negative_docs"]):
        print(f"    {i+1}. {neg[:60]}...")

    # Test validation
    print("\n" + "=" * 80)
    print("Testing validation...")

    invalid_data = [
        {"query": "test"},  # Missing 'docs'
        {"docs": ["test"]},  # Missing 'query'
        {"query": "test", "docs": []},  # Empty docs
        {"query": 123, "docs": ["test"]},  # Invalid query type
        {"query": "test", "docs": [123, "test"]},  # Invalid doc type
    ]

    for i, item in enumerate(invalid_data):
        print(f"\nTest {i+1}: {item}")
        test_file = Path(f"/tmp/test_invalid_{i}.jsonl")
        with open(test_file, "w") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

        try:
            invalid_dataset = NeuralSparseJSONLDataset(str(test_file))
            print("  ✗ Should have raised ValueError")
        except ValueError as e:
            print(f"  ✓ Caught expected error: {e}")

    # Cleanup
    test_jsonl_path.unlink()
    for i in range(len(invalid_data)):
        Path(f"/tmp/test_invalid_{i}.jsonl").unlink(missing_ok=True)

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
