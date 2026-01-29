# 데이터 파이프라인 배치 형식 불일치 해결 가이드

## 문제 요약

학습 중 `KeyError: 'queries'` 에러가 발생하는 이유는 **Dataset 클래스와 DataCollator 간의 인터페이스 불일치** 때문입니다.

## 데이터 흐름 분석

### 1. 데이터 포맷

**train.jsonl 파일 구조:**
```json
{
  "query": "질문 텍스트",
  "docs": ["문서1", "문서2", "문서3", ...],
  "scores": [10.0, 7.84, 7.36, ...]
}
```

- `docs[0]`: Positive document (가장 높은 점수)
- `docs[1:]`: Hard negative documents (낮은 점수들)

### 2. 데이터 파이프라인 단계

```
JSONL 파일
    ↓
Dataset.__getitem__()
    ↓
DataCollator.__call__()
    ↓
Batch (train_step 입력)
```

### 3. 현재 문제점

#### src/data/dataset.py의 HardNegativesDataset

**반환 값 (토큰화된 텐서만):**
```python
{
    'query_input_ids': torch.Tensor,
    'query_attention_mask': torch.Tensor,
    'pos_doc_input_ids': torch.Tensor,
    'pos_doc_attention_mask': torch.Tensor,
    'neg_doc_input_ids': torch.Tensor,
    'neg_doc_attention_mask': torch.Tensor,
}
```

**문제:**
- 원본 텍스트가 없음 → Teacher 모델이 사용할 수 없음
- DataCollator가 이미 토큰화된 데이터를 다시 처리하려고 시도

#### 올바른 Dataset 반환 값 (DataCollator와 호환)

**반환 값 (원본 텍스트):**
```python
{
    'query': str,                    # 원본 query 텍스트
    'positive_doc': str,             # 원본 positive document 텍스트
    'negative_docs': List[str],      # 원본 negative documents 텍스트
}
```

**DataCollator 출력 (train_step 입력):**
```python
{
    # 원본 텍스트 (Teacher 모델용)
    'queries': List[str],
    'positive_docs': List[str],
    'negative_docs': List[List[str]],

    # 토큰화된 입력 (Student 모델용)
    'query_input_ids': torch.Tensor,
    'query_attention_mask': torch.Tensor,
    'pos_doc_input_ids': torch.Tensor,
    'pos_doc_attention_mask': torch.Tensor,
    'neg_doc_input_ids': torch.Tensor,
    'neg_doc_attention_mask': torch.Tensor,
}
```

## 해결 방법

### Option 1: JSONL 데이터 로더를 위한 새 Dataset 클래스 생성 (권장)

train.jsonl 포맷에 최적화된 Dataset 클래스를 생성합니다.

```python
# src/data/jsonl_dataset.py

from typing import Dict, List
import json
from torch.utils.data import Dataset


class NeuralSparseJSONLDataset(Dataset):
    """
    Dataset for neural sparse training from JSONL files.

    Expected format:
    {"query": "...", "docs": [...], "scores": [...]}

    Where docs[0] is positive, docs[1:] are hard negatives.
    """

    def __init__(
        self,
        jsonl_path: str,
        num_negatives: int = 7,
    ):
        """
        Initialize dataset.

        Args:
            jsonl_path: Path to JSONL file
            num_negatives: Number of negative samples to use
        """
        self.num_negatives = num_negatives
        self.data = self._load_jsonl(jsonl_path)

        print(f"Loaded {len(self.data)} samples from {jsonl_path}")

    def _load_jsonl(self, path: str) -> List[Dict]:
        """Load data from JSONL file."""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a training sample.

        Returns raw text (NOT tokenized) for DataCollator to process.
        """
        item = self.data[idx]

        query = item['query']
        docs = item['docs']

        # First doc is positive, rest are negatives
        positive_doc = docs[0]
        negative_docs = docs[1:self.num_negatives + 1]

        # Pad negatives if not enough
        while len(negative_docs) < self.num_negatives:
            # Repeat last negative or use positive as fallback
            negative_docs.append(negative_docs[-1] if negative_docs else positive_doc)

        return {
            'query': query,
            'positive_doc': positive_doc,
            'negative_docs': negative_docs,
        }
```

**사용 예시:**

```python
from src.data.jsonl_dataset import NeuralSparseJSONLDataset
from src.training.data_collator import NeuralSparseDataCollator
from torch.utils.data import DataLoader

# Dataset 생성
train_dataset = NeuralSparseJSONLDataset(
    jsonl_path="dataset/neural_sparse_training/train.jsonl",
    num_negatives=7,
)

# DataCollator 생성
data_collator = NeuralSparseDataCollator(
    tokenizer=tokenizer,
    query_max_length=64,
    doc_max_length=256,
    num_negatives=7,
)

# DataLoader 생성
train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=4,
)

# 배치 구조 확인
batch = next(iter(train_dataloader))
print(batch.keys())
# Output:
# dict_keys(['queries', 'positive_docs', 'negative_docs',
#            'query_input_ids', 'query_attention_mask',
#            'pos_doc_input_ids', 'pos_doc_attention_mask',
#            'neg_doc_input_ids', 'neg_doc_attention_mask'])
```

### Option 2: 기존 HardNegativesDataset 수정

기존 클래스에 원본 텍스트 반환 기능을 추가합니다.

```python
# src/data/dataset.py 수정

class HardNegativesDataset(Dataset):
    def __init__(
        self,
        data_files: List[str],
        tokenizer: Optional[AutoTokenizer] = None,  # None이면 텍스트만 반환
        max_length: int = 256,
        return_raw_text: bool = False,  # True면 원본 텍스트 포함
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_raw_text = return_raw_text
        # ... 나머지 초기화 코드

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        # Extract text
        query = item['query']
        pos_doc = item.get('pos') or item.get('positive_doc')
        hard_negatives = item.get('negs') or item.get('hard_negatives')

        # If only raw text is needed (for use with DataCollator)
        if self.return_raw_text or self.tokenizer is None:
            return {
                'query': query,
                'positive_doc': pos_doc,
                'negative_docs': hard_negatives,
            }

        # Otherwise tokenize (existing behavior)
        # ... 기존 토큰화 코드
```

### Option 3: train.jsonl 포맷 변환

JSONL 파일을 기존 Dataset이 기대하는 포맷으로 변환합니다.

```python
# scripts/convert_jsonl_format.py

import json
from pathlib import Path

def convert_format(input_path: str, output_path: str):
    """
    Convert from {"query": "...", "docs": [...], "scores": [...]}
    to {"query": "...", "pos": "...", "negs": [...]}
    """
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            if not line.strip():
                continue

            item = json.loads(line)

            # Convert format
            converted = {
                'query': item['query'],
                'pos': item['docs'][0],  # First doc is positive
                'negs': item['docs'][1:],  # Rest are negatives
            }

            f_out.write(json.dumps(converted, ensure_ascii=False) + '\n')

    print(f"Converted {input_path} -> {output_path}")

# Usage
convert_format(
    "dataset/neural_sparse_training/train.jsonl",
    "dataset/neural_sparse_training/train_converted.jsonl"
)
```

## 권장 해결책

**Option 1 (새 Dataset 클래스)을 권장합니다:**

1. ✅ 깔끔한 분리: JSONL 포맷 전용 Dataset
2. ✅ DataCollator와 완벽한 호환
3. ✅ Knowledge distillation 지원 (teacher 모델에 원본 텍스트 제공)
4. ✅ 기존 코드 수정 불필요
5. ✅ 유지보수 용이

## 구현 체크리스트

- [ ] `src/data/jsonl_dataset.py` 파일 생성
- [ ] `NeuralSparseJSONLDataset` 클래스 구현
- [ ] 단위 테스트 작성 (`tests/test_jsonl_dataset.py`)
- [ ] 노트북에서 새 Dataset 사용
- [ ] DataLoader 배치 구조 검증
- [ ] Teacher 모델과 통합 테스트
- [ ] 학습 실행 및 에러 확인

## 검증 방법

```python
# 배치 구조 검증 스크립트
from src.data.jsonl_dataset import NeuralSparseJSONLDataset
from src.training.data_collator import NeuralSparseDataCollator
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Setup
tokenizer = AutoTokenizer.from_pretrained("opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1")
dataset = NeuralSparseJSONLDataset("dataset/neural_sparse_training/train.jsonl", num_negatives=7)
collator = NeuralSparseDataCollator(tokenizer, query_max_length=64, doc_max_length=256, num_negatives=7)
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collator)

# Get sample batch
batch = next(iter(dataloader))

# Verify keys
required_keys = [
    'queries', 'positive_docs', 'negative_docs',  # Raw text for teacher
    'query_input_ids', 'query_attention_mask',    # Tokenized for student
    'pos_doc_input_ids', 'pos_doc_attention_mask',
    'neg_doc_input_ids', 'neg_doc_attention_mask',
]

print("Batch structure validation:")
for key in required_keys:
    status = "✓" if key in batch else "✗"
    print(f"  {status} {key}")

# Verify shapes
batch_size = len(batch['queries'])
print(f"\nBatch size: {batch_size}")
print(f"Query shape: {batch['query_input_ids'].shape}")
print(f"Pos doc shape: {batch['pos_doc_input_ids'].shape}")
print(f"Neg docs shape: {batch['neg_doc_input_ids'].shape}")

# Verify text is present
print(f"\nSample query text: {batch['queries'][0][:50]}...")
print(f"Sample pos doc text: {batch['positive_docs'][0][:50]}...")
print(f"Number of negatives: {len(batch['negative_docs'][0])}")
```

## Best Practices

### 1. 데이터 파이프라인 표준화

**Dataset의 역할:**
- JSONL 파일 로딩
- 데이터 포맷 검증
- **원본 텍스트 반환** (토큰화 X)

**DataCollator의 역할:**
- 배치 생성
- 토큰화
- Padding 및 텐서 변환
- Teacher 모델용 원본 텍스트 유지

**train_step의 역할:**
- 모델 forward pass
- Loss 계산
- Backward pass

### 2. 키 네이밍 컨벤션

**Dataset 출력 (단수형):**
- `'query'`: str
- `'positive_doc'`: str
- `'negative_docs'`: List[str]

**DataCollator 출력 (복수형 + 접두사):**
- `'queries'`: List[str] - 원본
- `'positive_docs'`: List[str] - 원본
- `'negative_docs'`: List[List[str]] - 원본
- `'query_input_ids'`: Tensor - 토큰화
- `'pos_doc_input_ids'`: Tensor - 토큰화
- `'neg_doc_input_ids'`: Tensor - 토큰화

### 3. 타입 힌팅 및 검증

```python
from typing import Dict, List, Union
from dataclasses import dataclass

@dataclass
class TrainingSample:
    """Training sample with type validation."""
    query: str
    positive_doc: str
    negative_docs: List[str]

    def __post_init__(self):
        assert isinstance(self.query, str), f"query must be str, got {type(self.query)}"
        assert isinstance(self.positive_doc, str), f"positive_doc must be str, got {type(self.positive_doc)}"
        assert isinstance(self.negative_docs, list), f"negative_docs must be list, got {type(self.negative_docs)}"
        assert all(isinstance(d, str) for d in self.negative_docs), "All negative_docs must be str"
```

### 4. 유닛 테스트

```python
# tests/test_data_pipeline.py

import pytest
from src.data.jsonl_dataset import NeuralSparseJSONLDataset
from src.training.data_collator import NeuralSparseDataCollator

def test_dataset_returns_text():
    """Dataset should return raw text, not tokens."""
    dataset = NeuralSparseJSONLDataset("test_data.jsonl")
    sample = dataset[0]

    assert 'query' in sample
    assert isinstance(sample['query'], str)
    assert 'positive_doc' in sample
    assert isinstance(sample['positive_doc'], str)
    assert 'negative_docs' in sample
    assert isinstance(sample['negative_docs'], list)

def test_collator_returns_both_text_and_tokens(tokenizer):
    """DataCollator should return both raw text and tokens."""
    collator = NeuralSparseDataCollator(tokenizer)
    features = [
        {
            'query': "test query",
            'positive_doc': "test doc",
            'negative_docs': ["neg1", "neg2"],
        }
    ]

    batch = collator(features)

    # Check raw text
    assert 'queries' in batch
    assert isinstance(batch['queries'], list)

    # Check tokens
    assert 'query_input_ids' in batch
    assert batch['query_input_ids'].dim() == 2  # [batch_size, seq_len]
```

## 참고 자료

- DataCollator 구현: `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/src/training/data_collator.py`
- 기존 Dataset 구현: `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/src/data/dataset.py`
- 학습 노트북: `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/notebooks/opensearch-neural-v2/02_training_opensearch_neural_v2.ipynb`
- 데이터 준비 노트북: `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/notebooks/opensearch-neural-v2/01_data_preparation_neural_sparse.ipynb`
