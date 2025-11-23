# 데이터 파이프라인 배치 형식 불일치 분석 보고서

**작성일:** 2025-11-23
**상태:** ✅ 해결 완료

## 1. 문제 요약

### 에러 발생
```python
KeyError: 'queries'
# train_step에서 batch['queries']를 찾을 수 없음
```

### 근본 원인
Dataset 클래스가 이미 토큰화된 텐서를 반환하여, DataCollator가 원본 텍스트를 받지 못하고 teacher 모델이 사용할 수 없었습니다.

## 2. 데이터 흐름 분석

### 2.1 실제 데이터 형식

**train.jsonl 파일 구조:**
```json
{
  "query": "갈매기류",
  "docs": [
    "갈매기과()의 한 과이다...",  // Positive (score: 10.0)
    "갈매기과()의 한 과이다...",  // Hard negative (score: 7.84)
    "도요목 또는 물떼새목...",    // Hard negative (score: 7.36)
    ...
  ],
  "scores": [10.0, 7.84, 7.36, ...]
}
```

**데이터 통계:**
- 총 샘플 수: 21,590개
- Query당 문서 수: 8개 (positive 1개 + negatives 7개)
- 점수 범위: 0.5 ~ 10.0

### 2.2 데이터 파이프라인 단계

```
JSONL 파일
    ↓ Dataset.__getitem__()
원본 텍스트 Dictionary
    ↓ DataCollator.__call__()
Batch (토큰화 + 원본 텍스트)
    ↓ train_step()
모델 학습
```

## 3. 문제 진단

### 3.1 기존 Dataset 클래스의 문제

**src/data/dataset.py의 HardNegativesDataset:**

```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    return {
        'query_input_ids': query_encoded['input_ids'].squeeze(0),
        'query_attention_mask': query_encoded['attention_mask'].squeeze(0),
        'pos_doc_input_ids': pos_doc_encoded['input_ids'].squeeze(0),
        'pos_doc_attention_mask': pos_doc_encoded['attention_mask'].squeeze(0),
        'neg_doc_input_ids': neg_docs_encoded['input_ids'],
        'neg_doc_attention_mask': neg_docs_encoded['attention_mask'],
    }
```

**문제점:**
1. ❌ 이미 토큰화된 텐서만 반환
2. ❌ 원본 텍스트가 없음
3. ❌ Teacher 모델이 사용 불가
4. ❌ DataCollator와 인터페이스 불일치

### 3.2 DataCollator의 기대 입력

**src/training/data_collator.py의 NeuralSparseDataCollator:**

```python
def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    # 입력으로 원본 텍스트를 기대
    queries = [f["query"] for f in features]  # str 기대
    pos_docs = [f["positive_doc"] for f in features]  # str 기대

    # 원본 텍스트를 배치에 저장 (Teacher 모델용)
    batch["queries"] = queries
    batch["positive_docs"] = pos_docs
    batch["negative_docs"] = [f["negative_docs"] for f in features]

    # 토큰화 수행
    # ...
```

**기대 입력:**
- ✅ `'query'`: str (원본 텍스트)
- ✅ `'positive_doc'`: str (원본 텍스트)
- ✅ `'negative_docs'`: List[str] (원본 텍스트)

### 3.3 train_step의 기대 배치 구조

```python
def train_step(batch, model, teacher):
    # Teacher 모델에 원본 텍스트 필요
    queries = batch['queries']  # ← KeyError 발생 지점!
    positive_docs = batch['positive_docs']
    negative_docs = batch['negative_docs']

    teacher_scores = teacher.get_scores(queries, all_docs)

    # Student 모델에 토큰화된 입력 필요
    query_rep = model(
        input_ids=batch['query_input_ids'],
        attention_mask=batch['query_attention_mask'],
    )
```

**필요한 배치 구조:**
- Teacher 모델: `'queries'`, `'positive_docs'`, `'negative_docs'` (원본 텍스트)
- Student 모델: `'query_input_ids'`, `'pos_doc_input_ids'`, 등 (토큰화된 텐서)

## 4. 해결 방법

### 4.1 새 Dataset 클래스 구현 ✅

**src/data/jsonl_dataset.py - NeuralSparseJSONLDataset:**

```python
class NeuralSparseJSONLDataset(Dataset):
    """
    JSONL 포맷 전용 Dataset.

    원본 텍스트만 반환 (토큰화 X)
    DataCollator가 토큰화 담당
    """

    def __getitem__(self, idx: int) -> Dict[str, any]:
        item = self.data[idx]

        query = item["query"]
        docs = item["docs"]

        # docs[0]은 positive, docs[1:]은 negatives
        positive_doc = docs[0]
        negative_docs = docs[1:self.num_negatives + 1]

        # 원본 텍스트만 반환
        return {
            "query": query,                    # str
            "positive_doc": positive_doc,      # str
            "negative_docs": negative_docs,    # List[str]
        }
```

**특징:**
- ✅ 원본 텍스트만 반환
- ✅ DataCollator와 완벽 호환
- ✅ Knowledge distillation 지원
- ✅ JSONL 포맷에 최적화

### 4.2 데이터 파이프라인 완성

```python
# 1. Dataset 생성
dataset = NeuralSparseJSONLDataset(
    jsonl_path="dataset/neural_sparse_training/train.jsonl",
    num_negatives=7,
)

# 2. DataCollator 생성
data_collator = NeuralSparseDataCollator(
    tokenizer=tokenizer,
    query_max_length=64,
    doc_max_length=256,
    num_negatives=7,
)

# 3. DataLoader 생성
dataloader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=data_collator,
)

# 4. Batch 구조 확인
batch = next(iter(dataloader))
```

**Batch 구조 (검증 완료):**
```python
{
    # Teacher 모델용 (원본 텍스트)
    'queries': List[str],              # [batch_size]
    'positive_docs': List[str],        # [batch_size]
    'negative_docs': List[List[str]],  # [batch_size, num_negatives]

    # Student 모델용 (토큰화)
    'query_input_ids': Tensor,         # [batch_size, query_seq_len]
    'query_attention_mask': Tensor,    # [batch_size, query_seq_len]
    'pos_doc_input_ids': Tensor,       # [batch_size, doc_seq_len]
    'pos_doc_attention_mask': Tensor,  # [batch_size, doc_seq_len]
    'neg_doc_input_ids': Tensor,       # [batch_size, num_neg, doc_seq_len]
    'neg_doc_attention_mask': Tensor,  # [batch_size, num_neg, doc_seq_len]
}
```

## 5. 검증 결과

### 5.1 자동 검증 스크립트 실행

```bash
$ python scripts/validate_data_pipeline.py
```

**검증 결과:**
```
✓ Dataset: 21,590 samples loaded
✓ DataLoader: 5,398 batches
✓ Batch size: 4
✓ Num negatives: 7
✓ All required keys present
✓ All shapes correct
✓ All types valid

✓✓✓ DATA PIPELINE IS VALID ✓✓✓
```

### 5.2 배치 구조 검증

**Student 모델 입력 (토큰화):**
- ✅ query_input_ids: torch.Size([4, 19])
- ✅ query_attention_mask: torch.Size([4, 19])
- ✅ pos_doc_input_ids: torch.Size([4, 256])
- ✅ pos_doc_attention_mask: torch.Size([4, 256])
- ✅ neg_doc_input_ids: torch.Size([4, 7, 256])
- ✅ neg_doc_attention_mask: torch.Size([4, 7, 256])

**Teacher 모델 입력 (원본 텍스트):**
- ✅ queries: 4 strings
- ✅ positive_docs: 4 strings
- ✅ negative_docs: 4 lists of 7 strings each

## 6. Best Practices 정립

### 6.1 데이터 파이프라인 설계 원칙

```
[Principle 1] 역할 분리
- Dataset: 데이터 로딩 + 원본 텍스트 반환
- DataCollator: 배치 생성 + 토큰화
- train_step: 모델 학습

[Principle 2] 인터페이스 일관성
- Dataset 출력 ↔ DataCollator 입력 일치
- DataCollator 출력 ↔ train_step 입력 일치

[Principle 3] 유연성
- 원본 텍스트 유지 → Teacher 모델 지원
- 토큰화 분리 → 다양한 tokenizer 사용 가능
```

### 6.2 키 네이밍 컨벤션

**Dataset 출력 (단수형):**
```python
{
    'query': str,
    'positive_doc': str,
    'negative_docs': List[str],
}
```

**DataCollator 출력 (복수형 + 접두사):**
```python
{
    # 원본 (복수형)
    'queries': List[str],
    'positive_docs': List[str],
    'negative_docs': List[List[str]],

    # 토큰화 (접두사 + 복수형)
    'query_input_ids': Tensor,
    'pos_doc_input_ids': Tensor,
    'neg_doc_input_ids': Tensor,
}
```

### 6.3 타입 힌팅 및 검증

```python
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class TrainingSample:
    """Type-safe training sample."""
    query: str
    positive_doc: str
    negative_docs: List[str]

    def __post_init__(self):
        # Type validation
        assert isinstance(self.query, str)
        assert isinstance(self.positive_doc, str)
        assert isinstance(self.negative_docs, list)
        assert all(isinstance(d, str) for d in self.negative_docs)
```

## 7. 구현 파일 목록

### 7.1 새로 생성된 파일

| 파일 경로 | 용도 | 상태 |
|---------|------|------|
| `src/data/jsonl_dataset.py` | JSONL Dataset 클래스 | ✅ 완료 |
| `scripts/validate_data_pipeline.py` | 파이프라인 검증 스크립트 | ✅ 완료 |
| `docs/DATA_PIPELINE_FIX.md` | 해결 방법 가이드 | ✅ 완료 |
| `docs/DATA_PIPELINE_ANALYSIS_REPORT.md` | 분석 보고서 (본 문서) | ✅ 완료 |

### 7.2 기존 파일 (수정 불필요)

| 파일 경로 | 상태 | 비고 |
|---------|------|------|
| `src/training/data_collator.py` | ✅ 정상 | 수정 불필요 |
| `src/data/dataset.py` | ⚠️ 사용 안 함 | 기존 프로젝트용, 호환성 유지 |
| `notebooks/opensearch-neural-v2/02_training_opensearch_neural_v2.ipynb` | 📝 업데이트 필요 | Dataset 클래스 변경 |

## 8. 노트북 수정 가이드

### 8.1 필요한 변경사항

**02_training_opensearch_neural_v2.ipynb의 Dataset 생성 부분:**

**기존 코드 (노트북 내 SparseRetrievalDataset):**
```python
# 노트북에 정의된 클래스 사용
train_dataset = SparseRetrievalDataset(
    queries=train_queries,
    positive_docs=train_positive_docs,
    negative_docs=train_negative_docs,
)
```

**변경 후 코드:**
```python
# 새로운 JSONL Dataset 사용
from src.data.jsonl_dataset import NeuralSparseJSONLDataset

train_dataset = NeuralSparseJSONLDataset(
    jsonl_path="dataset/neural_sparse_training/train.jsonl",
    num_negatives=7,
    validate_format=True,
)

val_dataset = NeuralSparseJSONLDataset(
    jsonl_path="dataset/neural_sparse_training/val.jsonl",
    num_negatives=7,
    validate_format=True,
)
```

### 8.2 변경 이유

1. ✅ **단순화**: JSONL 파일에서 직접 로드
2. ✅ **일관성**: 데이터 준비 노트북과 동일한 포맷
3. ✅ **검증**: 자동 포맷 검증 내장
4. ✅ **유지보수**: 중복 코드 제거

## 9. 결론

### 9.1 해결 완료

✅ **JSONL Dataset 클래스 구현 완료**
- 원본 텍스트 반환
- DataCollator와 완벽 호환
- Knowledge distillation 지원

✅ **파이프라인 검증 완료**
- 21,590개 샘플 로드 확인
- 배치 구조 검증 완료
- Teacher/Student 모델 입력 준비 완료

✅ **Best Practices 정립**
- 역할 분리 명확화
- 키 네이밍 컨벤션 확립
- 타입 검증 표준화

### 9.2 다음 단계

**즉시 가능:**
1. ✅ 노트북에서 새 Dataset 사용
2. ✅ 학습 실행
3. ✅ Teacher 모델 통합

**향후 개선:**
1. 📝 더 많은 데이터 포맷 지원
2. 📝 Dynamic negative sampling
3. 📝 Data augmentation 추가

## 10. 참고 자료

### 10.1 구현 코드
- `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/src/data/jsonl_dataset.py`
- `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/src/training/data_collator.py`
- `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/scripts/validate_data_pipeline.py`

### 10.2 문서
- `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/docs/DATA_PIPELINE_FIX.md`
- `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/CLAUDE.md`

### 10.3 데이터
- `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/dataset/neural_sparse_training/train.jsonl` (21,590 samples)
- `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/dataset/neural_sparse_training/val.jsonl`

---

**보고서 작성:** Claude (Anthropic)
**검증 일시:** 2025-11-23
**상태:** ✅ 해결 완료 및 검증 통과
