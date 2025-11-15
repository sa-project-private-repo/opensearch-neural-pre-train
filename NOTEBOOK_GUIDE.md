# Modular Notebook Pipeline Guide

모놀리식 노트북을 3개의 독립적인 노트북으로 분리한 실행 가이드입니다.

## 📚 노트북 구조

### 1. [01_neural_sparse_base_training.ipynb](notebooks/01_neural_sparse_base_training.ipynb)
**목적**: 기본 Neural Sparse 모델 학습

**주요 작업**:
- 한국어 데이터셋 로딩 (KLUE, KorQuAD, Wikipedia 등)
- IDF 통계 계산 및 트렌드 키워드 부스팅
- 이중언어 동의어 사전 구축
- Neural Sparse Encoder 학습
- Query-Document 쌍 생성 (negative sampling 포함)

**출력 데이터** (`dataset/base_model/`):
- `documents.json` - 한국어 문서 데이터
- `idf_statistics.pkl` - IDF 통계 (token_id → idf_score)
- `qd_pairs_base.pkl` - 기본 QD 쌍 (augmented with negatives)
- `bilingual_synonyms.json` - 이중언어 동의어 사전
- `neural_sparse_v1_model/` - 학습된 base 모델

**실행 시간**: ~60-90분 (GPU 사용 시)

---

### 2. [02_llm_synthetic_data_generation.ipynb](notebooks/02_llm_synthetic_data_generation.ipynb)
**목적**: LLM 기반 합성 쿼리 데이터 생성

**전제조건**: Notebook 1 완료 필수

**주요 작업**:
- Qwen3-30B-A3B-Thinking-2507-FP8 모델 로딩 (~30GB)
- 문서 기반 합성 쿼리 생성 (document → query 역생성)
- LLM 기반 동의어 검증 및 확장

**출력 데이터** (`dataset/llm_generated/`):
- `synthetic_qd_pairs.pkl` - LLM 생성 Query-Document 쌍
- `enhanced_synonyms.json` - LLM 검증 동의어 사전

**실행 시간**: ~15-30분 (LLM 로딩 + 데이터 생성)

**참고**:
- 첫 실행 시 모델 다운로드에 시간이 소요됩니다
- ARM aarch64 환경에서 FP8 quantization 사용
- Triton 컴파일은 자동으로 비활성화됩니다

---

### 3. [03_llm_enhanced_training.ipynb](notebooks/03_llm_enhanced_training.ipynb)
**목적**: Enhanced 모델 학습 및 성능 비교

**전제조건**: Notebook 1, 2 완료 필수

**주요 작업**:
- Base 데이터 + LLM 생성 데이터 결합
- Enhanced Neural Sparse 모델 학습
- Base vs Enhanced 성능 비교

**출력 데이터** (`dataset/enhanced_model/`):
- `neural_sparse_v2_model/` - Enhanced 모델
- `evaluation/performance_comparison.json` - 성능 비교 결과

**실행 시간**: ~60-90분 (GPU 사용 시)

---

## 🚀 실행 방법

### 순차 실행 (권장)

```bash
# 1단계: Base 모델 학습
jupyter notebook notebooks/01_neural_sparse_base_training.ipynb
# 모든 셀 실행 후 kernel 종료 가능

# 2단계: LLM 합성 데이터 생성
jupyter notebook notebooks/02_llm_synthetic_data_generation.ipynb
# 모든 셀 실행 후 kernel 종료 가능

# 3단계: Enhanced 모델 학습 및 비교
jupyter notebook notebooks/03_llm_enhanced_training.ipynb
# 모든 셀 실행
```

### 일괄 실행 (자동화)

```bash
# nbconvert를 사용한 자동 실행
jupyter nbconvert --to notebook --execute \
    notebooks/01_neural_sparse_base_training.ipynb \
    --output 01_neural_sparse_base_training_output.ipynb

jupyter nbconvert --to notebook --execute \
    notebooks/02_llm_synthetic_data_generation.ipynb \
    --output 02_llm_synthetic_data_generation_output.ipynb

jupyter nbconvert --to notebook --execute \
    notebooks/03_llm_enhanced_training.ipynb \
    --output 03_llm_enhanced_training_output.ipynb
```

---

## 🔄 재실행 전략

### 시나리오 1: 데이터만 변경
**상황**: 다른 한국어 데이터셋 사용
**재실행**: Notebook 1 → 2 → 3 (전체)

### 시나리오 2: LLM 파라미터 변경
**상황**: LLM 합성 쿼리 생성 로직 수정
**재실행**: Notebook 2 → 3 (Notebook 1 건너뛰기)

### 시나리오 3: 학습 하이퍼파라미터 변경
**상황**: Learning rate, epochs 등 조정
**재실행**: Notebook 3만 (Notebook 1, 2 건너뛰기)

---

## 📊 데이터 흐름

```
Notebook 1                  Notebook 2                  Notebook 3
┌─────────────┐            ┌─────────────┐            ┌─────────────┐
│  한국어      │            │  LLM 모델    │            │  Base +     │
│  데이터셋    │            │  로딩        │            │  LLM 데이터 │
│  로딩       │            │             │            │  결합       │
└──────┬──────┘            └──────┬──────┘            └──────┬──────┘
       │                          │                          │
       ▼                          ▼                          ▼
┌─────────────┐            ┌─────────────┐            ┌─────────────┐
│  IDF 계산    │            │  합성 쿼리   │            │  Enhanced   │
│  & 트렌드    │            │  생성       │            │  모델 학습   │
│  부스팅     │            │             │            │             │
└──────┬──────┘            └──────┬──────┘            └──────┬──────┘
       │                          │                          │
       ▼                          ▼                          ▼
┌─────────────┐            ┌─────────────┐            ┌─────────────┐
│  Base 모델   │            │  동의어     │            │  성능 비교   │
│  학습       │            │  검증       │            │  & 평가     │
└──────┬──────┘            └──────┬──────┘            └──────┬──────┘
       │                          │                          │
       ▼                          ▼                          ▼
   dataset/                  dataset/                  dataset/
   base_model/              llm_generated/            enhanced_model/
```

---

## 💾 저장 데이터 구조

```
dataset/
├── metadata.json                    # 전체 데이터셋 메타데이터
│
├── base_model/                      # Notebook 1 출력
│   ├── documents.json               # 한국어 문서 (list)
│   ├── idf_statistics.pkl           # IDF dict (token_id → score)
│   ├── qd_pairs_base.pkl            # QD pairs with negatives
│   ├── bilingual_synonyms.json      # 한영 동의어 사전
│   └── neural_sparse_v1_model/      # Base 모델
│       ├── pytorch_model.bin
│       ├── config.json
│       └── tokenizer files...
│
├── llm_generated/                   # Notebook 2 출력
│   ├── synthetic_qd_pairs.pkl       # LLM 생성 쿼리
│   └── enhanced_synonyms.json       # LLM 검증 동의어
│
└── enhanced_model/                  # Notebook 3 출력
    ├── neural_sparse_v2_model/      # Enhanced 모델
    │   ├── pytorch_model.bin
    │   ├── config.json
    │   └── tokenizer files...
    └── evaluation/
        └── performance_comparison.json  # 성능 비교
```

---

## 🔍 의존성 검증

각 노트북은 시작 시 자동으로 필요한 데이터 파일을 확인합니다:

### Notebook 2 의존성
```python
required_files = [
    ("base_model", "documents.json"),
    ("base_model", "bilingual_synonyms.json"),
]
```

### Notebook 3 의존성
```python
required_files = [
    ("base_model", "documents.json"),
    ("base_model", "qd_pairs_base.pkl"),
    ("base_model", "neural_sparse_v1_model"),
    ("llm_generated", "synthetic_qd_pairs.pkl"),
    ("llm_generated", "enhanced_synonyms.json"),
]
```

의존성이 만족되지 않으면 자동으로 에러 메시지와 함께 실행이 중단됩니다.

---

## ⚙️ 데이터 관리

### 데이터 요약 보기
```python
from src.dataset_manager import DatasetManager

dm = DatasetManager(base_path="dataset")
dm.print_summary()
```

### 특정 디렉토리 정리
```python
# 주의: 되돌릴 수 없습니다!
dm.clear_subdirectory("llm_generated", confirm=True)
```

### 파일 존재 확인
```python
if dm.check_data_exists("documents.json", "base_model"):
    print("Base model data exists!")
```

---

## 🐛 문제 해결

### NameError: name 'documents' is not defined
**원인**: Notebook 1을 실행하지 않고 Notebook 2를 실행
**해결**: Notebook 1을 먼저 실행

### FileNotFoundError: documents.json
**원인**: Notebook 1의 데이터 저장 셀을 실행하지 않음
**해결**: Notebook 1의 마지막 섹션 (데이터 저장) 셀들을 실행

### CUDA out of memory
**원인**: GPU 메모리 부족
**해결**:
- Batch size 줄이기
- 이전 노트북의 kernel 종료
- `nvidia-smi`로 GPU 메모리 확인

### Triton compilation errors (ARM)
**원인**: ARM aarch64에서 Triton JIT 컴파일 실패
**해결**: 자동으로 비활성화됨 (환경 변수 설정됨)

---

## 📈 기대 효과

### 시간 절약
- **기존**: ~90분 (매번 전체 실행)
- **개선**: ~40분 (변경된 부분만 재실행)
- **절감**: 50% (~50분)

### 메모리 절약
- **기존**: ~40GB (전체 파이프라인 동시 로딩)
- **개선**: ~20GB (노트북별 독립 실행)
- **절감**: 50% (~20GB)

### 유연성 향상
- LLM 파라미터만 변경 → Notebook 2, 3만 재실행
- 학습 로직만 변경 → Notebook 3만 재실행
- 독립적인 디버깅 가능

---

## 📚 참고 자료

- [DatasetManager API](src/dataset_manager.py)
- [Pipeline Validation](validate_pipeline.py)
- [Original Plan](plan.md)
- [OpenSearch Neural Sparse Docs](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/)
