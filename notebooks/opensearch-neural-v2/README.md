# OpenSearch Neural Sparse Model - 학습 데이터 준비

OpenSearch neural sparse 모델 학습을 위한 한국어 데이터셋 준비 노트북입니다.

## 개요

이 노트북은 한국어 Wikipedia와 Namuwiki 데이터를 활용하여 `opensearch-sparse-model-tuning-sample` 레포지토리와 호환되는 학습 데이터를 생성합니다.

### 데이터 포맷

최종 생성되는 JSONL 파일은 **Pre-computed Knowledge Distillation** 형식을 따릅니다:

```json
{
    "query": "질문 텍스트",
    "docs": ["문서1", "문서2", "문서3", ...],
    "scores": [10.0, 7.5, 5.2, 3.1, ...]
}
```

## 파일 구조

```
notebooks/opensearch-neural-v2/
├── README.md                                # 이 파일
└── 01_data_preparation_neural_sparse.ipynb  # 데이터 준비 노트북
```

## 실행 방법

### 1. 환경 설정

```bash
# 프로젝트 루트에서
cd /home/west/Documents/cursor-workspace/opensearch-neural-pre-train

# 가상환경 활성화
source .venv/bin/activate

# 필요한 패키지 설치 (미설치 시)
pip install sentence-transformers faiss-cpu scikit-learn torch numpy pandas tqdm
```

### 2. 노트북 실행

```bash
# Jupyter 실행
jupyter lab notebooks/opensearch-neural-v2/01_data_preparation_neural_sparse.ipynb
```

또는 VSCode에서 직접 노트북을 열어 실행합니다.

### 3. 데이터 크기 조정

노트북 내에서 `load_korean_datasets()` 함수의 파라미터를 조정하여 데이터 크기를 설정할 수 있습니다:

```python
# 테스트용 (빠른 실행)
documents = load_korean_datasets(
    wiki_limit=10000,
    namuwiki_limit=5000,
)

# 전체 데이터 사용
documents = load_korean_datasets(
    wiki_limit=None,
    namuwiki_limit=None,
)
```

## 주요 처리 단계

### 1. 데이터 로딩 및 전처리
- Wikipedia (ko) + Namuwiki JSONL 파일 로드
- 텍스트 길이 필터링 (100~2000자)
- Document 객체로 변환

### 2. Query-Document 쌍 생성
- 문서 제목 → Query
- 문서 본문 → Positive Document

### 3. Embedding 생성
- 모델: `intfloat/multilingual-e5-large`
- Query embedding: `"query: " + 제목`
- Document embedding: `"passage: " + 본문`

### 4. Hard Negatives Mining
- FAISS를 사용한 유사도 기반 검색
- 각 query당 상위 K개 유사 문서 선택
- Cosine similarity를 점수로 변환

### 5. K-means 클러스터링
- 문서 임베딩을 클러스터링하여 관련 문서 그룹화
- 클러스터 내 문서들에 대한 관련성 점수 계산

### 6. 최종 데이터셋 생성
- 각 query당 8개 문서:
  - Positive document (1개, score=10.0)
  - Hard negatives (6~7개, score=유사도 기반)
  - Random negatives (필요시, score=0.5)
- Train/Val split (90/10)

## 출력 파일

실행 완료 후 다음 파일들이 생성됩니다:

```
dataset/neural_sparse_training/
├── train.jsonl                      # 학습 데이터
├── val.jsonl                        # 검증 데이터
├── metadata.json                    # 메타데이터
└── embeddings/
    ├── query_embeddings.npy         # Query 임베딩
    ├── document_embeddings.npy      # Document 임베딩
    └── (faiss_index.bin)            # FAISS 인덱스 (선택적)
```

### train.jsonl / val.jsonl 형식

```json
{"query": "지미 카터", "docs": ["제임스 얼 지미 카터...", "..."], "scores": [10.0, 7.5, 5.2, 3.1, 2.0, 1.5, 1.0, 0.5]}
{"query": "수학", "docs": ["수학은 수, 양, 구조...", "..."], "scores": [10.0, 8.2, 6.5, 4.3, 3.0, 2.1, 1.2, 0.8]}
```

### metadata.json 내용

```json
{
  "created_at": "2025-11-23T...",
  "total_documents": 15000,
  "total_queries": 15000,
  "train_samples": 13500,
  "val_samples": 1500,
  "docs_per_query": 8,
  "embedding_model": "intfloat/multilingual-e5-large",
  "embedding_dimension": 1024,
  "source_datasets": ["wikipedia_ko", "namuwiki"],
  "num_clusters": 100,
  "data_format": "pre-computed knowledge distillation",
  "compatible_with": "opensearch-sparse-model-tuning-sample"
}
```

## 데이터 통계 (예상)

| 항목 | 값 |
|------|------|
| 총 문서 수 | ~15,000 (테스트) / ~50,000+ (전체) |
| 학습 샘플 | ~13,500 (테스트) / ~45,000+ (전체) |
| 검증 샘플 | ~1,500 (테스트) / ~5,000+ (전체) |
| Query당 문서 수 | 8개 |
| Embedding 차원 | 1024 |
| 점수 범위 | 0.5 ~ 10.0 |

## 주의사항

### 리소스 요구사항

- **메모리**: 최소 16GB RAM (전체 데이터 사용 시 32GB+ 권장)
- **GPU**: CUDA 지원 GPU 권장 (CPU로도 가능하나 느림)
- **디스크**: 약 5~10GB 여유 공간

### 실행 시간 (예상)

| 단계 | 시간 (GPU) | 시간 (CPU) |
|------|-----------|-----------|
| 데이터 로딩 | ~2분 | ~2분 |
| Embedding 생성 (10K docs) | ~10분 | ~1시간 |
| FAISS 검색 | ~1분 | ~5분 |
| K-means 클러스터링 | ~2분 | ~10분 |
| 데이터셋 저장 | ~1분 | ~1분 |
| **총 소요시간** | **~15분** | **~1.5시간** |

*전체 데이터 사용 시 3~5배 소요*

## 다음 단계

생성된 데이터셋을 사용하여 모델을 학습합니다:

```bash
# opensearch-sparse-model-tuning-sample 레포지토리에서
python train_ir.py \
    --train_data dataset/neural_sparse_training/train.jsonl \
    --val_data dataset/neural_sparse_training/val.jsonl \
    --config configs/pretrain_korean.yaml
```

## 문제 해결

### Out of Memory 에러

```python
# 배치 크기 줄이기
query_embeddings = generate_embeddings(
    query_texts,
    model,
    batch_size=16,  # 32 → 16으로 줄임
)
```

### CUDA Out of Memory

```python
# CPU 사용
DEVICE = "cpu"
```

### 느린 실행 속도

```python
# 데이터 크기 제한
documents = load_korean_datasets(
    wiki_limit=5000,  # 더 작은 값으로
    namuwiki_limit=2000,
)
```

## 참고 자료

- [opensearch-sparse-model-tuning-sample](https://github.com/zhichao-aws/opensearch-sparse-model-tuning-sample)
- [논문: Towards Competitive Search Relevance For Inference-Free Learned Sparse Retrievers](../sparse-retriever.pdf)
- [intfloat/multilingual-e5-large 모델](https://huggingface.co/intfloat/multilingual-e5-large)
- [OpenSearch Neural Sparse 문서](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/)

## 라이선스

이 프로젝트는 Apache 2.0 라이선스를 따릅니다.
