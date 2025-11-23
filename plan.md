# OpenSearch Neural Sparse Model 학습 데이터 준비 계획

## 목표
한국어 공개 데이터셋을 활용하여 OpenSearch neural sparse 모델 학습에 사용할 수 있는 JSONL 형식의 데이터셋 생성

## 데이터 포맷 요구사항

### opensearch-sparse-model-tuning-sample 레포지토리 데이터 포맷

**Pre-computed Knowledge Distillation 형식** (권장):
```json
{
    "query": "질문 텍스트",
    "docs": ["관련 문서1", "관련 문서2", ...],
    "scores": [9.0, 7.5, 5.0, ...]
}
```

- `query`: 질의 텍스트
- `docs`: 후보 문서 리스트
- `scores`: 각 문서에 대한 관련성 점수 (teacher 모델에서 생성)

## 한국어 데이터셋

### 이미 확보된 데이터셋
- [x] 한국어 Wikipedia (~12 chunk 파일)
- [x] Namuwiki (~13 chunk 파일)

### 데이터 구조
```json
{
    "id": "문서ID",
    "url": "원본 URL",
    "title": "문서 제목",
    "text": "본문 내용",
    "language": "ko"
}
```

## 작업 단계

### 1. 환경 설정 ✓
- [x] .venv 가상환경 확인
- [x] 필요 라이브러리 확인 (transformers, sentence-transformers, scikit-learn, faiss-cpu)

### 2. 데이터 로딩 및 전처리
- [ ] Wikipedia + Namuwiki 데이터 로드
- [ ] 텍스트 정제 (너무 짧거나 긴 문서 필터링)
- [ ] 문서 청킹 (필요시 긴 문서를 적절한 길이로 분할)

### 3. 쿼리 생성
프로젝트 논문에서 제안된 방법:
- [ ] **방법 1**: 문서 제목을 query로 사용, 본문을 document로 사용
- [ ] **방법 2**: 문서의 첫 문장/요약문을 query로 추출
- [ ] **방법 3**: LLM 기반 질문 생성 (선택적)

### 4. Embedding 생성 (eintfloat/multilingual-e5-large)
- [ ] multilingual-e5-large 모델 로드
- [ ] 모든 문서에 대한 embedding 생성
- [ ] 모든 query에 대한 embedding 생성
- [ ] Embedding 저장 (FAISS 인덱스 또는 numpy array)

### 5. Hard Negatives Mining
- [ ] FAISS를 사용한 유사도 기반 문서 검색
- [ ] 각 query에 대해:
  - Positive document: 원본 문서 (score: 10.0)
  - Hard negatives: 유사하지만 관련 없는 문서 Top-K (score: 유사도 기반)
  - Easy negatives: 랜덤 샘플링 (score: 0.0 ~ 1.0)

### 6. Synonym/Related Terms 추출 (k-means clustering)
- [ ] 문서 embedding을 k-means로 클러스터링
- [ ] 같은 클러스터 내 문서들을 관련 문서로 간주
- [ ] 클러스터 중심과의 거리를 score로 변환

### 7. 최종 데이터셋 생성
- [ ] JSONL 포맷으로 저장:
  ```json
  {"query": "...", "docs": [...], "scores": [...]}
  ```
- [ ] Train/Val split (90/10)
- [ ] 데이터셋 통계 정보 생성

### 8. 검증
- [ ] 샘플 데이터 확인
- [ ] 포맷 검증
- [ ] opensearch-sparse-model-tuning-sample 학습 코드와 호환성 확인

## 예상 출력

### 디렉토리 구조
```
dataset/
├── neural_sparse_training/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── embeddings/
│   │   ├── document_embeddings.npy
│   │   ├── query_embeddings.npy
│   │   └── faiss_index.bin
│   └── metadata.json
```

### 메타데이터
```json
{
    "total_queries": 50000,
    "train_samples": 45000,
    "val_samples": 5000,
    "docs_per_query": 8,
    "embedding_model": "intfloat/multilingual-e5-large",
    "source_datasets": ["wikipedia_ko", "namuwiki"],
    "created_at": "2025-11-23"
}
```

## 노트북 파일

### notebooks/opensearch-neural-v2/01_data_preparation_neural_sparse.ipynb
주요 섹션:
1. 환경 설정
2. 데이터 로딩
3. Query 생성
4. Embedding 생성 (multilingual-e5-large)
5. Hard Negatives Mining
6. K-means 클러스터링
7. 최종 JSONL 생성
8. 데이터 검증

## 기술 스택

- Python 3.12
- sentence-transformers (intfloat/multilingual-e5-large)
- scikit-learn (k-means)
- faiss-cpu (유사도 검색)
- transformers
- numpy, pandas

## 주의사항

- Embedding 생성은 GPU 사용 권장 (시간 단축)
- FAISS 인덱스 빌드 시 메모리 사용량 주의
- 대규모 데이터 처리 시 배치 단위로 처리
- Type hints 필수
- Docstrings 필수
- 중간 결과 저장 (재실행 시간 단축)

## 성공 기준

- [ ] 최소 10,000개 이상의 학습 샘플 생성
- [ ] opensearch-sparse-model-tuning-sample 포맷과 100% 호환
- [ ] 각 query당 positive 1개 + hard negatives 5~7개 포함
- [ ] Score 분포가 합리적 (positive > hard negatives > easy negatives)
- [ ] 노트북 실행 시간 < 2시간 (전체 데이터 기준)
