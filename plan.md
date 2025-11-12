# OpenSearch Neural Sparse Pre-training 개선 계획

## 프로젝트 목표

뉴스 데이터를 활용하여 시간 가중치 기반 군집화로 동의어를 발견하고, 비지도 학습 방식으로 한국어 Neural Sparse 검색 모델을 개선합니다.

## 발견된 주요 문제점

### 🔴 CRITICAL 문제
- [ ] **손실 함수 오류**: BCE with logits가 dot product similarity와 맞지 않음
- [ ] **Contrastive Learning 부재**: Query-document ranking을 제대로 학습하지 못함
- [ ] **시간 정보 미활용**: 뉴스 데이터의 날짜 정보를 전혀 사용하지 않음

### 🟡 MODERATE 문제
- [ ] **약한 Negative Sampling**: 랜덤 negative 2개만 사용
- [ ] **학습 데이터 부족**: 10k pairs로는 부족
- [ ] **하드코딩된 트렌드 키워드**: 수동으로 정의된 TREND_BOOST

### 🟢 MINOR 문제
- [ ] **과도한 Sparsity**: L0 regularization이 너무 강함 (99.98%)
- [ ] **IDF penalty 모순**: Document encoder가 IDF statistics에 의존

---

## Phase 1: 손실 함수 및 학습 파이프라인 수정 (최우선)

### 1.1 새로운 손실 함수 구현
- [ ] In-batch negatives loss 함수 작성
  - File: `src/losses.py` (신규)
  - 함수: `in_batch_negatives_loss()`, `margin_ranking_loss()`
  - Temperature scaling 파라미터 추가

- [ ] Contrastive loss with hard negatives
  - 함수: `contrastive_loss_with_hard_negatives()`
  - Triplet margin loss 옵션 추가

### 1.2 학습 파이프라인 수정
- [ ] Batch size 증가 (8 → 32 or 64)
  - korean_neural_sparse_training.ipynb 수정
  - GPU 메모리 확인 및 최적화

- [ ] 손실 함수 교체
  - BCE → In-batch negatives loss
  - 하이퍼파라미터 조정 (temperature=0.05)

### 1.3 테스트 및 검증
- [ ] test_korean_neural_sparse.py 업데이트
  - 새로운 손실 함수로 교체
  - 학습 스크립트 검증

- [ ] 간단한 학습 실행 및 loss curve 확인

---

## Phase 2: 데이터 로딩 및 시간 기반 분석 구현

### 2.1 뉴스 데이터 로딩 개선
- [ ] 날짜 정보 추출 및 보존
  - File: `src/data_loader.py` (신규)
  - 함수: `load_korean_news_with_dates()`
  - 데이터 구조: `{text, date, category, ...}`

- [ ] 데이터셋 다각화
  - HuggingFace 뉴스 데이터셋 추가 탐색
  - RSS 피드 크롤러 구현 (선택)
  - 최소 50k+ documents 확보

### 2.2 시간 가중치 기반 IDF 구현
- [ ] Temporal IDF 함수 작성
  - File: `src/temporal_analysis.py` (신규)
  - 함수: `calculate_temporal_idf(documents, dates, decay_factor=0.95)`
  - Exponential decay: weight = decay_factor^(days_old)

- [ ] 시간 윈도우별 IDF 계산
  - 함수: `calculate_windowed_idf(documents, dates, window_days=[30, 90, 365])`
  - 여러 시간대의 IDF를 앙상블

### 2.3 자동 트렌드 감지 구현
- [ ] 트렌드 토큰 자동 발견
  - 함수: `detect_trending_tokens(documents, dates, recent_days=30)`
  - 최근 빈도 vs 과거 빈도 비교
  - TREND_BOOST 딕셔너리 자동 생성

- [ ] 시간대별 단어 빈도 분석
  - 함수: `analyze_token_frequency_over_time(documents, dates, tokens)`
  - 시계열 데이터로 저장

---

## Phase 3: 하드코딩 제거 및 비지도 학습 강화

### 3.1 Hard Negative Mining 구현
- [ ] BM25 기반 hard negatives
  - File: `src/negative_sampling.py` (신규)
  - 함수: `add_hard_negatives_bm25(qd_pairs, documents, top_k=100)`
  - rank-bm25 라이브러리 활용

- [ ] Negative sampling 전략 개선
  - Random negatives: 50%
  - Hard negatives: 50%
  - Negatives per query: 2 → 8+

### 3.2 하드코딩된 요소 제거
- [ ] SAMPLE_DOCUMENTS → 실제 데이터 샘플링으로 교체
  - test_korean_neural_sparse.py 수정
  - 랜덤 샘플링 함수 사용

- [ ] TREND_BOOST → 자동 감지로 교체
  - temporal_analysis.py의 detect_trending_tokens() 활용
  - 동적 부스팅 팩터 계산

- [ ] ai_domain_terminology.py 사용 방식 변경
  - 참고용/검증용으로만 사용
  - 자동 발견된 동의어 우선 적용

### 3.3 데이터 증강 개선
- [ ] Synonym-based augmentation 강화
  - 현재 expansion_ratio=0.2 → 0.5로 증가
  - 시간 기반 동의어 활용

- [ ] Query generation
  - 문서에서 자동으로 query 생성
  - T5/BART 등 생성 모델 활용 (선택)

---

## Phase 4: 시간 가중치 기반 군집화 및 동의어 발견

### 4.1 시간 기반 토큰 임베딩 클러스터링
- [ ] 시간 윈도우별 임베딩 추출
  - File: `src/temporal_clustering.py` (신규)
  - 함수: `extract_temporal_embeddings(documents, dates, time_windows)`
  - 각 시간대별로 BERT 토큰 임베딩 추출

- [ ] 군집화 알고리즘 적용
  - 함수: `cluster_tokens_temporal(embeddings, method='kmeans', n_clusters=500)`
  - K-means, DBSCAN, Hierarchical clustering 옵션
  - 시간에 따른 군집 변화 추적

### 4.2 동의어 자동 발견 개선
- [ ] 시간 가중치 적용 동의어 발견
  - 함수: `discover_synonyms_temporal(documents, dates, embeddings, decay_factor=0.95)`
  - 최근 데이터에 높은 가중치
  - Cosine similarity threshold: 0.75

- [ ] 군집 기반 동의어 그룹 형성
  - 함수: `build_synonym_groups_from_clusters(clusters, embeddings, threshold=0.8)`
  - 각 클러스터를 동의어 그룹으로 간주
  - 신뢰도 점수 계산

### 4.3 LLM 기반 동의어 검증 (선택)
- [ ] 로컬 LLM 로딩
  - File: `src/llm_validator.py` (신규)
  - 모델: GPT-OSS-20B 또는 양자화된 120B
  - 4-bit quantization 적용

- [ ] 동의어 후보 검증
  - 함수: `validate_synonyms_with_llm(synonym_pairs, llm_model)`
  - Batch processing으로 효율성 확보
  - 검증 통과한 동의어만 사용

---

## Phase 5: 통합 및 테스트

### 5.1 전체 파이프라인 통합
- [ ] korean_neural_sparse_training.ipynb 전면 수정
  - 새로운 모듈들 import
  - 전체 플로우 재구성
  - 섹션별 설명 markdown 추가

- [ ] 설정 파일 작성
  - File: `config.yaml` (신규)
  - 모든 하이퍼파라미터 중앙화
  - 실험 설정 버전 관리

### 5.2 학습 스크립트 작성
- [ ] CLI 학습 스크립트
  - File: `train.py` (신규)
  - argparse로 파라미터 제어
  - 체크포인트 저장/로드 기능

- [ ] 평가 스크립트
  - File: `evaluate.py` (신규)
  - 검색 성능 평가 (MRR, NDCG, Recall@K)
  - Sparsity 분석

### 5.3 문서화 및 검증
- [ ] README 업데이트
  - 전체 워크플로우 설명
  - 실행 방법 가이드
  - 요구사항 및 설치

- [ ] 최종 테스트
  - End-to-end 학습 실행
  - 결과 분석 및 리포트 작성
  - 기존 모델 대비 성능 비교

---

## 디렉토리 구조 (예정)

```
opensearch-neural-pre-train/
├── src/
│   ├── __init__.py
│   ├── losses.py                    # Phase 1
│   ├── data_loader.py               # Phase 2
│   ├── temporal_analysis.py         # Phase 2
│   ├── negative_sampling.py         # Phase 3
│   ├── temporal_clustering.py       # Phase 4
│   └── llm_validator.py            # Phase 4 (선택)
├── config.yaml                      # Phase 5
├── train.py                         # Phase 5
├── evaluate.py                      # Phase 5
├── korean_neural_sparse_training.ipynb
├── neural_sparse_inference.ipynb
├── test_korean_neural_sparse.py
├── requirements.txt
└── README.md

```

---

## 실행 순서

### Stage 1: 긴급 수정 (1-2일)
1. Phase 1.1-1.2: 손실 함수 수정 및 테스트
2. Phase 1.3: 간단한 학습으로 검증

### Stage 2: 데이터 개선 (2-3일)
3. Phase 2.1: 뉴스 데이터 날짜 정보 추출
4. Phase 2.2: Temporal IDF 구현
5. Phase 2.3: 자동 트렌드 감지

### Stage 3: 비지도 학습 강화 (2-3일)
6. Phase 3.1: Hard negative mining
7. Phase 3.2: 하드코딩 제거
8. Phase 3.3: 데이터 증강

### Stage 4: 고급 기능 (3-4일)
9. Phase 4.1: 시간 기반 군집화
10. Phase 4.2: 동의어 자동 발견
11. Phase 4.3: LLM 검증 (선택)

### Stage 5: 통합 (1-2일)
12. Phase 5.1-5.3: 통합 및 문서화

---

## 성공 기준

- [ ] **손실 함수 문제 해결**: Loss가 정상적으로 감소
- [ ] **시간 정보 활용**: Temporal IDF가 작동하고 트렌드 감지 성공
- [ ] **하드코딩 제거**: 자동화된 트렌드 감지 및 동의어 발견
- [ ] **검색 성능 개선**: 기존 대비 MRR/NDCG 향상
- [ ] **비지도 학습 달성**: 수동 레이블 없이 동의어 발견
- [ ] **재현 가능성**: 전체 파이프라인이 config.yaml로 재현 가능

---

## 위험 요소 및 대응

| 위험 요소 | 대응 방안 |
|----------|----------|
| GPU 메모리 부족 (batch size 증가) | Gradient accumulation 사용 |
| 뉴스 데이터 날짜 정보 없음 | 대체 데이터셋 탐색 또는 크롤링 |
| LLM 로딩 실패 (메모리) | 4-bit quantization 또는 더 작은 모델 |
| 학습 시간 과다 | 데이터 샘플링 또는 분산 학습 |
| 성능 저하 | 하이퍼파라미터 튜닝 및 ablation study |

---

## 진행 상황 추적

- 각 체크박스 완료 시 `[x]`로 표시
- Git commit 시 conventional commits 규칙 준수
- 각 Phase 완료 시 테스트 및 검증 필수
