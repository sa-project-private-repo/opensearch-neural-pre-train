# V25 Training Guide

Korean Neural Sparse Encoder V25 학습을 위한 단계별 가이드입니다.

## 목차

- [Prerequisites](#prerequisites)
- [Quick Start (자동 파이프라인)](#quick-start-자동-파이프라인)
- [Step-by-Step Guide (수동 실행)](#step-by-step-guide-수동-실행)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. 환경 설정

```bash
# 가상환경 생성 (최초 1회)
python3 -m venv .venv

# 환경 확인
make setup
```

### 2. 필수 디렉토리 구조

```
opensearch-neural-pre-train/
├── data/
│   └── v24.0/
│       └── train_*.jsonl     # 학습 데이터
├── configs/
│   └── train_v25.yaml        # V25 설정
├── outputs/
│   ├── idf_weights/          # IDF 가중치 저장
│   └── train_v25/            # 학습 출력
└── huggingface/
    └── v25/                  # HuggingFace 변환 모델
```

---

## Quick Start (자동 파이프라인)

전체 파이프라인을 한 번에 실행:

```bash
make v25-pipeline
```

이 명령은 다음을 순서대로 수행합니다:
1. IDF 가중치 계산
2. Stopword 마스크 생성
3. Quick validation (500 samples)

검증 완료 후 본 학습 시작:
```bash
make train-v25-bg
```

---

## Step-by-Step Guide (수동 실행)

### Phase 1: 데이터 전처리

#### Step 1.1: IDF 가중치 계산

학습 코퍼스에서 BM25 스타일 IDF 가중치를 계산합니다.

```bash
make prepare-v25-idf
```

**출력:**
- `outputs/idf_weights/xlmr_v25_idf.pt`

**소요 시간:** ~5분 (100K 샘플 기준)

#### Step 1.2: Korean Stopword 마스크 생성

한국어 조사/어미 토큰에 대한 마스크를 생성합니다.

```bash
make prepare-v25-stopwords
```

**출력:**
- `outputs/idf_weights/xlmr_stopword_mask.pt`
- 163개 stopword 토큰 마스킹

#### Step 1.3: 전처리 통합 실행 (선택)

위 두 단계를 한 번에 실행:

```bash
make prepare-v25-data
```

---

### Phase 2: 검증

#### Step 2.1: IDF 설정 검증

학습 없이 IDF 및 Stopword 설정만 확인:

```bash
make train-v25-verify
```

**확인 사항:**
- IDF 가중치 로딩 성공
- Stopword 마스크 적용
- Loss 함수 초기화

#### Step 2.2: Quick Training 검증

소규모 데이터로 학습 파이프라인 전체 검증:

```bash
make train-v25-quick
```

**설정:**
- Samples: 500
- Epochs: 2
- 목적: 전체 학습 루프 검증

**확인 사항:**
- Semantic ratio > 1.0 (의미 토큰이 stopword보다 높은 활성화)
- Loss 감소 추세
- GPU 메모리 사용량

---

### Phase 3: 본 학습

#### Step 3.1: 학습 시작 (Foreground)

```bash
make train-v25
```

#### Step 3.2: 학습 시작 (Background) - 권장

```bash
make train-v25-bg
```

**출력 정보:**
- PID 출력
- 로그 파일 경로: `outputs/train_v25/nohup.out`

#### Step 3.3: 학습 재개 (Checkpoint에서)

학습이 중단된 경우:

```bash
make train-v25-resume
```

---

### Phase 4: 평가

#### Step 4.1: 모델 평가

```bash
make eval-v25
```

#### Step 4.2: Sparsity 분석

Semantic vs Stopword 토큰 활성화 비율 분석:

```bash
make eval-v25-sparsity
```

**목표 지표:**
- Semantic tokens in top-10: 80%+
- Stopword activation: < 0.5

#### Step 4.3: V24 vs V25 비교

```bash
make eval-v25-compare
```

---

### Phase 5: 모델 변환 및 배포

#### Step 5.1: HuggingFace 형식 변환

```bash
make convert-v25-hf
```

**출력:**
- `huggingface/v25/best/`

---

## Monitoring

### 실시간 로그 확인

```bash
make logs-v25
```

### TensorBoard

```bash
make tensorboard-v25
# URL: http://localhost:6006
```

### GPU 모니터링

```bash
make monitor
```

---

## Command Reference

| 명령어 | 설명 | 단계 |
|--------|------|------|
| `make setup` | 환경 확인 | 사전 준비 |
| `make prepare-v25-idf` | IDF 가중치 계산 | Phase 1 |
| `make prepare-v25-stopwords` | Stopword 마스크 생성 | Phase 1 |
| `make prepare-v25-data` | 전처리 통합 실행 | Phase 1 |
| `make train-v25-verify` | IDF 설정 검증 | Phase 2 |
| `make train-v25-quick` | Quick training 검증 | Phase 2 |
| `make train-v25` | 본 학습 (foreground) | Phase 3 |
| `make train-v25-bg` | 본 학습 (background) | Phase 3 |
| `make train-v25-resume` | 학습 재개 | Phase 3 |
| `make eval-v25` | 모델 평가 | Phase 4 |
| `make eval-v25-sparsity` | Sparsity 분석 | Phase 4 |
| `make eval-v25-compare` | V24 vs V25 비교 | Phase 4 |
| `make convert-v25-hf` | HuggingFace 변환 | Phase 5 |
| `make logs-v25` | 로그 확인 | Monitoring |
| `make tensorboard-v25` | TensorBoard 시작 | Monitoring |
| `make v25-pipeline` | 자동 파이프라인 | Quick Start |

---

## Troubleshooting

### IDF weights not found 오류

```
Error: IDF weights not found. Run: make prepare-v25-idf
```

**해결:**
```bash
make prepare-v25-idf
```

### CUDA Out of Memory

**해결:**
1. `configs/train_v25.yaml`에서 `batch_size` 감소
2. `gradient_accumulation_steps` 증가

### Stopword ratio가 너무 높음

Semantic ratio < 1.0인 경우:

**해결:**
1. `configs/train_v25.yaml`에서 `stopword_penalty` 증가 (기본: 5.0)
2. `lambda_flops` 증가

### 학습이 느림

**해결:**
1. Mixed precision 확인: `bf16: true`
2. DataLoader `num_workers` 증가
3. `pin_memory: true` 설정

---

## V25 vs V24 차이점

| 항목 | V24 | V25 |
|------|-----|-----|
| IDF 가중치 | 설정만 존재 (미적용) | 필수 적용 |
| Stopword 마스킹 | 없음 | 163개 토큰 마스킹 |
| FLOPS Loss | 균일 페널티 | IDF-aware 페널티 |
| 목표 | 기본 학습 | 의미 토큰 강화 |

---

## 다음 단계

1. 모델 배포: HuggingFace Hub 업로드
2. OpenSearch 통합: Neural Sparse 플러그인 설정
3. 성능 벤치마크: 실제 검색 쿼리 평가
