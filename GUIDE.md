# Neural Sparse Model Training Guide

Korean Neural Sparse Encoder 학습을 위한 단계별 가이드입니다.

## 목차

- [Prerequisites](#prerequisites)
- [Version Overview](#version-overview)
- [Quick Start](#quick-start)
- [V28 Training (Latest)](#v28-training-latest)
- [Step-by-Step Guide](#step-by-step-guide)
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
│   ├── train_v26.yaml
│   ├── train_v27.yaml
│   └── train_v28.yaml        # 최신 설정
├── outputs/
│   ├── idf_weights/          # IDF 가중치 저장
│   ├── train_v27/            # V27 학습 출력
│   └── train_v28/            # V28 학습 출력
└── huggingface/
    └── v28/                  # HuggingFace 변환 모델
```

---

## Version Overview

| Version | Key Features | Status | Command |
|---------|--------------|--------|---------|
| V22 | KoBERT backbone, curriculum learning | Completed | `make train-v22` |
| V24 | XLM-RoBERTa + BGE-M3 teacher | Completed | `make train-v24` |
| V25 | IDF-aware FLOPS | Completed | `make train-v25` |
| V26 | Enhanced IDF + Special Token Fix | Completed | `make train-v26` |
| V27 | Travel Domain Data | Training | `make train-v27` |
| **V28** | **Korean Filter + Context Gate** | **Ready** | `make train-v28` |

---

## Quick Start

### 자동 파이프라인 (권장)

```bash
# V28 전체 파이프라인
make v28-pipeline

# 백그라운드 학습 시작
make train-v28-bg
```

### V27 완료 후 자동 시작

```bash
nohup ./scripts/run_v28_after_v27.sh > outputs/v28_auto.log 2>&1 &
```

---

## V28 Training (Latest)

V28은 두 가지 핵심 기능을 포함합니다:

### V28a: Korean Language Filtering

- 비한국어 토큰 억제 (`non_korean_penalty=100.0`)
- 목표: 한국어 토큰 비율 **>85%**
- 문제 해결: 다국어 토큰 누출 (기존 91% → 목표 <5%)

### V28b: Context-Gated Sparse Expansion (CGSE)

- 문맥 의존적 토큰 활성화
- Multi-head attention 기반 context pooling
- 목표: 컨텍스트 구분율 **>60%**

**예시:**
```
기존 (V26): "점심 메뉴" → 항상 동일한 토큰 활성화

V28b 목표:
- "출근했는데 점심 메뉴" → 회사, 직장인, 비빔밥
- "학교를 갔는데 점심 메뉴" → 학생, 급식, 도시락
```

### V28 Config 주요 설정

```yaml
# configs/train_v28.yaml
model:
  model_class: "SPLADEDocContextGated"
  use_context_gate: true
  context_gate_hidden: 256
  context_attention_heads: 4

loss:
  # V26 기반
  lambda_flops: 0.010
  special_token_penalty: 100.0
  stopword_penalty: 15.0

  # V28a: 언어 필터링
  enable_language_filtering: true
  non_korean_penalty: 100.0
  lambda_language: 0.5

  # V28b: 컨텍스트 게이트
  use_context_gate: true
  use_context_aware_kd: true
```

---

## Step-by-Step Guide

### Phase 1: 데이터 전처리

```bash
# IDF 가중치 계산
make prepare-v25-idf

# Korean Stopword 마스크 생성
make prepare-v25-stopwords

# 한국어 토큰 ID 빌드 (V28용)
make build-korean-tokens
```

### Phase 2: 검증

```bash
# IDF 설정 검증
make train-v28-verify

# Quick Training 검증 (500 samples, 2 epochs)
make train-v28-quick
```

### Phase 3: 본 학습

```bash
# Foreground 실행
make train-v28

# Background 실행 (권장)
make train-v28-bg

# 학습 재개 (Checkpoint)
make train-v28-resume
```

### Phase 4: 평가

```bash
# 한국어 토큰 비율 검증
make eval-v28-language

# 컨텍스트 구분율 검증
make eval-v28-context

# Ko-StrategyQA 벤치마크
make benchmark-ko-strategyqa-v28
```

### Phase 5: 모델 변환 및 배포

```bash
# HuggingFace 형식 변환
make convert-v28-hf

# HuggingFace Hub 업로드
huggingface-cli upload sewoong/korean-neural-sparse-encoder huggingface/v28
```

---

## Monitoring

### 실시간 로그 확인

```bash
make logs-v28
```

### TensorBoard

```bash
make tensorboard-v28
# URL: http://localhost:6006
```

### GPU 모니터링

```bash
make monitor
```

---

## Command Reference

| 명령어 | 설명 |
|--------|------|
| `make train-v28` | V28 학습 (foreground) |
| `make train-v28-bg` | V28 학습 (background) |
| `make train-v28-resume` | 학습 재개 |
| `make eval-v28` | 모델 평가 |
| `make eval-v28-language` | 한국어 토큰 비율 |
| `make eval-v28-context` | 컨텍스트 구분율 |
| `make logs-v28` | 로그 확인 |
| `make tensorboard-v28` | TensorBoard 시작 |
| `make v28-pipeline` | 자동 파이프라인 |

---

## Troubleshooting

### IDF weights not found 오류

```bash
make prepare-v25-idf
```

### CUDA Out of Memory

1. `configs/train_v28.yaml`에서 `batch_size` 감소 (24 → 16)
2. `gradient_accumulation_steps` 증가 (4 → 8)

### 한국어 토큰 비율이 낮음 (<85%)

1. `non_korean_penalty` 증가 (100.0 → 150.0)
2. `lambda_language` 증가 (0.5 → 1.0)

### Context Gate 효과가 없음

1. `context_gate_hidden` 증가 (256 → 512)
2. 학습 epoch 증가 (25 → 30)

---

## Version Comparison

| 항목 | V25 | V26 | V28 |
|------|-----|-----|-----|
| IDF 가중치 | 필수 | 필수 + 특수 토큰 분리 | 필수 |
| Stopword 마스킹 | 163개 | 확장 | 확장 |
| 언어 필터링 | 없음 | 없음 | **한국어 집중** |
| Context Gate | 없음 | 없음 | **CGSE** |
| 목표 | 의미 토큰 강화 | 특수 토큰 억제 | **문맥 인식** |

---

## 다음 단계

1. 모델 배포: HuggingFace Hub 업로드
2. OpenSearch 통합: Neural Sparse 플러그인 설정
3. 성능 벤치마크: Ko-StrategyQA Recall@1 > 40% 달성
