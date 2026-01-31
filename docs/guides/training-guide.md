# Korean Neural Sparse Model Training Guide

OpenSearch Neural Sparse Encoding 모델 학습 가이드

## 목차

1. [환경 설정](#환경-설정)
2. [데이터 준비](#데이터-준비)
3. [IDF 가중치 계산](#idf-가중치-계산)
4. [Training 실행](#training-실행)
5. [버전별 학습](#버전별-학습)
6. [Max Length 192 제약](#max-length-192-제약)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## 환경 설정

### 시스템 요구사항

- Python 3.12+
- CUDA 11.8+
- PyTorch 2.0+
- GPU 메모리: 최소 16GB (V100/A100 권장)

### Virtual Environment 설정

```bash
# venv 생성
python3.12 -m venv venv

# venv 활성화
source venv/bin/activate

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
```

### 주요 의존성

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
tensorboard>=2.13.0
pyyaml>=6.0
tqdm>=4.65.0
```

---

## 데이터 준비

### Triplet Format

Neural Sparse 모델은 triplet 형식의 데이터로 학습합니다.

```json
{
  "query": "서울 맛집 추천",
  "positive": "서울 강남구 신사동 맛집 베스트 10",
  "negative": "부산 해운대 카페 추천"
}
```

- **query**: 검색 쿼리
- **positive**: query와 관련성 높은 문서
- **negative**: query와 관련성 낮은 문서

### JSONL 파일 구조

```
data/v27.0/processed/
├── train.jsonl          # 학습 데이터
├── val.jsonl            # 검증 데이터
└── test.jsonl           # 테스트 데이터
```

각 줄은 하나의 triplet JSON 객체입니다.

```bash
# 예시 데이터 확인
head -n 1 data/v27.0/processed/train.jsonl | jq .
```

### Curriculum Learning 데이터 분류

Curriculum learning을 위해 데이터를 난이도별로 분류합니다.

| 단계 | 설명 | 예시 |
|------|------|------|
| **Easy** | 키워드 매칭이 명확 | "서울 날씨" → "서울 기온 정보" |
| **Medium** | 의미적 유사성 필요 | "저렴한 노트북" → "가성비 좋은 랩탑" |
| **Hard** | 복잡한 의미 추론 | "재택근무 장비" → "홈오피스 필수품 가이드" |

```python
# scripts/classify_difficulty.py
def classify_difficulty(triplet):
    query_tokens = set(triplet['query'].split())
    pos_tokens = set(triplet['positive'].split())
    overlap = len(query_tokens & pos_tokens) / len(query_tokens)

    if overlap > 0.5:
        return 'easy'
    elif overlap > 0.2:
        return 'medium'
    else:
        return 'hard'
```

---

## IDF 가중치 계산

### BM25 Smoothing 공식

IDF 가중치는 다음 공식으로 계산됩니다.

```
idf(term) = log((N - df + 0.5) / (df + 0.5) + 1)
```

- **N**: 전체 문서 수
- **df**: term이 출현하는 문서 수
- **+0.5**: Smoothing 상수 (희귀 단어 과대평가 방지)

### 계산 및 캐시

```bash
# IDF 가중치 계산
python -m src.preprocessing.idf_calculator \
  --corpus data/v27.0/processed/train.jsonl \
  --output data/v27.0/idf_weights.json
```

**캐시 경로**:
- `data/v{version}/idf_weights.json`
- `data/v{version}/idf_cache/`

학습 시 자동으로 로드되며, 없으면 자동 계산됩니다.

### IDF 가중치 확인

```python
import json

with open('data/v27.0/idf_weights.json') as f:
    idf = json.load(f)

# 상위 10개 희귀 단어
sorted_terms = sorted(idf.items(), key=lambda x: x[1], reverse=True)
print(sorted_terms[:10])
```

---

## Training 실행

### 기본 실행

```bash
# Makefile 사용 (권장)
make train-v27

# 직접 실행
python -m src.train.cli.train_v27 --config configs/train_v27.yaml
```

### Config 파일 예시

```yaml
# configs/train_v27.yaml
model:
  name: "xlm-roberta-base"
  max_length: 192  # IMPORTANT: 반드시 192 이하
  pooling: "mean"

training:
  batch_size: 32
  gradient_accumulation_steps: 4
  learning_rate: 2e-5
  num_epochs: 10
  warmup_steps: 500

  # Curriculum learning
  curriculum:
    enabled: true
    stages:
      - epoch: [0, 3]
        difficulty: "easy"
      - epoch: [3, 6]
        difficulty: ["easy", "medium"]
      - epoch: [6, 10]
        difficulty: ["easy", "medium", "hard"]

loss:
  type: "contrastive"
  margin: 0.3
  temperature: 0.05

data:
  train: "data/v27.0/processed/train.jsonl"
  val: "data/v27.0/processed/val.jsonl"
  idf_weights: "data/v27.0/idf_weights.json"

output:
  dir: "models/v27.0"
  checkpoint_every: 1000
  save_total_limit: 3

logging:
  tensorboard_dir: "runs/v27.0"
  log_every: 100
```

### 학습 모니터링

```bash
# TensorBoard 실행
tensorboard --logdir runs/v27.0 --port 6006

# 백그라운드 실행
nohup tensorboard --logdir runs/v27.0 --port 6006 > tensorboard.log 2>&1 &
```

### 분산 학습 (Multi-GPU)

```bash
# 2개 GPU 사용
torchrun --nproc_per_node=2 \
  -m src.train.cli.train_v27 \
  --config configs/train_v27.yaml
```

---

## 버전별 학습

### Version Overview

| Version | Command | Key Features |
|---------|---------|--------------|
| V22 | `make train-v22` | KoBERT backbone, curriculum learning |
| V24 | `make train-v24` | XLM-RoBERTa + BGE-M3 teacher |
| V25 | `make train-v25` | IDF-aware FLOPS |
| V26 | `make train-v26` | Enhanced IDF + Special Token Fix |
| V27 | `make train-v27` | Travel Domain Data |
| **V28** | `make train-v28` | **Korean Filter + Context Gate** |

### V28 Training (Latest)

V28은 두 가지 핵심 기능을 포함합니다:

**V28a: Korean Language Filtering**
- 비한국어 토큰 억제 (non_korean_penalty=100.0)
- 목표: 한국어 토큰 비율 >85%

**V28b: Context-Gated Sparse Expansion (CGSE)**
- 문맥 의존적 토큰 활성화
- Multi-head attention 기반 context pooling
- 목표: 컨텍스트 구분율 >60%

```bash
# V28 학습
make train-v28

# V27 완료 후 자동 시작
nohup ./scripts/run_v28_after_v27.sh > outputs/v28_auto.log 2>&1 &

# V28 검증
make eval-v28-language   # 한국어 토큰 비율
make eval-v28-context    # 컨텍스트 구분율
```

### V28 Config 주요 설정

```yaml
# configs/train_v28.yaml
model:
  model_class: "SPLADEDocContextGated"
  use_context_gate: true
  context_gate_hidden: 256

loss:
  # Korean language filtering
  enable_language_filtering: true
  non_korean_penalty: 100.0
  lambda_language: 0.5

  # Context gate
  use_context_gate: true
  use_context_aware_kd: true
```

---

## Max Length 192 제약

### 중요: Position Embedding 제한

XLM-RoBERTa 모델은 **최대 512 토큰**까지 지원하지만, OpenSearch Neural Sparse 플러그인은 **192 토큰 제한**이 있습니다.

| 모델 | 최대 길이 | OpenSearch 제한 |
|------|-----------|----------------|
| XLM-RoBERTa | 512 | 192 |
| BERT | 512 | 192 |
| DistilBERT | 512 | 192 |

### 긴 문서 처리: Truncation

기본 전략은 **simple truncation**입니다.

```python
# src/train/tokenizer.py
def tokenize(text, max_length=192):
    tokens = tokenizer(
        text,
        max_length=max_length,
        truncation=True,  # 192 토큰 초과 시 자름
        padding="max_length",
        return_tensors="pt"
    )
    return tokens
```

**장점**: 간단하고 빠름
**단점**: 문서 후반부 정보 손실

### 대안 1: Chunking

문서를 여러 chunk로 분할하여 처리합니다.

```python
def chunk_document(text, max_length=192, overlap=50):
    tokens = tokenizer.tokenize(text)
    chunks = []

    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))

    return chunks
```

**장점**: 전체 문서 활용
**단점**: 인덱싱 및 검색 복잡도 증가

### 대안 2: Sliding Window

Overlap을 가진 sliding window로 처리합니다.

```python
def sliding_window_encode(text, max_length=192, stride=128):
    encodings = tokenizer(
        text,
        max_length=max_length,
        stride=stride,
        truncation=True,
        return_overflowing_tokens=True,
        return_tensors="pt"
    )

    # 여러 window의 임베딩 평균
    embeddings = model(**encodings).last_hidden_state.mean(dim=0)
    return embeddings
```

**장점**: 문맥 보존, 부드러운 전환
**단점**: 계산 비용 증가

### 권장사항

| 문서 길이 | 권장 전략 | 비고 |
|-----------|----------|------|
| < 192 토큰 | Truncation | 추가 처리 불필요 |
| 192-512 토큰 | Chunking (2-3 chunks) | 적당한 오버헤드 |
| > 512 토큰 | Sliding Window | 최대 품질 |

---

## Monitoring

### TensorBoard 주요 지표

| 지표 | 설명 | 목표 |
|------|------|------|
| `loss/train` | 학습 손실 | 지속적 감소 |
| `loss/val` | 검증 손실 | Overfitting 감지 |
| `metrics/semantic_ratio` | 의미적 토큰 비율 | > 0.6 |
| `metrics/sparsity` | Sparse 벡터 희소성 | 0.95-0.99 |
| `learning_rate` | 학습률 스케줄 | Warmup 후 감소 |

### Semantic Ratio 추적

Semantic ratio는 모델이 의미 있는 토큰에 집중하는 정도를 나타냅니다.

```python
def calculate_semantic_ratio(logits, threshold=0.01):
    active_tokens = (logits > threshold).sum()
    total_tokens = logits.numel()
    return active_tokens / total_tokens
```

**해석**:
- **< 0.5**: 너무 sparse, 정보 손실 가능성
- **0.6-0.8**: 이상적
- **> 0.9**: 너무 dense, sparse 효과 감소

### 체크포인트 검증

```bash
# 특정 체크포인트 평가
python -m src.evaluation.evaluate \
  --checkpoint models/v27.0/checkpoint-5000 \
  --test-data data/v27.0/processed/test.jsonl
```

---

## Troubleshooting

### 1. OOM (Out of Memory) 해결

**증상**: CUDA out of memory 에러

**해결책**:

```yaml
# configs/train_v27.yaml
training:
  batch_size: 16          # 32 → 16
  gradient_accumulation_steps: 8  # 4 → 8 (effective batch 유지)
  mixed_precision: true   # FP16 학습 활성화
```

```bash
# Gradient checkpointing 활성화
python -m src.train.cli.train_v27 \
  --config configs/train_v27.yaml \
  --gradient-checkpointing
```

**추가 옵션**:
- Max length 감소: 192 → 128
- Dataloader workers 감소: 4 → 2

### 2. Low Semantic Ratio

**증상**: semantic_ratio < 0.5

**원인**:
- 학습률 너무 높음
- Temperature 너무 낮음
- 데이터 품질 문제

**해결책**:

```yaml
training:
  learning_rate: 1e-5  # 2e-5 → 1e-5

loss:
  temperature: 0.1  # 0.05 → 0.1 (더 soft한 분포)
```

**데이터 검증**:

```bash
# Triplet 품질 검증
python scripts/validate_triplets.py \
  --data data/v27.0/processed/train.jsonl \
  --min-positive-score 0.7
```

### 3. Loss 정체

**증상**: Loss가 특정 값에서 감소하지 않음

**원인**:
- Learning rate 너무 낮음
- Warmup 부족
- 데이터 편향

**해결책**:

```yaml
training:
  learning_rate: 3e-5  # 증가
  warmup_steps: 1000   # 500 → 1000

  # Learning rate scheduler
  scheduler:
    type: "cosine"
    num_cycles: 0.5
```

**Curriculum learning 재검토**:

```yaml
curriculum:
  stages:
    - epoch: [0, 2]      # Easy 단계 단축
      difficulty: "easy"
    - epoch: [2, 10]     # Medium/Hard 조기 도입
      difficulty: ["easy", "medium", "hard"]
```

### 4. Sparsity 너무 높음 (> 0.99)

**증상**: 거의 모든 토큰이 0

**원인**:
- Activation threshold 너무 높음
- 학습 부족

**해결책**:

```yaml
model:
  activation_threshold: 0.005  # 0.01 → 0.005

training:
  num_epochs: 15  # 10 → 15 (더 긴 학습)
```

### 5. Validation Loss 증가 (Overfitting)

**증상**: Train loss는 감소하지만 val loss는 증가

**해결책**:

```yaml
training:
  dropout: 0.2           # Dropout 추가
  weight_decay: 0.01     # L2 regularization
  early_stopping:
    patience: 3
    min_delta: 0.001
```

**Data augmentation**:

```python
# 쿼리 augmentation
def augment_query(query):
    # 동의어 치환, 순서 변경 등
    return augmented_query
```

---

## 추가 리소스

- [OpenSearch Neural Sparse Documentation](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/)
- [Hugging Face Model](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1)
- [Sample Training Code](https://github.com/zhichao-aws/opensearch-sparse-model-tuning-sample)
- [Sparse Retrieval Paper](./sparse-retriever.pdf)

---

## 라이선스

Apache 2.0

## 기여

이슈 및 PR은 GitHub 저장소로 제출해주세요.
