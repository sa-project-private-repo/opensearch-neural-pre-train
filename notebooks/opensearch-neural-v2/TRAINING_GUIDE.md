# OpenSearch Neural Sparse Model v2 Training Guide

이 가이드는 최신 연구 논문을 기반으로 한 OpenSearch Neural Sparse Model v2 학습 방법을 설명합니다.

## 기반 연구

**Paper**: "Towards Competitive Search Relevance For Inference-Free Learned Sparse Retrievers"
**Authors**: Zhichao Geng, Yiwen Wang, Dongyu Ru, Yang Yang (Amazon)
**arXiv**: 2411.04403v2 [cs.IR] 1 Jul 2025

### 핵심 개선사항

1. **IDF-Aware Penalty**: 중요한 저빈도 토큰을 보존하면서 불필요한 토큰 제거
2. **Heterogeneous Ensemble Knowledge Distillation**: Dense와 Sparse teacher 모델 결합

### 성능 결과 (BEIR Benchmark)
- SOTA inference-free model 대비 **+3.3 NDCG@10**
- Siamese 모델(SPLADE-v3-DistilBERT, ColBERTv2) 능가
- End-to-end latency: **BM25의 1.1배**

## 하드웨어 환경

### Nvidia DGX Spark 사양
```
GPU: Nvidia GB10
Compute Capability: 12.1
Driver Version: 580.95.05
CUDA: 13.0
Python: 3.12.3
PyTorch: 2.5.1
```

### 최적화 설정
- Mixed Precision Training (BF16)
- Gradient Accumulation
- PyTorch 2.0 Compile
- Efficient Data Loading

## 필수 패키지

`requirements_v2.txt` 참조:
```bash
torch==2.5.1
transformers==4.46.3
sentence-transformers==3.3.1
datasets==2.21.0
accelerate==1.1.1
# ... 기타 패키지
```

## 학습 파라미터

### Pre-training Configuration
```python
{
    "num_steps": 150000,
    "batch_size": 48,
    "learning_rate": 5e-5,
    "lambda_flops": 1e-7,  # 작은 값: 토큰 보존
    "num_hard_negatives": 7,
    "warmup_steps": 5000,
    "scale_constant_S": 10,
}
```

### Fine-tuning Configuration
```python
{
    "num_steps": 50000,
    "batch_size": 40,
    "learning_rate": 2e-5,
    "lambda_flops": 0.02,  # 큰 값: sparsity 조정
    "num_hard_negatives": 10,
    "warmup_steps": 2000,
    "scale_constant_S": 30,
}
```

## 주요 구현 세부사항

### 1. IDF-Aware Loss

#### IDF-weighted Matching Score
```python
# Equation 5 in paper
s(q, d) = Σ_t idf(t) · q_t · d_t
```

#### Gradient Composition
```python
# Equation 7 in paper
∂L/∂d_i,t = ∂L_rank-idf/∂d_i,t + λ · ∂L_FLOPS/∂d_i,t
```

**효과**:
- High-IDF 토큰: Ranking gradient >> FLOPS penalty → **보존**
- Low-IDF 토큰: FLOPS penalty 지배적 → **제거**

### 2. Ensemble Teacher

#### Teacher Models
- **Dense**: Alibaba-NLP/gte-large-en-v1.5
- **Sparse**: opensearch-neural-sparse-encoding-v1
- **Cross-Encoder** (fine-tuning only): cross-encoder/ms-marco-MiniLM-L-12-v2

#### Ensemble Process
```python
# Min-max normalization (Equation 8)
ŝ_i^j = (s_i^j - min(s^j)) / (max(s^j) - min(s^j))

# Weighted ensemble (Equation 9)
ŝ = S · (w_dense · ŝ_dense + w_sparse · ŝ_sparse)

# S = 10 (pre-training), 30 (fine-tuning)
# w_dense = w_sparse = 0.5
```

## 학습 프로세스

### Step 1: 환경 설정
```bash
# 가상환경 생성
python3.12 -m venv .venv
source .venv/bin/activate

# 패키지 설치
pip install -r notebooks/opensearch-neural-v2/requirements_v2.txt

# GPU 확인
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 2: IDF 가중치 계산
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
)

# MS MARCO 문서 코퍼스 로드
documents = [...]  # 8.8M passages

# IDF 계산
idf_weights = compute_idf_weights(
    documents=documents,
    tokenizer=tokenizer,
    save_path="idf_weights/idf_msmarco.json",
)
```

### Step 3: 데이터 준비

#### Pre-training Data
논문의 Table 1 참조:
- S2ORC: (Title, Abstract) pairs - 500,000 queries
- WikiAnswers: duplicate questions - 1,000,000 queries
- GOOAQ: (Question, Answer) pairs - 2,274,901 queries
- ... 총 13개 데이터셋

#### Fine-tuning Data
- MS MARCO passage ranking
- 502,939 training queries
- 8,841,823 passages

### Step 4: 모델 초기화
```python
from src.models.neural_sparse_encoder import NeuralSparseEncoder

model = NeuralSparseEncoder(
    model_name="opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
    max_length=256,
    use_relu=True,
)

# PyTorch 2.0 compile
if torch.__version__ >= '2.0':
    model = torch.compile(model)
```

### Step 5: Loss Function 초기화
```python
loss_fn = IDFAwareLoss(
    idf_weights=idf_weights,
    vocab_size=tokenizer.vocab_size,
    lambda_flops=0.02,  # Fine-tuning
    default_idf=1.0,
    device=device,
)
```

### Step 6: Teacher Models 초기화
```python
teacher = EnsembleTeacher(
    dense_teacher_name="Alibaba-NLP/gte-large-en-v1.5",
    sparse_teacher_name="opensearch-project/opensearch-neural-sparse-encoding-v1",
    dense_weight=0.5,
    sparse_weight=0.5,
    scale_constant=30.0,  # Fine-tuning
    device=device,
)
```

### Step 7: 학습 실행
```python
# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01,
)

# Scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=50000,
)

# Mixed precision scaler
scaler = GradScaler()

# Training loop
for step in range(num_training_steps):
    losses = train_step(
        batch=batch,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scaler=scaler,
        teacher=teacher,
    )
```

## 평가 및 벤치마킹

### BEIR Benchmark
```python
# 13개 데이터셋
datasets = [
    "trec-covid", "nfcorpus", "nq", "hotpotqa",
    "fiqa", "arguana", "touche-2020", "dbpedia-entity",
    "scidocs", "fever", "climate-fever", "scifact", "quora"
]

# 평가 메트릭
metrics = ["ndcg@10", "recall@1000", "mrr@10"]
```

### Efficiency Metrics
1. **Theoretical FLOPS**: Average L1 norm per document
2. **End-to-end Latency**: P99 latency, mean throughput
3. **Index Size**: Non-zero terms per document

## 하이퍼파라미터 튜닝

### λ_FLOPS Tuning
Pareto optimal point 찾기:
```python
# Test multiple values
lambda_values = [0.005, 0.01, 0.02, 0.05, 0.1]

for lambda_val in lambda_values:
    # Train model
    # Measure NDCG@10 and FLOPS
    # Plot relevance vs efficiency curve
```

추천 범위:
- **Pre-training**: 1e-7 ~ 1e-6 (작은 값)
- **Fine-tuning**: 0.01 ~ 0.05 (큰 값)

### Learning Rate
```python
# Pre-training: 5e-5
# Fine-tuning: 2e-5
# With warmup: 10-20% of total steps
```

### Batch Size
```python
# DGX Spark GPU
batch_size = 40  # Per device
gradient_accumulation_steps = 1
# Effective batch size = 40
```

## 메모리 최적화

### Out of Memory 해결
```python
# 1. 배치 크기 감소
CONFIG["finetuning"]["batch_size"] = 20

# 2. Gradient accumulation
CONFIG["finetuning"]["gradient_accumulation_steps"] = 2

# 3. Max length 감소
CONFIG["model"]["max_doc_length"] = 128

# 4. Gradient checkpointing (속도 trade-off)
model.gradient_checkpointing_enable()
```

### Mixed Precision
```python
# BF16 권장 (GB10 GPU)
with autocast(enabled=True, dtype=torch.bfloat16):
    outputs = model(input_ids, attention_mask)
```

## 트러블슈팅

### 1. 낮은 검색 성능
**원인**:
- IDF 가중치 미계산 또는 잘못된 계산
- λ_FLOPS가 너무 큼
- Pre-training 부족

**해결**:
```python
# IDF 가중치 검증
idf_values = list(idf_weights.values())
print(f"IDF range: {min(idf_values):.2f} ~ {max(idf_values):.2f}")
print(f"IDF mean: {np.mean(idf_values):.2f}")

# λ_FLOPS 감소
lambda_flops = 0.01  # 0.02 → 0.01

# Pre-training 더 수행
num_steps = 200000  # 150K → 200K
```

### 2. 너무 많은 FLOPS
**원인**:
- λ_FLOPS가 너무 작음
- Fine-tuning 부족

**해결**:
```python
# λ_FLOPS 증가
lambda_flops = 0.05  # 0.02 → 0.05

# Fine-tuning 더 수행
num_steps = 70000  # 50K → 70K

# Sparsity 모니터링
stats = model.get_sparsity_stats(doc_reps)
print(f"Avg non-zero terms: {stats['avg_nonzero_terms']:.1f}")
```

### 3. Teacher 모델 로딩 실패
**원인**:
- HuggingFace 토큰 미설정
- 네트워크 문제

**해결**:
```bash
# HuggingFace 로그인
export HF_TOKEN="your_token_here"

# 또는 Python에서
from huggingface_hub import login
login(token="your_token_here")

# Cache 디렉토리 설정
export HF_HOME="/path/to/cache"
```

## 예상 학습 시간

### Pre-training (150K steps)
- **Nvidia GB10 GPU**: ~30-40 hours
- Batch size 48, 7 hard negatives
- With teacher models

### Fine-tuning (50K steps)
- **Nvidia GB10 GPU**: ~10-15 hours
- Batch size 40, 10 hard negatives
- With ensemble teachers

## 출력 파일

```
outputs/opensearch-neural-v2/
├── checkpoint-1000/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── neural_sparse_head.pt
├── checkpoint-2000/
├── ...
├── final_model/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── neural_sparse_head.pt
├── training_losses.png
└── training_config.json
```

## 다음 단계

### 1. BEIR 평가
```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

# 데이터셋 로드
dataset = "scifact"
corpus, queries, qrels = GenericDataLoader(dataset).load(split="test")

# 평가
results = evaluate_model(model, corpus, queries, qrels)
```

### 2. OpenSearch 배포
```bash
# 모델을 OpenSearch 형식으로 변환
python scripts/export_opensearch_model.py \
    --model_path outputs/opensearch-neural-v2/final_model \
    --output_path opensearch_model.zip

# OpenSearch에 업로드
# (OpenSearch ML Commons 사용)
```

### 3. Production 테스트
- End-to-end latency 측정
- Throughput 벤치마크
- A/B 테스트

## 참고 자료

### 논문 및 코드
- [arXiv Paper](https://arxiv.org/abs/2411.04403)
- [OpenSearch Sparse Model Tuning](https://github.com/zhichao-aws/opensearch-sparse-model-tuning-sample)
- [SPLADE](https://github.com/naver/splade)

### 모델
- [opensearch-neural-sparse-multilingual-v1](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1)
- [gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)

### 벤치마크
- [BEIR](https://github.com/beir-cellar/beir)
- [MS MARCO](https://microsoft.github.io/msmarco/)

## 인용

```bibtex
@article{geng2025towards,
  title={Towards Competitive Search Relevance For Inference-Free Learned Sparse Retrievers},
  author={Geng, Zhichao and Wang, Yiwen and Ru, Dongyu and Yang, Yang},
  journal={arXiv preprint arXiv:2411.04403},
  year={2025}
}
```
