# OpenSearch Korean Neural Sparse Model

Neural sparse retrieval model for Korean text in OpenSearch.

## Overview

This repository contains training code for Korean SPLADE-doc models. The models enable neural sparse search with synonym expansion for Korean terms.

### Model Versions

| Version | Description | Key Features |
|---------|-------------|--------------|
| v19 | Cross-lingual (KO-EN) | XLM-RoBERTa, bilingual dictionary |
| v21.2 | Korean synonym expansion | Legal/Medical domain, A.X-Encoder |
| **v21.3** | **Enhanced Korean model** | **Noise filtering, balanced hard negatives, ranking metrics** |

---

## v21.3 Korean Neural Sparse Encoder (Latest)

v21.3는 데이터 품질과 평가 지표를 알고리즘적/통계적 방법으로 개선한 한국어 Neural Sparse 모델입니다.

### v21.2 대비 개선사항

| 항목 | v21.2 | v21.3 |
|------|-------|-------|
| 데이터 노이즈 | ~50% (BPE 노이즈) | **< 10%** (3단계 필터링) |
| 평가 지표 | 100% 포화 (Binary) | **Recall@K, MRR** (비포화) |
| 의료 데이터 | 로드 실패 | **4개 config 정상 로드** |
| Hard Negative | Random 샘플링 | **난이도별 균형 샘플링** |

### 데이터 전처리 파이프라인

#### 1. 데이터 수집 (`00_data_ingestion.ipynb`)

14개 한국어 데이터셋에서 텍스트 수집:

| 도메인 | 데이터셋 | 설명 |
|--------|----------|------|
| 백과사전 | Wikipedia (ko) | 일반 지식 |
| QA | KLUE-MRC, KorQuAD | 질의응답 컨텍스트 |
| 법률 | Korean Law Precedents | 판례, 법률 용어 |
| **의료** | **KorMedMCQA (4 configs)** | 의사/간호사/약사/치과 자격시험 |
| 대화 | NSMC, KorHate | 리뷰, 댓글 |

**의료 데이터 로드 (v21.2 버그 수정):**
```python
# v21.2: 실패 (config 미지정)
load_dataset("sean0042/KorMedMCQA", split="train")  # Error

# v21.3: 성공 (config 명시)
medical_configs = ["dentist", "doctor", "nurse", "pharm"]
for config in medical_configs:
    load_dataset("sean0042/KorMedMCQA", config, split="train")
```

#### 2. 노이즈 필터링 (`01_noise_filtering.ipynb`)

3가지 알고리즘적 필터링을 앙상블로 적용:

```
┌─────────────────────────────────────────────────────────────┐
│                    Raw Synonym Pairs                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │   IG    │  │   PMI   │  │   CE    │
    │ Filter  │  │ Filter  │  │ Filter  │
    └────┬────┘  └────┬────┘  └────┬────┘
         │            │            │
         └────────────┼────────────┘
                      ▼
              ┌───────────────┐
              │ Majority Vote │
              │   (2/3 통과)   │
              └───────┬───────┘
                      ▼
              Filtered Pairs
```

**필터링 방법:**

| 필터 | 알고리즘 | 목적 | Threshold |
|------|----------|------|-----------|
| **IG** | KNN Entropy (Kozachenko-Leonenko) | Truncation/Case 변경 제거 | Bottom 10% |
| **PMI** | 동시출현 확률 (Laplace smoothing) | False positive 제거 | Bottom 10% |
| **CE** | Cross-Encoder (bge-reranker-v2-m3) | 의미적 비유사 쌍 제거 | Bottom 10% |

**Information Gain 계산:**
```python
IG(source → target) = H(target) - H(target|source)

# H(target): 전체 코퍼스에서 target의 entropy
# H(target|source): source 이웃 내에서 target의 conditional entropy
# 높은 IG = 의미적 확장이 유의미 (유지)
# 낮은 IG = 단순 truncation (제거)
```

**PMI 계산:**
```python
PMI(x, y) = log(P(x,y) / (P(x) * P(y)))

# 높은 PMI = 동시 출현 빈도 높음 (진짜 동의어)
# 낮은 PMI = 독립적 출현 (임베딩 유사도만 높은 false positive)
```

**앙상블 결정:**
- 3개 필터 중 **2개 이상 통과**해야 유지 (다수결 투표)
- 하드코딩 threshold 없이 **percentile 기반** 자동 결정

#### 3. 데이터 준비 (`02_data_preparation.ipynb`)

**난이도별 Hard Negative Mining:**

| 난이도 | 유사도 범위 | 비율 | 특징 |
|--------|-------------|------|------|
| Easy | 0.3 - 0.5 | 33% | 쉽게 구분 가능 |
| Medium | 0.5 - 0.7 | 33% | 중간 난이도 |
| Hard | 0.7 - 0.9 | 33% | 어려운 negative |

```python
# Triplet 형식: (anchor, positive, negative)
{
    "anchor": "인공지능",
    "positive": "AI",           # 동의어
    "negative": "자동화",       # Hard negative (유사하지만 다름)
    "difficulty": "hard"
}
```

### 학습 (`03_training.ipynb`)

**모델 구조:**
```
Input → A.X-Encoder-base → log(1 + ReLU(logits)) → Max Pooling → Sparse Vector
```

**손실 함수:**
```python
L_total = λ_self * L_self           # Self-reconstruction
        + λ_synonym * L_positive    # Synonym activation
        + λ_margin * L_triplet      # Triplet margin loss
        + λ_flops * L_flops         # Sparsity regularization
```

**Hyperparameters (v21.3):**
```yaml
model_name: skt/A.X-Encoder-base
batch_size: 64
num_epochs: 25
learning_rate: 3e-6
lambda_self: 4.0
lambda_synonym: 10.0
lambda_margin: 2.5
lambda_flops: 8e-3
target_margin: 1.5
```

### 평가 지표 (v21.2 포화 문제 해결)

| 지표 | 설명 | v21.2 문제 | v21.3 해결 |
|------|------|------------|------------|
| Recall@K | Top-K에 정답 동의어 비율 | N/A | ✓ 구현 |
| MRR | 첫 정답 순위의 역수 평균 | N/A | ✓ 구현 |
| nDCG | 순위별 가중치 적용 점수 | N/A | ✓ 구현 |
| Binary Accuracy | 정답 포함 여부 | 100% 포화 | 사용 안함 |

```python
# Recall@K
Recall@K = |Retrieved@K ∩ Relevant| / |Relevant|

# MRR (Mean Reciprocal Rank)
MRR = (1/|Q|) * Σ (1/rank_i)
```

### 실행 방법

```bash
cd notebooks/opensearch-neural-v21.3/

# 1. 데이터 수집 (의료 데이터 포함)
jupyter nbconvert --execute 00_data_ingestion.ipynb

# 2. 노이즈 필터링 (30-60분 소요)
jupyter nbconvert --execute 01_noise_filtering.ipynb

# 3. Hard Negative Mining
jupyter nbconvert --execute 02_data_preparation.ipynb

# 4. 모델 학습 (GPU 필요)
jupyter nbconvert --execute 03_training.ipynb

# 5. 추론 테스트
jupyter nbconvert --execute 04_inference_test.ipynb
```

### 출력 디렉토리

```
dataset/v21.3_filtered_enhanced/
├── raw_synonym_pairs.jsonl        # 원본 동의어 쌍
├── filtered_synonym_pairs.jsonl   # 필터링된 동의어 쌍
├── removed_synonym_pairs.jsonl    # 제거된 쌍 (분석용)
├── filtering_stats.json           # 필터링 통계
├── triplet_dataset/               # HuggingFace Dataset
├── train_triplets.jsonl           # 학습 데이터
└── val_triplets.jsonl             # 검증 데이터

outputs/v21.3_korean_enhanced/
├── best_model.pt                  # 최고 성능 모델
├── final_model.pt                 # 최종 모델
├── training_history.json          # 학습 기록
└── training_curves.png            # 학습 곡선
```

### 핵심 모듈

| 모듈 | 경로 | 설명 |
|------|------|------|
| Information Gain | `src/information_gain.py` | KNN Entropy 기반 IG 계산 |
| PMI | `src/pmi/` | 동시출현 행렬 및 PMI 계산 |
| Ranking Metrics | `src/evaluation/ranking_metrics.py` | Recall@K, MRR, nDCG |

---

## v19 Cross-lingual Model (Legacy)

## Model Specification

| Property | Value |
|----------|-------|
| Base Model | `xlm-roberta-large` |
| Parameters | 560M |
| Vocabulary Size | 250,002 tokens |
| Max Sequence Length | 512 |
| Supported Languages | Korean, English |
| Output Format | Sparse vector (rank_features) |

## Architecture

The model implements SPLADE-doc (Sparse Lexical AnD Expansion) architecture:

```
Document → XLM-RoBERTa Encoder → log(1 + ReLU(logits)) → Max Pooling → Sparse Vector
```

Key characteristics:
- Document-only mode: No query encoder required at search time
- Inference-free queries: Query encoding uses IDF lookup only
- Cross-lingual expansion: Korean terms activate related English tokens and vice versa

## Training Data

### Data Sources

| Source | Description | Pairs |
|--------|-------------|-------|
| MUSE | Facebook bilingual dictionary | ~20,000 |
| Wikidata | Entity labels (ko/en) | ~10,000 |
| IT Terminology | Technical terms | ~80 |

### Data Format

All training data consists of single-token pairs only. Multi-word phrases are not supported.

```json
{"ko": "프로그램", "en": "program", "source": "muse"}
{"ko": "네트워크", "en": "network", "source": "muse"}
{"ko": "머신러닝", "en": "machine", "source": "it_terminology"}
{"ko": "머신러닝", "en": "learning", "source": "it_terminology"}
```

### Data Processing Pipeline

1. **Data Ingestion** (`00_data_ingestion.ipynb`)
   - Collect term pairs from MUSE, Wikidata, IT terminology
   - Filter to single tokens only (no spaces allowed)
   - Validate Korean/English token format

2. **Data Preparation** (`01_data_preparation.ipynb`)
   - Generate embeddings using `intfloat/multilingual-e5-large`
   - K-means clustering for semantic grouping
   - Extract 1:N mappings with dot product similarity scores
   - Configuration: `similarity_threshold=0.8`, `max_targets=8`

## Training

### Configuration

```yaml
model_name: xlm-roberta-large
batch_size: 64
gradient_accumulation_steps: 2
num_epochs: 15
learning_rate: 3e-6
warmup_ratio: 0.1
lambda_positive: 1.0
lambda_negative: 1.0
lambda_sparsity: 0.01
similarity_threshold: 0.8
max_targets_per_source: 8
```

### Loss Function

The training uses similarity-weighted loss:

```
L = L_positive + λ_neg * L_negative + λ_sparse * L_sparsity

L_positive = -Σ sim_i * log(σ(score_i))    # Activate target tokens
L_negative = -Σ log(1 - σ(score_j))         # Suppress non-target tokens
L_sparsity = mean(score)                    # Encourage sparsity
```

### Training Pipeline

```bash
# 1. Data collection
jupyter notebook notebooks/opensearch-neural-v19/00_data_ingestion.ipynb

# 2. Data preparation
jupyter notebook notebooks/opensearch-neural-v19/01_data_preparation.ipynb

# 3. Model training
jupyter notebook notebooks/opensearch-neural-v19/02_training.ipynb

# 4. Inference test
jupyter notebook notebooks/opensearch-neural-v19/03_inference_test.ipynb
```

## Output

### Model Files

```
outputs/v19_xlm_large/
├── checkpoint.pt          # Model weights
├── tokenizer/             # XLM-RoBERTa tokenizer
└── training_curves.png    # Loss visualization
```

### Checkpoint Format

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "best_loss": best_loss,
    "config": {
        "model_name": "xlm-roberta-large",
        "max_length": 512,
        "vocab_size": 250002
    }
}
```

## Usage

### Load Trained Model

```python
import torch
from transformers import AutoTokenizer
from src.model.splade_model import SPLADEDoc

# Load checkpoint
checkpoint = torch.load("outputs/v19_xlm_large/checkpoint.pt")
config = checkpoint["config"]

# Initialize model
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
model = SPLADEDoc(model_name=config["model_name"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Encode document
def encode_document(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        sparse_vec = model(inputs["input_ids"], inputs["attention_mask"])
    return sparse_vec
```

### OpenSearch Integration

Register model with ML Commons:

```json
POST /_plugins/_ml/models/_register
{
    "name": "korean-english-neural-sparse-v19",
    "version": "1.0.0",
    "model_format": "TORCH_SCRIPT",
    "function_name": "SPARSE_ENCODING"
}
```

Create index with rank_features:

```json
PUT /documents
{
    "mappings": {
        "properties": {
            "content": {"type": "text"},
            "sparse_embedding": {"type": "rank_features"}
        }
    }
}
```

Neural sparse query:

```json
POST /documents/_search
{
    "query": {
        "neural_sparse": {
            "sparse_embedding": {
                "query_text": "검색 쿼리",
                "model_id": "<model_id>"
            }
        }
    }
}
```

## Project Structure

```
opensearch-neural-pre-train/
├── notebooks/
│   └── opensearch-neural-v19/
│       ├── 00_data_ingestion.ipynb      # Data collection
│       ├── 01_data_preparation.ipynb    # Preprocessing
│       ├── 02_training.ipynb            # Model training
│       └── 03_inference_test.ipynb      # Evaluation
├── src/
│   └── model/
│       └── splade_model.py              # SPLADE-doc implementation
├── dataset/
│   └── v19_high_quality/
│       ├── term_pairs.jsonl             # Raw term pairs
│       └── term_mappings.jsonl          # Processed 1:N mappings
├── outputs/
│   └── v19_xlm_large/                   # Trained model
└── scripts/
    └── collect_term_data_v19.py         # CLI data collection
```

## Requirements

- Python 3.12+
- PyTorch 2.0+
- transformers 4.35+
- CUDA 11.8+ (for GPU training)

```bash
pip install torch transformers tqdm scikit-learn matplotlib
```

## References

- [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)
- [OpenSearch Neural Sparse Search](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/)
- [XLM-RoBERTa](https://huggingface.co/xlm-roberta-large)
- [MUSE Bilingual Dictionaries](https://github.com/facebookresearch/MUSE)

## License

Apache License 2.0
