# OpenSearch Korean Neural Sparse Model

Korean SPLADE-doc neural sparse retrieval model for OpenSearch.

## Overview

This repository contains training code and benchmarks for Korean neural sparse search models. The models enable semantic sparse search with synonym expansion for Korean terms.

### Latest Version: V26 (Enhanced IDF with Special Token Fix)

| Property | Value |
|----------|-------|
| Base Model | `xlm-roberta-base` |
| Parameters | 278M |
| Vocabulary | 250,002 tokens |
| Max Length | 192 |
| Teacher | BAAI/bge-m3 |

**Key Features (V26):**
- Special Token Fix: `<s>`, `</s>` excluded from IDF normalization
- Enhanced FLOPS: 5x weight increase (0.002 → 0.010)
- Stronger Stopword Penalty: 3x increase (5.0 → 15.0)
- Sharper IDF Curve: alpha 2.5 → 4.0
- Extended Stopword List: 177 tokens (vs V25's 98)

**V26 vs V25 Hyperparameters:**

| Parameter | V25 | V26 | Reason |
|-----------|-----|-----|--------|
| `lambda_flops` | 0.002 | **0.010** | 5x FLOPS penalty boost |
| `stopword_penalty` | 5.0 | **15.0** | 3x stopword penalty boost |
| `idf_alpha` | 2.5 | **4.0** | Sharper penalty curve |
| `special_token_penalty` | - | **100.0** | Fixed penalty for special tokens |

**Target Metrics:**
- Semantic tokens in top-10: 80%+
- Stopword activation: <0.3
- Recall@1: BM25 parity or better

### V26 Benchmark Results

| Metric | V26 Neural Sparse | BM25 | BGE-M3 Dense |
|--------|-------------------|------|--------------|
| Recall@1 | **40.7%** | 30.0% | 37.1% |
| Recall@5 | **51.4%** | 45.2% | 48.3% |
| MRR | **0.4555** | 0.3612 | 0.4189 |

### V26 vs V25 Improvement

| Metric | V25 | V26 | Improvement |
|--------|-----|-----|-------------|
| Recall@1 | 28.2% | 40.7% | **+44.3%** |
| Semantic Ratio | 73.2% | 95.8% | **+30.9%** |

---

## Quick Start

> **V26 Training Guide**: 단계별 학습 가이드는 [GUIDE.md](./GUIDE.md)를 참조하세요.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn

# Load model
tokenizer = AutoTokenizer.from_pretrained("sewoong/korean-neural-sparse-encoder")
model = AutoModelForMaskedLM.from_pretrained("sewoong/korean-neural-sparse-encoder")

# Encode text
def encode(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=192)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        relu = nn.ReLU()
        token_scores = torch.log1p(relu(logits))
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        sparse_repr = (token_scores * mask).max(dim=1).values[0]
    return sparse_repr

# Example
sparse = encode("당뇨병 치료 방법")
top_values, top_indices = sparse.topk(10)
for idx, val in zip(top_indices, top_values):
    print(f"{tokenizer.decode([idx])}: {val:.4f}")
```

---

## Project Structure

```
opensearch-neural-pre-train/
├── benchmark/                             # Benchmark framework
│   ├── runner.py                          # Main benchmark runner
│   ├── encoders.py                        # Dense/Sparse encoders
│   ├── searchers.py                       # BM25/Semantic/Sparse/Hybrid
│   ├── metrics.py                         # Recall, MRR, nDCG
│   └── config.py                          # Benchmark configuration
├── src/
│   ├── model/                             # Model implementations
│   ├── train/                             # Training scripts (V26, V27)
│   ├── preprocessing/                     # Data processing pipeline
│   └── evaluation/                        # Ranking metrics
├── configs/                               # Training configurations
│   ├── train_v26.yaml                     # V26 configuration
│   └── train_v27.yaml                     # V27 travel domain config
├── data/                                  # Training data
├── huggingface/v26/                       # V26 model for local use
└── outputs/                               # Training outputs
```

---

## Benchmark

Compare search methods against a validation dataset.

### Search Methods

| Method | Description | Index Type |
|--------|-------------|------------|
| BM25 | Lexical keyword search | text |
| Semantic | Dense vector k-NN (bge-m3) | knn_vector |
| Neural Sparse | SPLADE sparse vectors | rank_features |
| Hybrid | BM25 + k-NN combination | text + knn_vector |

### Running Benchmarks

```bash
# Full benchmark (with indexing)
python -m benchmark.runner --sample-size 1000 --output-dir outputs/benchmark

# Skip indexing (data already indexed)
python -m benchmark.runner --sample-size 1000 --skip-setup --output-dir outputs/benchmark

# Parallel execution (all methods simultaneously)
python -m benchmark.runner --sample-size 1000 --skip-setup --parallel --output-dir outputs/benchmark
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--sample-size` | Number of queries | 2000 |
| `--skip-setup` | Skip index creation | False |
| `--parallel` | Run all methods in parallel | False |
| `--cleanup` | Delete indices after benchmark | False |
| `--output-dir` | Output directory | outputs/benchmark |

### Output

```
outputs/benchmark/
├── report.md          # Summary report
├── metrics.json       # Detailed metrics (Recall@K, MRR, nDCG, latency)
└── benchmark.log      # Execution log
```

---

## Training Pipeline

### V26 Training (Recommended)

V26 fixes V25's stopword dominance problem by excluding special tokens from IDF normalization and increasing regularization strength.

> 상세 가이드: [GUIDE.md](./GUIDE.md)

#### V26 Make Commands

| Command | Description |
|---------|-------------|
| `make prepare-v26-idf` | Compute IDF weights for V26 |
| `make prepare-v26-data` | Prepare all V26 data |
| `make train-v26` | Start V26 training (foreground) |
| `make train-v26-bg` | Start V26 training (background with nohup) |
| `make train-v26-resume` | Resume V26 training from checkpoint |
| `make logs-v26` | View V26 training logs (real-time) |
| `make tensorboard-v26` | Start TensorBoard for V26 |
| `make convert-v26-hf` | Export V26 to HuggingFace format |
| `make eval-v26` | Evaluate V26 model |
| `make eval-v26-sparsity` | Analyze V26 sparsity patterns |
| `make validate-semantic-ratio` | Validate semantic token ratio |
| `make v26-pipeline` | Run full V26 pipeline |

#### Full Training Pipeline

```bash
# 1. IDF weights computation
make prepare-v26-idf

# 2. Start training (foreground)
make train-v26

# 3. Or start in background (recommended for long training)
make train-v26-bg

# 4. Monitor training
make logs-v26
make tensorboard-v26

# 5. Resume from checkpoint (if interrupted)
make train-v26-resume

# 6. After training: export and validate
make convert-v26-hf
make validate-semantic-ratio
```

#### Background Execution

```bash
# Method 1: Make target (recommended)
make train-v26-bg
# PID and log path will be shown

# Method 2: tmux (for interactive monitoring)
tmux new -d -s train 'make train-v26'
tmux attach -t train    # Attach to session
# Ctrl+b, d              # Detach from session

# Method 3: nohup (manual)
nohup make train-v26 > outputs/train_v26/training.log 2>&1 &
```

#### TensorBoard Monitoring

```bash
# Use make target
make tensorboard-v26

# Or manually
tensorboard --logdir outputs/train_v26/tensorboard --port 6006 --bind_all
```

**Remote Access (SSH Tunneling):**
```bash
ssh -L 6006:localhost:6006 ec2-user@<EC2-IP>
# Then open http://localhost:6006 in browser
```

#### Training Status Check

```bash
# View live logs
make logs-v26

# Or directly
tail -f outputs/train_v26/training.log

# Check GPU usage
nvidia-smi -l 1
```

**V26 Loss Function:**
```
L_total = λ_infonce * L_infonce      # Contrastive learning
        + λ_self * L_self            # Self-reconstruction
        + λ_positive * L_positive    # Positive alignment
        + λ_flops * L_idf_flops      # IDF-weighted sparsity (enhanced)
        + λ_min_act * L_min_act      # Minimum activation
        + λ_kd * L_kd                # Knowledge distillation
```

**V26 IDF-Aware FLOPS Enhancements:**
- Special tokens (`<s>`, `</s>`) → Fixed penalty 100.0 (excluded from IDF normalization)
- High IDF (rare tokens like 서울, 맛있는) → Low penalty
- Low IDF (common tokens like 을, 는) → High penalty (15x stopword multiplier)

---

## Data Pipeline

### Data Sources

Training data is loaded from HuggingFace datasets:

| Domain | Sources |
|--------|---------|
| Encyclopedia | Wikipedia (ko) |
| QA | KLUE-MRC, KorQuAD 1.0/2.0 |
| Legal | Korean Law Precedents |
| Medical | KorMedMCQA (doctor, nurse, pharm, dentist) |

### Training Data Format

Training data uses JSONL (JSON Lines) format where each line is a triplet:

```
data/v21.4/
├── training_triplets.jsonl      # Main training set (~423K triplets)
├── validation_triplets.jsonl    # Validation set (~47K triplets)
├── phase1_single_term_focus_triplets.jsonl  # Curriculum phase 1
├── phase2_balanced_triplets.jsonl           # Curriculum phase 2
└── phase3_full_triplets.jsonl               # Curriculum phase 3
```

#### Triplet Schema

| Field | Type | Description |
|-------|------|-------------|
| `anchor` | string | Input text (query or term) |
| `positive` | string | Semantically similar text (synonym, paraphrase) |
| `negative` | string | Hard negative (different meaning) |
| `difficulty` | string | `easy`, `medium`, `hard` |
| `length_class` | string | `single_term`, `short_phrase`, `sentence` |
| `pair_type` | string | Data source identifier |

#### Examples

**Single-term synonym (Korean):**
```json
{"anchor": "추천", "positive": "권장", "negative": "반대", "difficulty": "easy", "length_class": "single_term", "pair_type": "single_term"}
```

**Spelling variation:**
```json
{"anchor": "인터컨티넨탈", "positive": "인터콘티넨탈", "negative": "힐튼", "difficulty": "easy", "length_class": "short_phrase", "pair_type": "original"}
```

**Question-Answer pair (KorQuAD):**
```json
{"anchor": "대한민국의 수도는 어디인가?", "positive": "서울은 대한민국의 수도이다", "negative": "부산은 항구도시이다", "difficulty": "medium", "length_class": "sentence", "pair_type": "korquad"}
```

**Medical terminology:**
```json
{"anchor": "당뇨병", "positive": "diabetes mellitus", "negative": "고혈압", "difficulty": "medium", "length_class": "single_term", "pair_type": "original"}
```

**MS MARCO Korean:**
```json
{"anchor": "비타민D 결핍 증상", "positive": "비타민D가 부족하면 뼈가 약해지고 피로감이 증가합니다", "negative": "비타민C는 면역력 강화에 도움이 됩니다", "difficulty": "hard", "length_class": "sentence", "pair_type": "msmarco_ko"}
```

#### Data Distribution

| Pair Type | Count | Description |
|-----------|-------|-------------|
| original | 151K | Wikipedia entities, synonyms |
| msmarco_ko | 97K | MS MARCO Korean translation |
| korquad | 54K | KorQuAD question-context pairs |
| korquad_context | 54K | KorQuAD context-question pairs |
| naver_news | 35K | News article pairs |
| klue_nli | 15K | KLUE NLI entailment pairs |
| klue_sts | 11K | KLUE STS similar sentences |
| kobest_copa | 6K | KoBEST COPA reasoning pairs |
| single_term | 448 | Explicit single-term synonyms |

#### Curriculum Learning Phases

v21.4 uses 3-phase curriculum learning:

| Phase | Epochs | Focus | Lambda Self |
|-------|--------|-------|-------------|
| Phase 1 | 1-10 | Single-term pairs (100%) | 8.0 |
| Phase 2 | 11-20 | Balanced mix (50% single-term) | 6.0 |
| Phase 3 | 21-30 | Full dataset | 4.0 |

---

## OpenSearch Integration

### Create Index

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

### Neural Sparse Query

```json
POST /documents/_search
{
    "query": {
        "bool": {
            "should": [
                {"rank_feature": {"field": "sparse_embedding.당뇨병", "boost": 2.5}},
                {"rank_feature": {"field": "sparse_embedding.치료", "boost": 1.8}},
                {"rank_feature": {"field": "sparse_embedding.방법", "boost": 1.2}}
            ]
        }
    }
}
```

---

## Documentation

상세 기술 문서는 [docs/](./docs/) 디렉토리를 참조하세요.

### Concepts
- [Neural Sparse Model 개요](./docs/concepts/01-neural-sparse-overview.md) - Neural Sparse 검색의 정의 및 비교
- [SPLADE Architecture Deep Dive](./docs/concepts/02-splade-architecture.md) - XLM-RoBERTa 기반 아키텍처 상세
- [Model Operation](./docs/concepts/03-model-operation.md) - Forward Pass 및 Inference 동작
- [Loss Functions 상세](./docs/concepts/04-loss-functions.md) - 6개 손실 함수 및 Knowledge Distillation

### Guides
- [Training 가이드](./docs/guides/training-guide.md) - 환경 설정부터 학습까지
- [OpenSearch 통합 가이드](./docs/guides/opensearch-integration.md) - Index 생성, 쿼리, Hybrid Search
- [Model Loading 가이드](./docs/guides/model-loading-guide.md) - 다양한 로딩 시나리오

### Reference
- [Hyperparameter 참조](./docs/reference/hyperparameters.md) - V26 설정 및 튜닝 가이드
- [한국어 Stopword 처리](./docs/reference/korean-stopwords.md) - 177개 불용어 및 처리 메커니즘

---

## Version History

| Version | Description | Status |
|---------|-------------|--------|
| **V26** | XLM-RoBERTa + Enhanced IDF + Special token fix + Extended stopwords | **Production** |
| V27 | Travel/Tourism domain enhancement (in development) | Development |

### V26 Technical Details

| Aspect | Value |
|--------|-------|
| Base Model | `xlm-roberta-base` |
| Special Token Handling | Excluded from IDF normalization + fixed penalty 100.0 |
| FLOPS Weight | 0.010 |
| Stopword Penalty | 15.0 |
| IDF Alpha | 4.0 |
| Stopword Count | 177 |
| Semantic Ratio | 95.8% |

---

## Requirements

- Python 3.12+
- PyTorch 2.0+
- transformers 4.35+
- CUDA 11.8+ (for GPU training)

```bash
pip install torch transformers sentence-transformers opensearch-py tqdm scikit-learn matplotlib
```

---

## References

- [SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720)
- [OpenSearch Neural Sparse Search](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/)
- [A.X-Encoder](https://huggingface.co/skt/A.X-Encoder-base)

## License

Apache License 2.0
