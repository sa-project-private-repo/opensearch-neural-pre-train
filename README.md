# OpenSearch Korean Neural Sparse Model

Korean SPLADE-doc neural sparse retrieval model for OpenSearch.

## Overview

Training code and benchmarks for Korean neural sparse search models. The models enable semantic sparse search with synonym expansion for Korean terms.

### Latest Version: V28 (Context-Gated + DDP B200 x8)

| Property | Value |
|----------|-------|
| Base Model | `xlm-roberta-base` |
| Model Class | SPLADEDocContextGated |
| Parameters | 345M |
| Vocabulary | 250,002 tokens |
| Max Length | 192 |
| Teacher | BAAI/bge-m3 |
| Training | DDP 8x NVIDIA B200 (183GB each) |
| Effective Batch | 2048 (32 x 8 x 8 GPUs) |

**Key Features (V28):**
- Context-Gated Sparse Expansion (multi-head attention gate)
- Korean Language Filtering with warmup schedule
- Collapse Detection with auto-halving
- IDF-Aware FLOPS (BM25 smoothing)
- Curriculum Learning (3 phases, 25 epochs)
- Knowledge Distillation from BGE-M3

**V29 Data Pipeline:**
- 3.6M unique training triplets (18.7% dedup from 4.4M)
- Sources: KorQuAD, KLUE, mC4, Wikipedia, AI Hub, travel domain
- Rust-based parallel IDF computation (~30s vs 47min Python)
- Streaming MD5 hash dedup (~1min vs 30min+ Python MinHash)

---

## Quick Start

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Training (DDP Multi-GPU)

```bash
# Full pipeline: data collection -> dedup -> DDP training
make v29-pipeline

# Or step by step:
make build-v29-data       # Merge, dedup, shard (~1 min)
make compute-idf-rust     # IDF weights (~30s, requires Rust)
make train-v28-ddp        # DDP training (foreground)
make train-v28-ddp-bg     # DDP training (background)

# Monitoring
make logs-v28-ddp         # Real-time logs
make tensorboard-v28-ddp  # TensorBoard (port 6006)
make monitor              # GPU usage
```

### Inference

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn

tokenizer = AutoTokenizer.from_pretrained("sewoong/korean-neural-sparse-encoder")
model = AutoModelForMaskedLM.from_pretrained("sewoong/korean-neural-sparse-encoder")

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

sparse = encode("당뇨병 치료 방법")
top_values, top_indices = sparse.topk(10)
for idx, val in zip(top_indices, top_values):
    print(f"{tokenizer.decode([idx])}: {val:.4f}")
```

---

## Make Targets

### Data Pipeline

| Command | Description |
|---------|-------------|
| `make collect-v29-data` | Collect Korean datasets from HuggingFace |
| `make build-v29-data` | Merge, dedup, shard (bash, ~1-2 min) |
| `make compute-idf-rust` | IDF weights with Rust (~30s) |
| `make v29-data-stats` | Show data statistics |

### DDP Training (B200 x8)

| Command | Description |
|---------|-------------|
| `make train-v28-ddp` | DDP training (foreground) |
| `make train-v28-ddp-bg` | DDP training (background, nohup) |
| `make train-v28-ddp-resume` | Resume from checkpoint |
| `make logs-v28-ddp` | Real-time training logs |
| `make tensorboard-v28-ddp` | TensorBoard (port 6006) |

### Pipeline & Monitoring

| Command | Description |
|---------|-------------|
| `make v29-pipeline` | Full pipeline: collect -> build -> train |
| `make benchmark-ko-strategyqa` | Ko-StrategyQA benchmark |
| `make monitor` | GPU usage (real-time) |
| `make info` | System information |

---

## Project Structure

```
opensearch-neural-pre-train/
├── benchmark/                  # Benchmark framework
│   ├── runner.py               # Main benchmark runner
│   ├── encoders.py             # Dense/Sparse encoders
│   ├── searchers.py            # BM25/Semantic/Sparse/Hybrid
│   └── metrics.py              # Recall, MRR, nDCG
├── src/
│   ├── model/
│   │   └── splade_xlmr.py      # SPLADEDocContextGated model
│   ├── train/
│   │   ├── cli/train_v28_ddp.py # DDP training entry point
│   │   ├── core/trainer.py      # Base trainer
│   │   ├── core/ddp_trainer.py  # DDP trainer (weight untying, broadcast_buffers)
│   │   └── idf/                 # IDF computation
│   ├── preprocessing/           # Data processing pipeline
│   └── evaluation/              # Ranking metrics
├── tools/
│   └── idf-compute/            # Rust IDF computation tool
├── scripts/
│   ├── build_v29_data_fast.sh  # Fast bash dedup pipeline
│   └── collect_korean_datasets.py
├── configs/
│   └── train_v28_b200.yaml     # V28 DDP config (B200 x8)
├── data/v29.0/                 # Training data (3.6M triplets)
└── outputs/train_v28_ddp/      # Training outputs
```

---

## Benchmark Results (V26)

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | P50 (ms) |
|--------|----------|----------|-----------|-----|----------|
| **Sparse+Semantic (RRF)** | **44.6%** | **53.0%** | **56.4%** | **0.486** | 122.8 |
| Neural Sparse | 40.7% | 51.4% | 56.1% | 0.456 | 13.0 |
| Semantic (Dense) | 37.1% | 50.2% | 53.1% | 0.431 | 15.6 |
| BM25+Semantic (RRF) | 37.1% | 48.2% | 51.6% | 0.421 | 94.8 |
| BM25 | 30.0% | 42.2% | 44.6% | 0.354 | 11.6 |

---

## Training Configuration (V28)

**Loss Function:**
```
L = λ_infonce * L_infonce       # Contrastive learning (3.0)
  + λ_self * L_self             # Self-reconstruction (0.5)
  + λ_positive * L_positive     # Positive alignment (2.0)
  + λ_flops * L_idf_flops       # IDF-weighted sparsity (0.010)
  + λ_min_act * L_min_act       # Minimum activation (5.0)
  + λ_kd * L_kd                 # Knowledge distillation (2.0)
  + λ_language * L_language      # Korean language filtering (0.1)
```

**Curriculum Phases:**

| Phase | Epochs | Temperature | Focus |
|-------|--------|-------------|-------|
| 1 | 1-8 | 0.08 | Foundation with BGE-M3 teacher |
| 2 | 9-17 | 0.05 | Balanced with hard negatives |
| 3 | 18-25 | 0.04 | Hard negative refinement |

**DDP Configuration:**

| Parameter | Value |
|-----------|-------|
| GPUs | 8x NVIDIA B200 |
| Per-GPU Batch | 32 |
| Gradient Accumulation | 8 |
| Effective Batch | 2048 |
| Mixed Precision | BF16 |
| NCCL NVLS | Disabled (B200 stability) |
| broadcast_buffers | False (multi-forward-pass fix) |
| find_unused_parameters | False |

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
- [Neural Sparse Model 개요](./docs/concepts/01-neural-sparse-overview.md)
- [SPLADE Architecture Deep Dive](./docs/concepts/02-splade-architecture.md)
- [Model Operation](./docs/concepts/03-model-operation.md)
- [Loss Functions 상세](./docs/concepts/04-loss-functions.md)

### Guides
- [Training 가이드](./docs/guides/training-guide.md)
- [OpenSearch 통합 가이드](./docs/guides/opensearch-integration.md)
- [Model Loading 가이드](./docs/guides/model-loading-guide.md)

### Reference
- [Hyperparameter 참조](./docs/reference/hyperparameters.md)
- [한국어 Stopword 처리](./docs/reference/korean-stopwords.md)

---

## Version History

| Version | Description | Status |
|---------|-------------|--------|
| **V28** | Context-Gated expansion + DDP B200 x8 + V29 data (3.6M) | **Training** |
| V27 | Travel/Tourism domain enhancement | Complete |
| V26 | Enhanced IDF + Special token fix | Complete |
| V25 | IDF-Aware FLOPS training | Complete |
| V24 | XLM-RoBERTa baseline | Complete |

---

## Requirements

- Python 3.12+
- PyTorch 2.0+ (CUDA 12.x)
- 8x NVIDIA B200 (or compatible multi-GPU)
- Rust toolchain (for IDF computation)

```bash
pip install torch transformers sentence-transformers opensearch-py tqdm scikit-learn matplotlib tensorboard
```

---

## References

- [SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720)
- [OpenSearch Neural Sparse Search](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/)
- [opensearch-neural-sparse-encoding-multilingual-v1](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1)

## License

Apache License 2.0
