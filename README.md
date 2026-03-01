# OpenSearch Korean Neural Sparse Model (V33)

Korean SPLADE neural sparse retrieval model for OpenSearch, based on SPLADEModernBERT architecture.

## Overview

| Property | Value |
|----------|-------|
| Base Model | `skt/A.X-Encoder-base` (ModernBERT) |
| Architecture | SPLADE-max (MLM -> log(1+ReLU) -> max pool) |
| Parameters | 149M |
| Vocabulary | 50,000 tokens (48.4% Korean) |
| Hidden Size | 768, 22 layers |
| Max Length | 256 (doc) / 64 (query) |
| Loss | InfoNCE + FLOPS (quadratic warmup) |
| Training | DDP 8x NVIDIA B200 (183GB each) |
| Effective Batch | 2048 (64 x 4 x 8 GPUs) |
| Training Data | 4.84M triplets (data/v29.0/) |
| HuggingFace | [sewoong/korean-neural-sparse-encoder](https://huggingface.co/sewoong/korean-neural-sparse-encoder) |

---

## Benchmark Results

| Benchmark | Queries | BM25 R@1 | Neural Sparse R@1 | Semantic R@1 |
|-----------|---------|----------|-------------------|--------------|
| Ko-StrategyQA | 592 | 53.7% | **62.2%** | 73.5% |
| MIRACL-ko | 213 | 44.1% | **62.0%** | 70.9% |
| Mr.TyDi-ko | 421 | 55.6% | **73.4%** | 84.1% |

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
make train          # DDP training (foreground)
make train-bg       # DDP training (background)
make train-resume   # Resume from checkpoint

# Monitoring
make logs           # Real-time training logs
make tensorboard    # TensorBoard (port 6006)
make monitor        # GPU usage
```

### Benchmark

```bash
make benchmark                   # Run all benchmarks
make benchmark-ko-strategyqa     # Ko-StrategyQA only
make benchmark-miracl            # MIRACL-ko only
make benchmark-mrtydi            # Mr.TyDi-ko only
```

### Export to HuggingFace

```bash
make export-hf
```

### Inference

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn

tokenizer = AutoTokenizer.from_pretrained("sewoong/korean-neural-sparse-encoder")
model = AutoModelForMaskedLM.from_pretrained("sewoong/korean-neural-sparse-encoder")

def encode(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        sparse_repr = torch.log1p(torch.relu(logits))
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        sparse_repr = (sparse_repr * mask).max(dim=1).values[0]
    return sparse_repr

sparse = encode("당뇨병 치료 방법")
top_values, top_indices = sparse.topk(10)
for idx, val in zip(top_indices, top_values):
    print(f"{tokenizer.decode([idx])}: {val:.4f}")
```

---

## Project Structure

```
opensearch-neural-pre-train/
├── benchmark/                  # Benchmark framework
│   ├── hf_runner.py            # HuggingFace dataset benchmark runner
│   ├── runner.py               # Custom data benchmark runner
│   ├── encoders.py             # Dense (BGE-M3) / Sparse (V33) encoders
│   ├── searchers.py            # BM25 / Semantic / Sparse / Hybrid
│   └── metrics.py              # Recall, MRR, nDCG
├── src/
│   ├── model/
│   │   ├── splade_modern.py    # SPLADEModernBERT model
│   │   ├── losses.py           # SPLADELossV33
│   │   └── teachers.py         # BGE-M3 teacher
│   ├── train/
│   │   ├── cli/train_v33_ddp.py  # V33 DDP training entry point
│   │   ├── config/v33.py         # V33Config
│   │   ├── core/trainer.py       # Base trainer
│   │   └── core/ddp_trainer.py   # DDP trainer
│   ├── preprocessing/           # Data processing pipeline
│   └── evaluation/              # Ranking metrics
├── scripts/
│   ├── launch_v33_b200.sh      # B200 x8 DDP launch script
│   └── export_v33_hf.py        # Export to HuggingFace format
├── configs/
│   └── train_v33.yaml          # V33 training configuration
├── data/v29.0/                 # Training data (4.84M triplets)
├── huggingface/v33/            # Exported HuggingFace model
└── outputs/train_v33/          # Training outputs & checkpoints
```

---

## Training Configuration

**Loss Function (SPLADE v2):**
```
L = L_infonce + lambda_q * L_FLOPS_q + lambda_d * L_FLOPS_d
```

| Component | Description |
|-----------|-------------|
| InfoNCE | In-batch negatives + explicit hard negatives |
| FLOPS | sum_j(mean_i(w_j^i))^2 per query/doc |
| Lambda Schedule | Quadratic warmup (10% floor -> target) |

**DDP Configuration:**

| Parameter | Value |
|-----------|-------|
| GPUs | 8x NVIDIA B200 |
| Per-GPU Batch | 64 |
| Gradient Accumulation | 4 |
| Effective Batch | 2048 |
| Epochs | 25 |
| Mixed Precision | BF16 |

---

## Make Targets

| Command | Description |
|---------|-------------|
| `make setup` | Setup Python venv and install dependencies |
| `make test` | Test GPU and model setup |
| `make train` | Start V33 DDP training |
| `make train-bg` | Start training in background |
| `make train-resume` | Resume from checkpoint |
| `make benchmark` | Run all benchmarks |
| `make export-hf` | Export to HuggingFace format |
| `make tensorboard` | Start TensorBoard |
| `make logs` | Tail training logs |
| `make monitor` | GPU usage |
| `make lint` | Code quality checks |
| `make format` | Format code |
| `make clean` | Clean generated files |
| `make info` | System information |

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

## References

- [SPLADE v2: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2109.10086)
- [OpenSearch Neural Sparse Search](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/)
- [opensearch-neural-sparse-encoding-multilingual-v1](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1)

## Requirements

- Python 3.12+
- PyTorch 2.0+ (CUDA 12.x)
- 8x NVIDIA B200 (or compatible multi-GPU)

## License

Apache License 2.0
