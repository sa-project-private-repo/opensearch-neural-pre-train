---
language:
- ko
license: apache-2.0
library_name: transformers
tags:
- sparse-retrieval
- splade
- opensearch
- korean
- neural-sparse
- infonce
pipeline_tag: feature-extraction
---

# Korean Neural Sparse Encoder v22.0

한국어 신경망 희소 인코더 - OpenSearch Neural Sparse 검색을 위한 SPLADE 기반 모델

## Model Description

This model is a SPLADE-based sparse encoder fine-tuned for Korean text, specifically optimized for:
- Legal domain terminology
- Medical domain terminology
- General Korean synonym expansion

### v22.0 Improvements (over v21.4)

- **InfoNCE Contrastive Loss**: In-batch negatives for better discriminative representations
- **Temperature Annealing**: 0.07 → 0.05 → 0.03 for progressively sharper discrimination
- **Expanded Training Data**: 840,859 total triplets across 3 phases
- **Curriculum Learning**: 3-phase training with dynamic InfoNCE weight increase

### Training Results

| Metric | v21.4 | v22.0 |
|--------|-------|-------|
| Training Recall@1 | - | **99.87%** |
| Training MRR | - | **0.9994** |
| General Terms Recall | 78.7% | **81.5%** |
| Garbage Outputs | 0/5 | 0/5 |

### Training Phases

| Phase | Epochs | Data Size | Temperature | InfoNCE Weight |
|-------|--------|-----------|-------------|----------------|
| Phase 1 (Single-term) | 1-10 | 66,685 | 0.07 | 1.0 |
| Phase 2 (Balanced) | 11-20 | 224,177 | 0.05 | 1.5 |
| Phase 3 (Full) | 21-30 | 549,997 | 0.03 | 2.0 |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn

# Load model
tokenizer = AutoTokenizer.from_pretrained("sewoong/korean-neural-sparse-encoder")
model = AutoModelForMaskedLM.from_pretrained("sewoong/korean-neural-sparse-encoder")

# Encode text
def encode(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
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

### Example Expansions

| Query | Top Expansions |
|-------|----------------|
| 손해배상 | 손해, 피해, 배상, 손실, 보상, 소송, 위자료 |
| 인공지능 | AI, 지능, 컴퓨터, IT, 로봇, 알고리즘 |
| 당뇨병 | 당뇨, 혈당, 인슐린, 비만, 콜레스테롤 |
| 계약서 | 계약, 약정, 협약, 합의, 계약금, 약관 |

## OpenSearch Integration

This model is designed to work with OpenSearch Neural Sparse Search. See the [OpenSearch documentation](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/) for integration details.

## Base Model

- **Base**: skt/A.X-Encoder-base
- **Parameters**: 149,372,240
- **Vocabulary**: 50,000 tokens
- **Max Length**: 64 tokens

## Citation

```bibtex
@misc{korean-neural-sparse-encoder,
  author = {Sewoong Lee},
  title = {Korean Neural Sparse Encoder},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/sewoong/korean-neural-sparse-encoder}
}
```
