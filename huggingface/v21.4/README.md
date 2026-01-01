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
pipeline_tag: feature-extraction
---

# Korean Neural Sparse Encoder v21.4

한국어 신경망 희소 인코더 - OpenSearch Neural Sparse 검색을 위한 SPLADE 기반 모델

## Model Description

This model is a SPLADE-based sparse encoder fine-tuned for Korean text, specifically optimized for:
- Legal domain terminology
- Medical domain terminology
- General Korean synonym expansion

### v21.4 Improvements

- **Curriculum Learning**: 3-phase training focusing on single-terms → balanced → full coverage
- **Dynamic Lambda Self**: Higher weight (8.0) for single-term self-reconstruction, lower (4.0) for sentences
- **Minimum Activation Loss**: Prevents garbage outputs by ensuring meaningful top-k activations
- **Enhanced Training Data**: Added explicit single-term synonym pairs for problem terms

### Training Results

- **Best Epoch**: 27
- **Recall@1**: 99.8%
- **MRR**: 0.9990

### Problem Terms Performance

| Term | v21.4 | v21.3 | v21.2 |
|------|-------|-------|-------|
| 추천 | 100% | 0% | 100% |
| 데이터베이스 | 75% | 0% | 50% |
| 증상 | 75% | 0% | 75% |
| 질환 | 75% | 0% | 75% |
| 인슐린 | 75% | 0% | 75% |
| **Average** | **80%** | 0% | 75% |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn

# Load model
tokenizer = AutoTokenizer.from_pretrained("sewoong/korean-neural-sparse-encoder-v21.4")
model = AutoModelForMaskedLM.from_pretrained("sewoong/korean-neural-sparse-encoder-v21.4")

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

## OpenSearch Integration

This model is designed to work with OpenSearch Neural Sparse Search. See the [OpenSearch documentation](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/) for integration details.

## Base Model

- **Base**: skt/A.X-Encoder-base
- **Parameters**: 149,372,240
- **Vocabulary**: 49,999 tokens

## Citation

```bibtex
@misc{korean-neural-sparse-v21.4,
  author = {Sewoong Lee},
  title = {Korean Neural Sparse Encoder v21.4},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/sewoong/korean-neural-sparse-encoder-v21.4}
}
```
