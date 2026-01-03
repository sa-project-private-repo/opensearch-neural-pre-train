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

# Korean Neural Sparse Encoder

A SPLADE-based sparse encoder fine-tuned for Korean text, designed for neural sparse retrieval with OpenSearch.

## Model Description

This model generates sparse vector representations for Korean text using the SPLADE (Sparse Lexical and Expansion) approach. It is optimized for:

- **Legal domain terminology**: Korean legal terms and concepts
- **Medical domain terminology**: Korean medical and healthcare terms
- **General Korean text**: Everyday Korean language with synonym expansion

The model uses `log(1 + ReLU(MLM_logits))` activation to produce sparse representations suitable for inverted index-based retrieval systems like OpenSearch Neural Sparse Search.

## Training Details

- **Base Model**: [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base)
- **Training Method**: Curriculum learning with contrastive loss
- **Parameters**: 149,372,240
- **Vocabulary Size**: 49,999 tokens
- **Max Sequence Length**: 64 tokens

### Training Results

| Metric | Score |
|--------|-------|
| Recall@1 | 99.8% |
| MRR | 0.9990 |

## Usage

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn

# Load model
tokenizer = AutoTokenizer.from_pretrained("sewoong/korean-neural-sparse-encoder-v1")
model = AutoModelForMaskedLM.from_pretrained("sewoong/korean-neural-sparse-encoder-v1")

def encode(text: str) -> torch.Tensor:
    """Encode text to sparse representation."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        relu = nn.ReLU()
        token_scores = torch.log1p(relu(logits))
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        sparse_repr = (token_scores * mask).max(dim=1).values[0]
    return sparse_repr

# Example: Encode a query
sparse = encode("diabetes treatment methods")
top_values, top_indices = sparse.topk(10)

for idx, val in zip(top_indices, top_values):
    if val > 0:
        print(f"{tokenizer.decode([idx])}: {val:.4f}")
```

### Get Top Activated Tokens

```python
def get_top_tokens(text: str, top_k: int = 20) -> list:
    """Get top-k activated tokens from text."""
    sparse = encode(text)
    top_values, top_indices = sparse.topk(top_k)

    results = []
    for idx, val in zip(top_indices.tolist(), top_values.tolist()):
        if val > 0:
            token = tokenizer.decode([idx]).strip()
            results.append((token, round(val, 4)))
    return results

# Example
tokens = get_top_tokens("real estate contract termination conditions")
for token, weight in tokens:
    print(f"{token}: {weight}")
```

## OpenSearch Integration

This model is designed to work with [OpenSearch Neural Sparse Search](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/).

### Register Model in OpenSearch

```json
POST /_plugins/_ml/models/_register
{
  "name": "korean-neural-sparse-encoder",
  "version": "1.0.0",
  "model_format": "TORCH_SCRIPT",
  "model_task_type": "SPARSE_ENCODING"
}
```

### Create Neural Sparse Index

```json
PUT /my-neural-sparse-index
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "content": {
        "type": "text"
      },
      "content_sparse": {
        "type": "rank_features"
      }
    }
  }
}
```

## Intended Use

- **Primary Use**: Semantic search for Korean documents
- **Domains**: Legal, medical, and general Korean text
- **Task**: Document retrieval using sparse vector representations

## Limitations

- Optimized for Korean text; performance on other languages is not guaranteed
- Maximum sequence length is 64 tokens
- Best suited for short to medium-length queries and passages

## Citation

```bibtex
@misc{korean-neural-sparse-encoder,
  author = {Sewoong Lee},
  title = {Korean Neural Sparse Encoder},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/sewoong/korean-neural-sparse-encoder-v1}
}
```

## License

Apache 2.0
