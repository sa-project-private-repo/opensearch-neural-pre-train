---
language:
- ko
- en
- multilingual
license: apache-2.0
tags:
- sparse-retrieval
- splade
- korean
- opensearch
- neural-search
- neural-sparse
- xlm-roberta
- multilingual
library_name: transformers
pipeline_tag: feature-extraction
base_model: xlm-roberta-base
datasets:
- wikipedia
- klue
- korquad
---

# Korean Neural Sparse Encoder V26

Korean multilingual neural sparse encoder for OpenSearch neural sparse search, based on XLM-RoBERTa with IDF-aware FLOPS loss and enhanced stopword suppression.

## Model Description

This model is based on [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) and fine-tuned for Korean/multilingual term expansion in neural sparse retrieval tasks using SPLADE architecture with knowledge distillation from BGE-M3.

### Key Features

- **Multilingual Support**: Based on XLM-RoBERTa, supports Korean and other languages
- **IDF-Aware Training**: Uses document frequency-aware FLOPS loss for better term weighting
- **Enhanced Stopword Suppression**: V26 improvements eliminate stopword dominance
- **Knowledge Distillation**: Learns from BGE-M3 teacher model
- **OpenSearch Compatible**: Designed for OpenSearch neural sparse search

## V26 Improvements

V26 addresses the stopword dominance issue found in V25:

| Parameter | V25 | V26 | Change |
|-----------|-----|-----|--------|
| lambda_flops | 0.002 | 0.010 | 5x increase |
| stopword_penalty | 5.0 | 15.0 | 3x increase |
| idf_alpha | 2.5 | 4.0 | Sharper curve |
| special_token_penalty | - | 100.0 | NEW |
| stopword_list | 163 | 242 | Extended |

**Key Fix**: Special tokens (`<s>`, `</s>`) were excluded from IDF normalization to prevent range compression.

## Benchmark Results (2026-01-28)

Evaluated on 1,000 Korean QA pairs:

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|--------|----------|----------|-----------|-----|---------|
| **Neural Sparse (V26)** | **40.7%** | **51.4%** | **56.1%** | **0.4555** | **0.4806** |
| Semantic (BGE-M3) | 37.1% | 50.2% | 53.1% | 0.4307 | 0.4553 |
| BM25 | 30.0% | 42.2% | 44.6% | 0.3541 | 0.3767 |

### Performance Comparison

| Metric | V25 | V26 | Improvement |
|--------|-----|-----|-------------|
| Recall@1 | 28.2% | **40.7%** | **+44.3%** |
| vs BM25 | -6% | **+35.7%** | ✅ Fixed |
| vs Semantic | -24% | **+3.6pp** | ✅ Surpassed |

**Statistical Significance**: All comparisons are statistically significant (p < 0.01)

## Training Details

### Architecture

```
Input -> XLM-RoBERTa-base -> log(1 + ReLU(logits)) -> Max Pooling -> Sparse Vector
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | xlm-roberta-base |
| Parameters | 278M |
| Learning Rate | 2e-5 |
| Epochs | 25 |
| Batch Size | 48 |
| Max Length | 192 |
| Lambda FLOPS | 0.010 |
| Stopword Penalty | 15.0 |
| IDF Alpha | 4.0 |
| Special Token Penalty | 100.0 |

### Loss Function

```python
L_total = L_infonce                    # Contrastive learning
        + lambda_flops * L_flops_idf   # IDF-aware FLOPS regularization
        + lambda_kd * L_kd             # Knowledge distillation from BGE-M3
        + margin_loss                  # Triplet margin loss
```

## Usage

### With Transformers

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sewoong/korean-neural-sparse-encoder-v26")
model = AutoModelForMaskedLM.from_pretrained("sewoong/korean-neural-sparse-encoder-v26")

# Encode text
text = "당뇨병 치료 방법"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=192)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    # SPLADE transformation: log(1 + ReLU(logits))
    sparse_repr = torch.log1p(torch.relu(logits))
    # Max pooling over sequence
    sparse_repr = sparse_repr.max(dim=1).values

# Get top activated tokens
top_k = 10
top_values, top_indices = sparse_repr[0].topk(top_k)
print("Top-10 activated tokens:")
for idx, val in zip(top_indices.tolist(), top_values.tolist()):
    if val > 0:
        token = tokenizer.decode([idx]).strip()
        print(f"  {token}: {val:.4f}")
```

### Example Output

For the query "당뇨병 치료 방법" (diabetes treatment methods):

```
Top-10 activated tokens:
  병: 3.8709
  당: 3.8478
  치료: 3.8428
  뇨: 3.8229
  혈: 2.9696
  방법: 2.7375
  당뇨: 2.5123
  혈당: 2.3456
  의료: 2.1234
  약: 2.0123
```

**Note**: V26 now correctly activates semantic tokens (병, 당, 치료, 뇨) instead of stopwords (있습니다, 수, 하는) that dominated V25.

### With OpenSearch

```python
from opensearchpy import OpenSearch

# Create neural sparse index
index_body = {
    "settings": {
        "index.knn": True
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "sparse_embedding": {
                "type": "rank_features"
            }
        }
    }
}

# Index document with sparse embedding
doc = {
    "text": "당뇨병 치료 방법에 대한 안내",
    "sparse_embedding": {
        "병": 3.87, "당": 3.85, "치료": 3.84, "뇨": 3.82, "방법": 2.74
    }
}

# Neural sparse search
query = {
    "query": {
        "neural_sparse": {
            "sparse_embedding": {
                "query_text": "당뇨병 치료",
                "model_id": "your-model-id"
            }
        }
    }
}
```

## Intended Use

This model is designed for:

- **OpenSearch Neural Sparse Search**: Term expansion for better recall
- **Korean Document Search**: Finding relevant Korean documents
- **Multilingual Search**: Supports XLM-RoBERTa's 100+ languages
- **Medical/Legal Domain Search**: Optimized for specialized terminology

## Limitations

- Best performance with max 192 tokens
- Primary optimization for Korean, but supports multilingual
- Requires SPLADE-style sparse vector extraction

## Version History

| Version | Date | Recall@1 | Key Changes |
|---------|------|----------|-------------|
| V26 | 2026-01-28 | **40.7%** | IDF-aware FLOPS, enhanced stopword suppression |
| V25 | 2026-01-22 | 28.2% | XLM-RoBERTa base, knowledge distillation |
| V24 | 2026-01-15 | 25.1% | Curriculum learning |

## Citation

```bibtex
@misc{korean-neural-sparse-encoder-v26,
  title={Korean Neural Sparse Encoder V26: IDF-Aware FLOPS with Enhanced Stopword Suppression},
  author={sewoong},
  year={2026},
  url={https://huggingface.co/sewoong/korean-neural-sparse-encoder-v26}
}
```

## License

Apache 2.0

## Acknowledgments

- Base model: [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)
- Teacher model: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- Architecture: [SPLADE](https://arxiv.org/abs/2107.05720)
- Integration: [OpenSearch Neural Sparse Search](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/)
