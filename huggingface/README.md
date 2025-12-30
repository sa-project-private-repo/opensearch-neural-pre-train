---
language:
- ko
license: apache-2.0
tags:
- sparse-retrieval
- splade
- korean
- opensearch
- neural-search
- neural-sparse
- legal
- medical
library_name: transformers
pipeline_tag: feature-extraction
base_model: skt/A.X-Encoder-base
datasets:
- wikipedia
- klue
- korquad
- sean0042/KorMedMCQA
---

# Korean Neural Sparse Encoder

Korean neural sparse encoder for OpenSearch neural sparse search, optimized with filtered synonym data across legal and medical domains.

## Model Description

This model is based on [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base) and fine-tuned for Korean term expansion in neural sparse retrieval tasks using SPLADE architecture.

### Key Features

- **Korean-optimized**: Trained on 66,070 filtered Korean synonym pairs
- **Multi-domain**: Covers 14 domains including legal and medical terminology
- **Quality-filtered**: Uses IG/PMI/Cross-encoder ensemble filtering
- **Sparse representation**: SPLADE (Sparse Lexical AnD Expansion) architecture
- **OpenSearch compatible**: Designed for OpenSearch neural sparse search

## Training Details

### Architecture

```
Input -> A.X-Encoder-base -> log(1 + ReLU(logits)) -> Max Pooling -> Sparse Vector
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | skt/A.X-Encoder-base |
| Parameters | 149M |
| Learning Rate | 3e-6 |
| Epochs | 25 |
| Batch Size | 64 |
| Max Length | 64 |
| Lambda Self | 4.0 |
| Lambda Synonym | 10.0 |
| Lambda Margin | 2.5 |
| Lambda FLOPS | 8e-3 |

### Training Data

**Data Sources (14 domains):**

| Domain | Dataset | Description |
|--------|---------|-------------|
| Encyclopedia | Wikipedia (ko) | General knowledge |
| QA | KLUE-MRC, KorQuAD | Question-answering context |
| Legal | Korean Law Precedents | Case law, legal terminology |
| Medical | KorMedMCQA (4 configs) | Doctor/Nurse/Pharmacist/Dentist exams |
| Dialogue | NSMC, KorHate | Reviews, comments |

**Data Statistics:**

| Metric | Value |
|--------|-------|
| Raw synonym pairs | 75,732 |
| Filtered synonym pairs | 66,070 |
| Unique anchors | 28,371 |
| Training triplets | 315,729 |
| Validation triplets | 35,081 |

**Data Filtering Pipeline:**

Three algorithmic filters with ensemble voting:
1. **Information Gain (IG)**: Removes truncations and trivial pairs
2. **PMI**: Removes false positives based on corpus co-occurrence
3. **Cross-Encoder**: Validates semantic similarity

### Loss Function

```python
L_total = lambda_self * L_self           # Self-reconstruction
        + lambda_synonym * L_positive    # Synonym activation
        + lambda_margin * L_triplet      # Triplet margin loss
        + lambda_flops * L_flops         # Sparsity regularization
```

## Usage

### With Transformers

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sewoong/korean-neural-sparse-encoder-v1")
model = AutoModel.from_pretrained(
    "sewoong/korean-neural-sparse-encoder-v1",
    trust_remote_code=True
)

# Encode text
text = "손해배상 청구"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    sparse_repr, token_weights = model(**inputs)

# Get top activated tokens
top_k = 20
top_values, top_indices = sparse_repr[0].topk(top_k)
for idx, val in zip(top_indices.tolist(), top_values.tolist()):
    if val > 0:
        token = tokenizer.decode([idx]).strip()
        print(f"{token}: {val:.4f}")
```

### Example Output

For the query "손해배상" (damages/compensation):

```
손해: 3.1094
배상: 3.0312
보상: 2.2969
피해: 2.2656
손실: 2.2500
```

For the query "인공지능" (artificial intelligence):

```
인공지능: 3.0938
AI: 2.5625
지능: 2.3594
알고리즘: 2.2969
로봇: 2.1250
```

## Intended Use

This model is designed for:

- **OpenSearch Neural Sparse Search**: Term expansion for better recall
- **Korean Legal Document Search**: Finding relevant legal documents
- **Korean Medical Document Search**: Medical terminology expansion
- **General Korean Search**: Improving search quality with synonyms

## Limitations

- Optimized for short queries (max 64 tokens)
- Best performance on Korean text
- Requires `trust_remote_code=True` for custom model loading

## Citation

```bibtex
@misc{korean-neural-sparse-encoder,
  title={Korean Neural Sparse Encoder: Filtered Sparse Retrieval Model for Legal and Medical Domains},
  author={sewoong},
  year={2024},
  url={https://huggingface.co/sewoong/korean-neural-sparse-encoder-v1}
}
```

## License

Apache 2.0

## Acknowledgments

- Base model: [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base)
- Architecture: [SPLADE](https://arxiv.org/abs/2107.05720)
- Integration: [OpenSearch Neural Sparse Search](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/)
