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
- custom
---

# Korean Neural Sparse Encoder v1

Korean neural sparse encoder for OpenSearch neural sparse search, fine-tuned with enhanced legal and medical domain vocabulary.

## Model Description

This model is based on [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base) (ModernBERT architecture) and fine-tuned for Korean term expansion in neural sparse retrieval tasks.

### Key Features

- **Korean-optimized**: Trained on Korean synonym pairs across 14 domains
- **Domain-specific**: Enhanced vocabulary for legal (법률) and medical (의료) domains
- **Sparse representation**: Uses SPLADE (Sparse Lexical AnD Expansion) architecture
- **OpenSearch compatible**: Designed for use with OpenSearch neural sparse search

## Training Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | skt/A.X-Encoder-base |
| Learning Rate | 3e-6 |
| Epochs | 25 |
| Batch Size | 64 |
| Max Length | 64 |
| Lambda Self | 4.0 |
| Lambda Synonym | 10.0 |
| Lambda Margin | 2.5 |
| Lambda FLOPS | 8e-3 |

### Training Data

- **Total triplets**: 119,753 (83,827 train / 35,926 test)
- **Domains**: 14 domains including general, legal (법률), and medical (의료)
- **Format**: Triplet (anchor, positive synonym, hard negative)

### Evaluation Results

| Metric | Score |
|--------|-------|
| Source Preservation | 100.0% |
| Synonym Activation | 100.0% |
| Combined Score | 200.0 |

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
- **Korean Legal Document Search**: Finding relevant legal documents with synonym matching
- **Korean Medical Document Search**: Finding relevant medical documents with terminology expansion
- **General Korean Search**: Improving search quality with synonym expansion

## Limitations

- Optimized for short queries (max 64 tokens)
- Best performance on Korean text
- Domain-specific vocabulary may not cover all specialized terms
- Requires `trust_remote_code=True` for custom model loading

## Citation

If you use this model, please cite:

```bibtex
@misc{korean-neural-sparse-encoder-v1,
  title={Korean Neural Sparse Encoder v1: Legal and Medical Domain Enhanced Sparse Retrieval Model},
  author={sewoong},
  year={2024},
  url={https://huggingface.co/sewoong/korean-neural-sparse-encoder-v1}
}
```

## License

Apache 2.0

## Acknowledgments

- Base model: [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base)
- Architecture inspired by [OpenSearch Neural Sparse](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/)
