---
language:
  - ko
license: apache-2.0
tags:
  - neural-sparse
  - splade
  - opensearch
  - korean
  - information-retrieval
  - sparse-retrieval
  - modernbert
library_name: transformers
pipeline_tag: feature-extraction
datasets:
  - kor-triplet-v1.0
  - aihub-624
  - aihub-86
  - OPUS-100
  - mC4-ko
  - Wikipedia-ko
  - ko-wikidata-QA
  - KorQuAD2
  - KLUE-NLI
  - KLUE-MRC
  - KLUE-STS
  - nsmc
  - ynat
  - persona-chat-ko
  - OIG-smallchip2-ko
  - ko-alpaca-bingsu
  - KoAlpaca
  - Open-Orca-Ko
  - sharegpt-deepl-ko
---

# korean-neural-sparse-encoder

A Korean-specific SPLADE-max sparse encoder fine-tuned from [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base) (ModernBERT). It maps Korean sentences and paragraphs to a 50,000-dimensional sparse vector space for semantic search and sparse retrieval tasks.

## Model Details

| Property | Value |
|----------|-------|
| **Model Type** | SPLADE Sparse Encoder (SPLADE-max) |
| **Base Model** | [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base) (ModernBERT) |
| **Parameters** | 149M |
| **Output Dimensionality** | 50,000 |
| **Hidden Size** | 768 |
| **Layers** | 22 |
| **Korean Token Ratio** | 48.4% of vocabulary |
| **Similarity Function** | Dot Product |
| **Maximum Sequence Length** | 8,192 tokens |

### Architecture

```
ModernBertForMaskedLM
  → MLM Head (hidden_size → vocab_size)
  → log(1 + ReLU(logits))        # SPLADE activation
  → Max Pooling over sequence     # Position-invariant representation
  → Sparse Vector (50,000-dim)
```

## Usage

### Direct Usage (Transformers)

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_name = "sewoong/korean-neural-sparse-encoder"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id,
               tokenizer.pad_token_id, tokenizer.unk_token_id}

@torch.no_grad()
def encode(text: str, max_length: int = 256) -> dict[str, float]:
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=max_length, truncation=True)
    logits = model(**inputs).logits
    sparse = torch.log1p(torch.relu(logits))
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    vec = (sparse * mask).max(dim=1).values.squeeze(0)

    result = {}
    for idx in (vec > 0).nonzero(as_tuple=True)[0].tolist():
        if idx not in special_ids:
            token = tokenizer.convert_ids_to_tokens(idx)
            result[token] = round(vec[idx].item(), 4)
    return result

# Example
vec = encode("한국 전쟁의 원인과 결과")
print(f"Active dimensions: {len(vec)}")
print(sorted(vec.items(), key=lambda x: -x[1])[:10])
```

### Usage with OpenSearch (Client-Side Encoding)

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_name = "sewoong/korean-neural-sparse-encoder"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id,
               tokenizer.pad_token_id, tokenizer.unk_token_id}

@torch.no_grad()
def encode_for_opensearch(text: str, max_length: int = 256) -> dict[str, float]:
    """Encode text to sparse vector with integer token IDs for sparse_vector field."""
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=max_length, truncation=True)
    logits = model(**inputs).logits
    sparse = torch.log1p(torch.relu(logits))
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    vec = (sparse * mask).max(dim=1).values.squeeze(0)

    result = {}
    for idx in (vec > 0).nonzero(as_tuple=True)[0].tolist():
        if idx not in special_ids:
            weight = round(vec[idx].item(), 4)
            if weight > 0:
                result[str(idx)] = weight  # Integer token ID as string key
    return result
```

#### Create Index

```json
PUT /my-sparse-index
{
  "mappings": {
    "properties": {
      "content": {
        "type": "text",
        "analyzer": "nori"
      },
      "sparse_embedding": {
        "type": "sparse_vector",
        "index": true,
        "method": {
          "name": "seismic",
          "parameters": {
            "n_postings": 300,
            "cluster_ratio": 0.1,
            "summary_prune_ratio": 0.4
          }
        }
      }
    }
  }
}
```

#### Search

```json
GET /my-sparse-index/_search
{
  "query": {
    "neural_sparse": {
      "sparse_embedding": {
        "query_tokens": {
          "31380": 2.5134,
          "32470": 1.8921,
          "15678": 1.2045
        }
      }
    }
  }
}
```

> **Note**: The `sparse_vector` field type requires **integer token IDs** as keys (e.g., `"31380"`), not string tokens (e.g., `"한국"`). Use `encode_for_opensearch()` above for correct format.

## Evaluation

### Korean Retrieval Benchmarks

Evaluated on standard Korean retrieval benchmarks using OpenSearch with `neural_sparse` search. All differences vs. BM25 are statistically significant (paired t-test, p < 0.001).

| Benchmark | Queries | Corpus | Description |
|-----------|---------|--------|-------------|
| Ko-StrategyQA | 592 | 9,251 | Korean multi-hop retrieval (translated from StrategyQA) |
| MIRACL-ko | 213 | 10,000 | Wikipedia-based Korean document retrieval |
| Mr.TyDi-ko | 421 | 10,000 | Wikipedia-based Korean document retrieval |

### Performance Summary (Recall@1)

| Benchmark | BM25 | **Neural Sparse (Ours)** | Dense (BGE-M3) |
|-----------|------|--------------------------|----------------|
| Ko-StrategyQA | 53.7% | **62.2%** (+8.5pp) | 73.5% |
| MIRACL-ko | 44.1% | **62.0%** (+17.9pp) | 70.9% |
| Mr.TyDi-ko | 55.6% | **73.4%** (+17.8pp) | 84.1% |
| **Average** | 51.1% | **65.9%** (+14.7pp) | 76.2% |

### Detailed Metrics

| Benchmark | Method | R@1 | R@5 | R@10 | MRR | NDCG@10 | P50 Latency |
|-----------|--------|-----|-----|------|-----|---------|-------------|
| Ko-StrategyQA | BM25 | 53.7% | 75.3% | 81.9% | 0.626 | 0.673 | 8.2ms |
| Ko-StrategyQA | **Neural Sparse** | **62.2%** | **80.6%** | **83.6%** | **0.700** | **0.734** | **9.4ms** |
| Ko-StrategyQA | Dense (BGE-M3) | 73.5% | 87.3% | 89.4% | 0.795 | 0.819 | 11.8ms |
| MIRACL-ko | BM25 | 44.1% | 80.8% | 90.6% | 0.589 | 0.666 | 7.9ms |
| MIRACL-ko | **Neural Sparse** | **62.0%** | **89.7%** | **93.4%** | **0.733** | **0.783** | **9.5ms** |
| MIRACL-ko | Dense (BGE-M3) | 70.9% | 93.9% | 97.7% | 0.810 | 0.851 | 11.8ms |
| Mr.TyDi-ko | BM25 | 55.6% | 79.1% | 85.7% | 0.656 | 0.705 | 8.3ms |
| Mr.TyDi-ko | **Neural Sparse** | **73.4%** | **92.4%** | **94.8%** | **0.816** | **0.849** | **9.6ms** |
| Mr.TyDi-ko | Dense (BGE-M3) | 84.1% | 95.7% | 96.9% | 0.894 | 0.913 | 12.0ms |

### Comparison with Other Sparse Models

| Model | Parameters | Ko-StrategyQA R@1 | MIRACL-ko R@1 | Mr.TyDi-ko R@1 |
|-------|-----------|-------------------|---------------|-----------------|
| **sewoong/korean-neural-sparse-encoder** | **149M** | **62.2%** | **62.0%** | **73.4%** |
| opensearch-neural-sparse-encoding-multilingual-v1 | 110M | — | — | — |

### Hybrid Search Performance (Ko-StrategyQA)

Combining BM25 + Neural Sparse + Dense retrieval with linear interpolation:

| Method | R@1 | R@5 | R@10 | MRR | NDCG@10 |
|--------|-----|-----|------|-----|---------|
| BM25 only | 53.7% | 75.3% | 81.9% | 0.626 | 0.673 |
| Neural Sparse only | 62.2% | 80.6% | 83.6% | 0.700 | 0.734 |
| Dense (BGE-M3) only | 73.5% | 87.3% | 89.4% | 0.795 | 0.819 |
| **Hybrid (sparse=0.3, dense=0.7)** | **72.3%** | **87.5%** | **89.2%** | **0.788** | **0.814** |
| Hybrid (sparse=0.4, dense=0.6) | 71.8% | 87.0% | 89.4% | 0.784 | 0.811 |
| Hybrid (sparse=0.5, dense=0.5) | 70.3% | 86.3% | 89.0% | 0.773 | 0.802 |

## Sparsity Characteristics

| Property | Query | Document |
|----------|-------|----------|
| Avg. active dimensions | ~33 | ~54 |
| Sparsity rate | 99.93% | 99.89% |
| Vocabulary size | 50,000 | 50,000 |

The model produces ultra-sparse representations where only 0.07%~0.11% of the vocabulary dimensions are activated, enabling efficient inverted index storage and retrieval.

## Training Details

### Training Data

4.59M Korean triplets (query, positive document, hard negative) from 28 sources:

| Dataset | Samples | Ratio | Type |
|---------|---------|-------|------|
| AIHub News QA (#624) | 1,325,966 | 28.9% | News question-answering |
| OPUS-100 (ko-en) | 732,044 | 15.9% | Parallel corpus |
| kor-triplet-v1.0 | 681,896 | 14.8% | Retrieval triplets |
| mC4-ko | 475,292 | 10.3% | Web passage pairs |
| Wikipedia-ko | 328,733 | 7.2% | Wikipedia passage pairs |
| Korean NLI | 252,063 | 5.5% | Natural language inference |
| AIHub Dialog QA (#86) | 150,771 | 3.3% | Dialog-based QA |
| ko-wikidata-QA | 130,657 | 2.8% | Wikidata QA |
| OIG-smallchip2-ko | 117,887 | 2.6% | Instruction following |
| KorQuAD 2.0 | 80,914 | 1.8% | Machine reading comprehension |
| Others (18 datasets) | 317,384 | 6.9% | NLI, STS, classification, dialog |
| **Total** | **4,593,607** | **100%** | |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | skt/A.X-Encoder-base (ModernBERT) |
| Loss function | InfoNCE + FLOPS regularization |
| Temperature | 1.0 (sparse dot-product) |
| FLOPS lambda (query) | 0.01 |
| FLOPS lambda (document) | 0.003 |
| FLOPS warmup | 20,000 steps (quadratic schedule) |
| Learning rate | 5e-5 (cosine decay) |
| Warmup ratio | 0.06 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Effective batch size | 2,048 (64/GPU x 4 grad_accum x 8 GPUs) |
| Epochs | 25 |
| Mixed precision | BF16 |
| Query max length | 64 tokens |
| Document max length | 256 tokens |
| Seed | 42 |

### Hardware

| Component | Specification |
|-----------|---------------|
| GPU | 8x NVIDIA B200 (183GB VRAM each) |
| Total VRAM | 1,464 GB |
| Training time | ~24 hours |
| DDP | DistributedDataParallel (NCCL) |

### Framework Versions

| Framework | Version |
|-----------|---------|
| Python | 3.12 |
| PyTorch | 2.6 |
| Transformers | 4.48 |
| CUDA | 12.8 |

## Limitations

- **Korean-focused**: Optimized for Korean text; performance on other languages is not guaranteed.
- **Query length**: Best results with queries under 64 tokens. Longer queries are truncated.
- **Term expansion scope**: SPLADE expansion is bounded by the 50K vocabulary. Out-of-vocabulary terms fall back to subword tokenization.
- **No built-in reranking**: For best results, combine with a cross-encoder reranker.

## Citation

```bibtex
@software{korean-neural-sparse-encoder,
  author       = {Sewoong Kim},
  title        = {korean-neural-sparse-encoder},
  subtitle     = {Korean SPLADE-max Sparse Encoder for Neural Sparse Retrieval},
  publisher    = {Hugging Face},
  year         = {2026},
  month        = {2},
  version      = {1.0.0},
  url          = {https://huggingface.co/sewoong/korean-neural-sparse-encoder}
}
```

## References

- [SPLADE v2: From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective (Formal et al., 2022)](https://arxiv.org/abs/2205.04733)
- [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking (Formal et al., 2021)](https://arxiv.org/abs/2107.05720)
- [Minimizing FLOPs to Learn Efficient Sparse Representations (Paria et al., 2020)](https://arxiv.org/abs/2004.05665)
- [ModernBERT: A Modern BERT Architecture (Warner et al., 2024)](https://arxiv.org/abs/2412.13663)

## Author

**Sewoong Kim** - February 2026

## License

Apache 2.0
