# OpenSearch Korean-English Neural Sparse Model

Cross-lingual neural sparse retrieval model for OpenSearch.

## Overview

This repository contains training code for a Korean-English cross-lingual SPLADE-doc model. The model enables neural sparse search with lexical expansion across Korean and English terms.

## Model Specification

| Property | Value |
|----------|-------|
| Base Model | `xlm-roberta-large` |
| Parameters | 560M |
| Vocabulary Size | 250,002 tokens |
| Max Sequence Length | 512 |
| Supported Languages | Korean, English |
| Output Format | Sparse vector (rank_features) |

## Architecture

The model implements SPLADE-doc (Sparse Lexical AnD Expansion) architecture:

```
Document → XLM-RoBERTa Encoder → log(1 + ReLU(logits)) → Max Pooling → Sparse Vector
```

Key characteristics:
- Document-only mode: No query encoder required at search time
- Inference-free queries: Query encoding uses IDF lookup only
- Cross-lingual expansion: Korean terms activate related English tokens and vice versa

## Training Data

### Data Sources

| Source | Description | Pairs |
|--------|-------------|-------|
| MUSE | Facebook bilingual dictionary | ~20,000 |
| Wikidata | Entity labels (ko/en) | ~10,000 |
| IT Terminology | Technical terms | ~80 |

### Data Format

All training data consists of single-token pairs only. Multi-word phrases are not supported.

```json
{"ko": "프로그램", "en": "program", "source": "muse"}
{"ko": "네트워크", "en": "network", "source": "muse"}
{"ko": "머신러닝", "en": "machine", "source": "it_terminology"}
{"ko": "머신러닝", "en": "learning", "source": "it_terminology"}
```

### Data Processing Pipeline

1. **Data Ingestion** (`00_data_ingestion.ipynb`)
   - Collect term pairs from MUSE, Wikidata, IT terminology
   - Filter to single tokens only (no spaces allowed)
   - Validate Korean/English token format

2. **Data Preparation** (`01_data_preparation.ipynb`)
   - Generate embeddings using `intfloat/multilingual-e5-large`
   - K-means clustering for semantic grouping
   - Extract 1:N mappings with dot product similarity scores
   - Configuration: `similarity_threshold=0.8`, `max_targets=8`

## Training

### Configuration

```yaml
model_name: xlm-roberta-large
batch_size: 64
gradient_accumulation_steps: 2
num_epochs: 15
learning_rate: 3e-6
warmup_ratio: 0.1
lambda_positive: 1.0
lambda_negative: 1.0
lambda_sparsity: 0.01
similarity_threshold: 0.8
max_targets_per_source: 8
```

### Loss Function

The training uses similarity-weighted loss:

```
L = L_positive + λ_neg * L_negative + λ_sparse * L_sparsity

L_positive = -Σ sim_i * log(σ(score_i))    # Activate target tokens
L_negative = -Σ log(1 - σ(score_j))         # Suppress non-target tokens
L_sparsity = mean(score)                    # Encourage sparsity
```

### Training Pipeline

```bash
# 1. Data collection
jupyter notebook notebooks/opensearch-neural-v19/00_data_ingestion.ipynb

# 2. Data preparation
jupyter notebook notebooks/opensearch-neural-v19/01_data_preparation.ipynb

# 3. Model training
jupyter notebook notebooks/opensearch-neural-v19/02_training.ipynb

# 4. Inference test
jupyter notebook notebooks/opensearch-neural-v19/03_inference_test.ipynb
```

## Output

### Model Files

```
outputs/v19_xlm_large/
├── checkpoint.pt          # Model weights
├── tokenizer/             # XLM-RoBERTa tokenizer
└── training_curves.png    # Loss visualization
```

### Checkpoint Format

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "best_loss": best_loss,
    "config": {
        "model_name": "xlm-roberta-large",
        "max_length": 512,
        "vocab_size": 250002
    }
}
```

## Usage

### Load Trained Model

```python
import torch
from transformers import AutoTokenizer
from src.model.splade_model import SPLADEDoc

# Load checkpoint
checkpoint = torch.load("outputs/v19_xlm_large/checkpoint.pt")
config = checkpoint["config"]

# Initialize model
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
model = SPLADEDoc(model_name=config["model_name"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Encode document
def encode_document(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        sparse_vec = model(inputs["input_ids"], inputs["attention_mask"])
    return sparse_vec
```

### OpenSearch Integration

Register model with ML Commons:

```json
POST /_plugins/_ml/models/_register
{
    "name": "korean-english-neural-sparse-v19",
    "version": "1.0.0",
    "model_format": "TORCH_SCRIPT",
    "function_name": "SPARSE_ENCODING"
}
```

Create index with rank_features:

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

Neural sparse query:

```json
POST /documents/_search
{
    "query": {
        "neural_sparse": {
            "sparse_embedding": {
                "query_text": "검색 쿼리",
                "model_id": "<model_id>"
            }
        }
    }
}
```

## Project Structure

```
opensearch-neural-pre-train/
├── notebooks/
│   └── opensearch-neural-v19/
│       ├── 00_data_ingestion.ipynb      # Data collection
│       ├── 01_data_preparation.ipynb    # Preprocessing
│       ├── 02_training.ipynb            # Model training
│       └── 03_inference_test.ipynb      # Evaluation
├── src/
│   └── model/
│       └── splade_model.py              # SPLADE-doc implementation
├── dataset/
│   └── v19_high_quality/
│       ├── term_pairs.jsonl             # Raw term pairs
│       └── term_mappings.jsonl          # Processed 1:N mappings
├── outputs/
│   └── v19_xlm_large/                   # Trained model
└── scripts/
    └── collect_term_data_v19.py         # CLI data collection
```

## Requirements

- Python 3.12+
- PyTorch 2.0+
- transformers 4.35+
- CUDA 11.8+ (for GPU training)

```bash
pip install torch transformers tqdm scikit-learn matplotlib
```

## References

- [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)
- [OpenSearch Neural Sparse Search](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/)
- [XLM-RoBERTa](https://huggingface.co/xlm-roberta-large)
- [MUSE Bilingual Dictionaries](https://github.com/facebookresearch/MUSE)

## License

Apache License 2.0
