# OpenSearch Korean Neural Sparse Model

Neural sparse retrieval model for Korean text in OpenSearch.

## Overview

This repository contains training code for Korean SPLADE-doc models. The models enable neural sparse search with synonym expansion for Korean terms.

### Model Versions

| Version | Description | Key Features |
|---------|-------------|--------------|
| v19 | Cross-lingual (KO-EN) | XLM-RoBERTa, bilingual dictionary |
| v21.2 | Korean synonym expansion | Legal/Medical domain, A.X-Encoder |
| **v21.3** | **Enhanced Korean model** | **Noise filtering, balanced hard negatives, ranking metrics** |

---

## v21.3 Korean Neural Sparse Encoder (Latest)

v21.3 is a Korean Neural Sparse model with improved data quality and evaluation metrics using algorithmic/statistical methods.

### Improvements over v21.2

| Item | v21.2 | v21.3 |
|------|-------|-------|
| Data Noise | ~50% (BPE noise) | **< 10%** (3-stage filtering) |
| Evaluation Metrics | 100% saturated (Binary) | **Recall@K, MRR** (non-saturating) |
| Medical Data | Load failure | **4 configs loaded successfully** |
| Hard Negatives | Random sampling | **Difficulty-balanced sampling** |

### Data Preprocessing Pipeline

#### 1. Data Ingestion (`00_data_ingestion.ipynb`)

Collects text from 14 Korean datasets:

| Domain | Dataset | Description |
|--------|----------|------|
| Encyclopedia | Wikipedia (ko) | General knowledge |
| QA | KLUE-MRC, KorQuAD | Question-answering context |
| Legal | Korean Law Precedents | Case law, legal terminology |
| **Medical** | **KorMedMCQA (4 configs)** | Doctor/Nurse/Pharmacist/Dentist licensing exams |
| Dialogue | NSMC, KorHate | Reviews, comments |

**Medical Data Loading (v21.2 bug fix):**
```python
# v21.2: Failed (config not specified)
load_dataset("sean0042/KorMedMCQA", split="train")  # Error

# v21.3: Success (config explicitly specified)
medical_configs = ["dentist", "doctor", "nurse", "pharm"]
for config in medical_configs:
    load_dataset("sean0042/KorMedMCQA", config, split="train")
```

#### 2. Noise Filtering (`01_noise_filtering.ipynb`)

Applies 3 algorithmic filters in an ensemble:

```
┌─────────────────────────────────────────────────────────────┐
│                    Raw Synonym Pairs                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │   IG    │  │   PMI   │  │   CE    │
    │ Filter  │  │ Filter  │  │ Filter  │
    └────┬────┘  └────┬────┘  └────┬────┘
         │            │            │
         └────────────┼────────────┘
                      ▼
              ┌───────────────┐
              │ Majority Vote │
              │   (2/3 pass)  │
              └───────┬───────┘
                      ▼
              Filtered Pairs
```

**Filtering Methods:**

| Filter | Algorithm | Purpose | Threshold |
|------|----------|------|-----------|
| **IG** | KNN Entropy (Kozachenko-Leonenko) | Remove truncation/case changes | Bottom 10% |
| **PMI** | Co-occurrence probability (Laplace smoothing) | Remove false positives | Bottom 10% |
| **CE** | Cross-Encoder (bge-reranker-v2-m3) | Remove semantically dissimilar pairs | Bottom 10% |

**Information Gain Calculation:**
```python
IG(source → target) = H(target) - H(target|source)

# H(target): Entropy of target in the entire corpus
# H(target|source): Conditional entropy of target within source neighbors
# High IG = Meaningful semantic expansion (keep)
# Low IG = Simple truncation (remove)
```

**PMI Calculation:**
```python
PMI(x, y) = log(P(x,y) / (P(x) * P(y)))

# High PMI = High co-occurrence frequency (true synonyms)
# Low PMI = Independent occurrence (false positives with only high embedding similarity)
```

**Ensemble Decision:**
- Must pass **2 or more of 3 filters** to be retained (majority voting)
- **Percentile-based** automatic threshold determination (no hardcoded values)

#### 3. Data Preparation (`02_data_preparation.ipynb`)

**Difficulty-balanced Hard Negative Mining:**

| Difficulty | Similarity Range | Ratio | Characteristics |
|--------|-------------|------|------|
| Easy | 0.3 - 0.5 | 33% | Easily distinguishable |
| Medium | 0.5 - 0.7 | 33% | Medium difficulty |
| Hard | 0.7 - 0.9 | 33% | Difficult negatives |

```python
# Triplet format: (anchor, positive, negative)
{
    "anchor": "인공지능",
    "positive": "AI",           # Synonym
    "negative": "자동화",       # Hard negative (similar but different)
    "difficulty": "hard"
}
```

### Training (`03_training.ipynb`)

**Model Architecture:**
```
Input → A.X-Encoder-base → log(1 + ReLU(logits)) → Max Pooling → Sparse Vector
```

**Loss Function:**
```python
L_total = λ_self * L_self           # Self-reconstruction
        + λ_synonym * L_positive    # Synonym activation
        + λ_margin * L_triplet      # Triplet margin loss
        + λ_flops * L_flops         # Sparsity regularization
```

**Hyperparameters (v21.3):**
```yaml
model_name: skt/A.X-Encoder-base
batch_size: 64
num_epochs: 25
learning_rate: 3e-6
lambda_self: 4.0
lambda_synonym: 10.0
lambda_margin: 2.5
lambda_flops: 8e-3
target_margin: 1.5
```

### Evaluation Metrics (Solving v21.2 Saturation Problem)

| Metric | Description | v21.2 Problem | v21.3 Solution |
|------|------|------------|------------|
| Recall@K | Ratio of correct synonyms in Top-K | N/A | ✓ Implemented |
| MRR | Mean reciprocal rank of first correct answer | N/A | ✓ Implemented |
| nDCG | Rank-weighted score | N/A | ✓ Implemented |
| Binary Accuracy | Whether correct answer is included | 100% saturated | Not used |

```python
# Recall@K
Recall@K = |Retrieved@K ∩ Relevant| / |Relevant|

# MRR (Mean Reciprocal Rank)
MRR = (1/|Q|) * Σ (1/rank_i)
```

### Execution

```bash
cd notebooks/opensearch-neural-v21.3/

# 1. Data ingestion (including medical data)
jupyter nbconvert --execute 00_data_ingestion.ipynb

# 2. Noise filtering (takes 30-60 minutes)
jupyter nbconvert --execute 01_noise_filtering.ipynb

# 3. Hard Negative Mining
jupyter nbconvert --execute 02_data_preparation.ipynb

# 4. Model training (GPU required)
jupyter nbconvert --execute 03_training.ipynb

# 5. Inference test
jupyter nbconvert --execute 04_inference_test.ipynb
```

### Output Directories

```
dataset/v21.3_filtered_enhanced/
├── raw_synonym_pairs.jsonl        # Raw synonym pairs
├── filtered_synonym_pairs.jsonl   # Filtered synonym pairs
├── removed_synonym_pairs.jsonl    # Removed pairs (for analysis)
├── filtering_stats.json           # Filtering statistics
├── triplet_dataset/               # HuggingFace Dataset
├── train_triplets.jsonl           # Training data
└── val_triplets.jsonl             # Validation data

outputs/v21.3_korean_enhanced/
├── best_model.pt                  # Best performance model
├── final_model.pt                 # Final model
├── training_history.json          # Training history
└── training_curves.png            # Training curves
```

### Core Modules

| Module | Path | Description |
|------|------|------|
| Information Gain | `src/information_gain.py` | KNN Entropy-based IG calculation |
| PMI | `src/pmi/` | Co-occurrence matrix and PMI calculation |
| Ranking Metrics | `src/evaluation/ranking_metrics.py` | Recall@K, MRR, nDCG |

---

## v19 Cross-lingual Model (Legacy)

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
