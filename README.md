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

v21.3 is a Korean Neural Sparse model with improved data quality using algorithmic/statistical filtering methods.

### Data Statistics

| Metric | Value |
|--------|-------|
| Corpus Documents | 646,700 |
| Term Embeddings | 150,000 |
| Raw Synonym Pairs | 75,732 |
| Filtered Synonym Pairs | 66,070 (87.2%) |
| Removed Pairs | 9,662 (12.8%) |
| Unique Anchors | 28,371 |
| Training Triplets | 315,729 |
| Validation Triplets | 35,081 |

---

### Data Preprocessing Pipeline

#### Step 1: Data Ingestion (`00_data_ingestion.ipynb`)

Collects text from 14 Korean datasets across multiple domains:

| Domain | Dataset | Description | Documents |
|--------|---------|-------------|-----------|
| Encyclopedia | Wikipedia (ko) | General knowledge | ~500,000 |
| QA | KLUE-MRC | Machine reading comprehension | ~17,000 |
| QA | KorQuAD 1.0/2.0 | Korean question answering | ~70,000 |
| Legal | Korean Law Precedents | Case law, legal terminology | ~30,000 |
| Medical | KorMedMCQA (doctor) | Doctor licensing exam | ~6,000 |
| Medical | KorMedMCQA (nurse) | Nurse licensing exam | ~6,000 |
| Medical | KorMedMCQA (pharm) | Pharmacist licensing exam | ~6,000 |
| Medical | KorMedMCQA (dentist) | Dentist licensing exam | ~3,000 |
| Dialogue | NSMC | Naver movie reviews | ~150,000 |
| Dialogue | KorHate | Hate speech comments | ~8,000 |

**Processing Steps:**

1. **Text Extraction**: Extract text fields from each dataset
2. **Noun Extraction**: Use Kiwi tokenizer to extract nouns (NNG, NNP, NNB, SL, SH tags)
3. **Embedding Generation**: Generate 1024-dim embeddings using `intfloat/multilingual-e5-large`
4. **Synonym Mining**: Find similar term pairs using cosine similarity (threshold > 0.85)

```python
# Kiwi tokenizer configuration
VALID_POS_TAGS = {'NNG', 'NNP', 'NNB', 'SL', 'SH'}  # Nouns only
MIN_TERM_LENGTH = 2
MAX_TERM_LENGTH = 15
```

---

#### Step 2: Noise Filtering (`01_noise_filtering.ipynb`)

Raw embedding-based synonym pairs contain significant noise (truncations, false positives). Three algorithmic filters are applied in an ensemble:

```
┌─────────────────────────────────────────────────────────────┐
│              Raw Synonym Pairs (75,732)                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │   IG    │  │   PMI   │  │   CE    │
    │ Filter  │  │ Filter  │  │ Filter  │
    │ (90%)   │  │ (34%)   │  │ (90%)   │
    └────┬────┘  └────┬────┘  └────┬────┘
         │            │            │
         └────────────┼────────────┘
                      ▼
              ┌───────────────┐
              │ Majority Vote │
              │   (≥2/3 pass) │
              └───────┬───────┘
                      ▼
              Filtered Pairs (66,070)
```

##### Filter 1: Information Gain (IG)

Measures semantic expansion value using KNN entropy estimation (Kozachenko-Leonenko estimator).

**Algorithm:**
```python
IG(source → target) = H(target) - H(target|source)

# H(target): Marginal entropy in entire embedding space
# H(target|source): Conditional entropy within source's k-nearest neighbors
```

**Configuration:**
```python
k_entropy = 10           # k for entropy estimation
k_neighborhood = 50      # k for neighborhood definition
percentile_threshold = 10.0  # Remove bottom 10%
```

**What it catches:**
- Truncation pairs: `다음날오전 → 다음날` (low IG)
- Case changes: `iPhone → iphone` (low IG)

**Results:**
- Threshold: 2.1905
- Pass rate: 90.0% (68,158 pairs)

##### Filter 2: Pointwise Mutual Information (PMI)

Measures co-occurrence probability in the corpus using Laplace-smoothed PMI.

**Algorithm:**
```python
PMI(x, y) = log2(P(x,y) / (P(x) * P(y)))

# P(x,y): Joint probability (co-occurrence in same sentence)
# P(x), P(y): Marginal probabilities
```

**Co-occurrence Matrix Construction:**
```python
window_type = "sentence"      # Sentence-level co-occurrence
min_term_freq = 5             # Minimum term frequency
max_vocab_size = 120,000      # Maximum vocabulary
symmetric = True              # Bidirectional co-occurrence
laplace_smoothing = 1.0       # Laplace smoothing factor
```

**Statistics:**
- Vocabulary size: 120,000 terms
- Total windows: 5,599,725
- Total co-occurrences: 66,162,183

**What it catches:**
- False positives with high embedding similarity but no corpus co-occurrence
- Example: Random similar-sounding terms that never appear together

**Results:**
- Threshold: 0.4312
- Valid scores: 28,642 (47,090 OOV)
- Pass rate: 34.0% (25,778 pairs)

##### Filter 3: Cross-Encoder (CE)

Validates semantic similarity using a neural cross-encoder model.

**Model:**
```python
model = "BAAI/bge-reranker-v2-m3"
normalize = True
use_fp16 = True
batch_size = 256
```

**What it catches:**
- Semantically dissimilar pairs that have similar surface forms
- Example: `동계올림픽 → 하계올림픽` (winter vs summer - opposites)

**Results:**
- Threshold: 0.4287
- Pass rate: 90.0% (68,158 pairs)

##### Ensemble Voting

A pair is **kept** if it passes at least 2 of 3 filters:

| Filters Passed | Count | Percentage | Decision |
|----------------|-------|------------|----------|
| 0/3 | 666 | 0.9% | REMOVE |
| 1/3 | 8,996 | 11.9% | REMOVE |
| 2/3 | 45,112 | 59.6% | KEEP |
| 3/3 | 20,958 | 27.7% | KEEP |

**Filter Combination Analysis:**

| IG | PMI | CE | Count | Decision |
|----|-----|-----|-------|----------|
| ✓ | ✗ | ✓ | 40,751 | KEEP |
| ✓ | ✓ | ✓ | 20,958 | KEEP |
| ✗ | ✗ | ✓ | 4,564 | REMOVE |
| ✓ | ✗ | ✗ | 3,973 | REMOVE |
| ✓ | ✓ | ✗ | 2,476 | KEEP |
| ✗ | ✓ | ✓ | 1,885 | KEEP |

**Examples of Correctly Removed Pairs:**
```
다음날오전 → 다음날오후    (opposite: morning vs afternoon)
동계올림픽 → 하계올림픽    (opposite: winter vs summer)
북쪽출구 → 남쪽출구       (opposite: north vs south)
MBC대하드라마 → SBS대하드라마 (different companies)
```

**Examples of Correctly Kept Pairs:**
```
Keyboard → 키보드          (EN-KO translation)
bytes → byte              (singular/plural)
제천 → 제천시              (place name variant)
화산 → 화산학              (term + field)
```

---

#### Step 3: Data Preparation (`02_data_preparation.ipynb`)

Creates triplet dataset with difficulty-balanced hard negatives for contrastive learning.

##### Hard Negative Mining

For each anchor, negatives are sampled from different similarity ranges:

| Difficulty | Similarity Range | Target Ratio | Actual Ratio |
|------------|------------------|--------------|--------------|
| Easy | 0.3 - 0.5 | 33% | 27.3% |
| Medium | 0.5 - 0.7 | 33% | 20.1% |
| Hard | 0.7 - 0.9 | 34% | 52.6% |

**Algorithm:**
```python
def mine_hard_negatives(anchor, positives, embeddings):
    # Compute similarity to all terms
    similarities = cosine_similarity(anchor_emb, all_embeddings)

    # Categorize candidates by difficulty
    easy = [t for t, sim in candidates if 0.3 <= sim < 0.5]
    medium = [t for t, sim in candidates if 0.5 <= sim < 0.7]
    hard = [t for t, sim in candidates if 0.7 <= sim < 0.9]

    # Sample with balanced ratios
    return sample_balanced(easy, medium, hard, n=5)
```

##### Triplet Format

```python
{
    "anchor": "인공지능",
    "positive": "AI",           # Ground truth synonym
    "negative": "자동화",       # Hard negative (similar but different)
    "negative_similarity": 0.72,
    "difficulty": "hard"
}
```

##### Dataset Split

| Split | Triplets | Ratio |
|-------|----------|-------|
| Train | 315,729 | 90% |
| Validation | 35,081 | 10% |

##### Output Files

```
dataset/v21.3_filtered_enhanced/
├── raw_synonym_pairs.jsonl        # 75,732 raw pairs
├── filtered_synonym_pairs.jsonl   # 66,070 filtered pairs
├── removed_synonym_pairs.jsonl    # 9,662 removed pairs
├── filtering_stats.json           # Filter statistics
├── corpus_texts.jsonl             # 646,700 corpus documents
├── term_embeddings.npy            # 150,000 x 1024 embeddings
├── term_list.json                 # Term vocabulary
├── cooccurrence_matrix/           # PMI co-occurrence data
├── triplet_dataset/               # HuggingFace Dataset
├── train_triplets.jsonl           # Training data
└── val_triplets.jsonl             # Validation data
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
