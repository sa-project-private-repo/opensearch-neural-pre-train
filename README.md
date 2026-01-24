# OpenSearch Korean Neural Sparse Model

Korean SPLADE-doc neural sparse retrieval model for OpenSearch.

## Overview

This repository contains training code and benchmarks for Korean neural sparse search models. The models enable semantic sparse search with synonym expansion for Korean terms.

### Latest Version: V25 (XLM-RoBERTa with IDF-Aware FLOPS)

| Property | Value |
|----------|-------|
| Base Model | `xlm-roberta-base` |
| Parameters | 278M |
| Vocabulary | 250,002 tokens |
| Max Length | 512 |
| Teacher | BAAI/bge-m3 |

**Key Features (V25):**
- IDF-Aware FLOPS: BM25-style IDF weighting (mandatory)
- Korean Stopword Masking: 163 particles/endings penalized
- Knowledge Distillation: BGE-M3 dense embeddings
- Semantic Token Ratio Monitoring: Tracks semantic vs stopword activation

**Target Metrics:**
- Semantic tokens in top-10: 80%+ (vs V24's 30%)
- Stopword activation: <0.5 (vs V24's 5.8)

### Previous Version: v21.4

| Property | Value |
|----------|-------|
| Base Model | `skt/A.X-Encoder-base` |
| Parameters | 149M |
| Vocabulary | 49,999 tokens |
| Max Length | 64 |

**Key Improvements (v21.4):**
- Curriculum Learning: 3-phase training (single-terms -> balanced -> full)
- Dynamic Lambda Self: 8.0 for single-term, 4.0 for sentences
- Minimum Activation Loss: Ensures meaningful top-k activations

---

## Quick Start

> **V25 Training Guide**: 단계별 학습 가이드는 [GUIDE.md](./GUIDE.md)를 참조하세요.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn

# Load model
tokenizer = AutoTokenizer.from_pretrained("huggingface/v21.4")
model = AutoModelForMaskedLM.from_pretrained("huggingface/v21.4")

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

---

## Project Structure

```
opensearch-neural-pre-train/
├── notebooks/
│   ├── opensearch-neural-v21.4/          # Latest version
│   │   ├── 00_huggingface_data_loading.ipynb
│   │   ├── 01_data_augmentation.ipynb
│   │   ├── 02_data_preparation.ipynb
│   │   ├── 03_training.ipynb
│   │   └── 04_evaluation.ipynb
│   └── opensearch-neural-v19/            # Legacy cross-lingual
├── benchmark/                             # Benchmark framework
│   ├── runner.py                          # Main benchmark runner
│   ├── encoders.py                        # Dense/Sparse encoders
│   ├── searchers.py                       # BM25/Semantic/Sparse/Hybrid
│   ├── metrics.py                         # Recall, MRR, nDCG
│   └── config.py                          # Benchmark configuration
├── src/
│   ├── model/                             # Model implementations
│   ├── evaluation/                        # Ranking metrics
│   └── pmi/                               # Co-occurrence analysis
├── data/                                  # Training data
├── huggingface/                           # Model releases
│   ├── v21.4/                             # Latest model
│   └── v1/                                # Initial release
└── outputs/                               # Training outputs
```

---

## Benchmark

Compare search methods against a validation dataset.

### Search Methods

| Method | Description | Index Type |
|--------|-------------|------------|
| BM25 | Lexical keyword search | text |
| Semantic | Dense vector k-NN (bge-m3) | knn_vector |
| Neural Sparse | SPLADE sparse vectors | rank_features |
| Hybrid | BM25 + k-NN combination | text + knn_vector |

### Running Benchmarks

```bash
# Full benchmark (with indexing)
python -m benchmark.runner --sample-size 1000 --output-dir outputs/benchmark

# Skip indexing (data already indexed)
python -m benchmark.runner --sample-size 1000 --skip-setup --output-dir outputs/benchmark

# Parallel execution (all methods simultaneously)
python -m benchmark.runner --sample-size 1000 --skip-setup --parallel --output-dir outputs/benchmark
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--sample-size` | Number of queries | 2000 |
| `--skip-setup` | Skip index creation | False |
| `--parallel` | Run all methods in parallel | False |
| `--cleanup` | Delete indices after benchmark | False |
| `--output-dir` | Output directory | outputs/benchmark |

### Output

```
outputs/benchmark/
├── report.md          # Summary report
├── metrics.json       # Detailed metrics (Recall@K, MRR, nDCG, latency)
└── benchmark.log      # Execution log
```

---

## Training Pipeline

### V25 Training (Recommended)

V25 uses XLM-RoBERTa with IDF-aware FLOPS to suppress grammatical particles and promote semantic tokens.

> 상세 가이드: [GUIDE.md](./GUIDE.md)

#### Full Training Pipeline

```bash
# 1. Data preparation (BGE-M3 hard negatives mining)
make prepare-v25-data

# 2. IDF weights computation (for FLOPS regularization)
make prepare-v25-idf

# 3. Verify IDF setup
make train-v25-verify

# 4. Start training
make train-v25

# 5. Resume from checkpoint (if interrupted)
make train-v25-resume
```

#### Background Execution

For long-running training on remote servers:

```bash
# Method 1: nohup (simple)
nohup make train-v25 > outputs/train_v25/training.log 2>&1 &
echo $!  # Save PID for later

# Method 2: tmux (recommended)
tmux new -d -s train 'make train-v25'
tmux attach -t train    # Attach to session
# Ctrl+b, d              # Detach from session
tmux kill-session -t train  # Kill session
```

#### TensorBoard Monitoring

```bash
# Start TensorBoard (foreground)
tensorboard --logdir outputs/train_v25/tensorboard --port 6006 --bind_all

# Start TensorBoard (background)
nohup tensorboard --logdir outputs/train_v25/tensorboard --port 6006 --bind_all > /tmp/tensorboard.log 2>&1 &

# Or use make target
make tensorboard-v25
```

**Remote Access (SSH Tunneling):**
```bash
# On local machine
ssh -L 6006:localhost:6006 ec2-user@<EC2-IP>
# Then open http://localhost:6006 in browser
```

#### Training Status Check

```bash
# View live logs
make logs-v25

# Or directly
tail -f outputs/train_v25/training.log

# Check GPU usage
nvidia-smi -l 1
```

**V25 Loss Function:**
```
L_total = λ_infonce * L_infonce      # Contrastive learning
        + λ_self * L_self            # Self-reconstruction
        + λ_positive * L_positive    # Positive alignment
        + λ_flops * L_idf_flops      # IDF-weighted sparsity (NEW)
        + λ_min_act * L_min_act      # Minimum activation
        + λ_kd * L_kd                # Knowledge distillation
```

**IDF-Aware FLOPS:**
- High IDF (rare tokens like 서울, 맛있는) → Low penalty
- Low IDF (common tokens like 을, 는) → High penalty (+ 5x stopword multiplier)

### v21.4 Training (Legacy)

```bash
cd notebooks/opensearch-neural-v21.4/

# 1. Load data from HuggingFace
jupyter nbconvert --execute 00_huggingface_data_loading.ipynb

# 2. Data augmentation
jupyter nbconvert --execute 01_data_augmentation.ipynb

# 3. Create triplet dataset
jupyter nbconvert --execute 02_data_preparation.ipynb

# 4. Train model (GPU required)
jupyter nbconvert --execute 03_training.ipynb

# 5. Evaluate
jupyter nbconvert --execute 04_evaluation.ipynb
```

### Model Architecture

```
Document -> A.X-Encoder -> log(1 + ReLU(logits)) -> Max Pooling -> Sparse Vector
```

### Loss Function

```
L_total = λ_self * L_self           # Self-reconstruction
        + λ_synonym * L_positive    # Synonym activation
        + λ_margin * L_triplet      # Triplet margin loss
        + λ_flops * L_flops         # Sparsity regularization
        + λ_min_act * L_min_act     # Minimum activation (v21.4)
```

### Hyperparameters (v21.4)

```yaml
model_name: skt/A.X-Encoder-base
batch_size: 64
num_epochs: 30
learning_rate: 3e-6
lambda_self: 4.0 (8.0 for single-term)
lambda_synonym: 10.0
lambda_margin: 2.5
lambda_flops: 8e-3
lambda_min_act: 0.1
target_margin: 1.5
```

---

## Data Pipeline

### Data Sources

Training data is loaded from HuggingFace datasets:

| Domain | Sources |
|--------|---------|
| Encyclopedia | Wikipedia (ko) |
| QA | KLUE-MRC, KorQuAD 1.0/2.0 |
| Legal | Korean Law Precedents |
| Medical | KorMedMCQA (doctor, nurse, pharm, dentist) |

### Training Data Format

Training data uses JSONL (JSON Lines) format where each line is a triplet:

```
data/v21.4/
├── training_triplets.jsonl      # Main training set (~423K triplets)
├── validation_triplets.jsonl    # Validation set (~47K triplets)
├── phase1_single_term_focus_triplets.jsonl  # Curriculum phase 1
├── phase2_balanced_triplets.jsonl           # Curriculum phase 2
└── phase3_full_triplets.jsonl               # Curriculum phase 3
```

#### Triplet Schema

| Field | Type | Description |
|-------|------|-------------|
| `anchor` | string | Input text (query or term) |
| `positive` | string | Semantically similar text (synonym, paraphrase) |
| `negative` | string | Hard negative (different meaning) |
| `difficulty` | string | `easy`, `medium`, `hard` |
| `length_class` | string | `single_term`, `short_phrase`, `sentence` |
| `pair_type` | string | Data source identifier |

#### Examples

**Single-term synonym (Korean):**
```json
{"anchor": "추천", "positive": "권장", "negative": "반대", "difficulty": "easy", "length_class": "single_term", "pair_type": "single_term"}
```

**Spelling variation:**
```json
{"anchor": "인터컨티넨탈", "positive": "인터콘티넨탈", "negative": "힐튼", "difficulty": "easy", "length_class": "short_phrase", "pair_type": "original"}
```

**Question-Answer pair (KorQuAD):**
```json
{"anchor": "대한민국의 수도는 어디인가?", "positive": "서울은 대한민국의 수도이다", "negative": "부산은 항구도시이다", "difficulty": "medium", "length_class": "sentence", "pair_type": "korquad"}
```

**Medical terminology:**
```json
{"anchor": "당뇨병", "positive": "diabetes mellitus", "negative": "고혈압", "difficulty": "medium", "length_class": "single_term", "pair_type": "original"}
```

**MS MARCO Korean:**
```json
{"anchor": "비타민D 결핍 증상", "positive": "비타민D가 부족하면 뼈가 약해지고 피로감이 증가합니다", "negative": "비타민C는 면역력 강화에 도움이 됩니다", "difficulty": "hard", "length_class": "sentence", "pair_type": "msmarco_ko"}
```

#### Data Distribution

| Pair Type | Count | Description |
|-----------|-------|-------------|
| original | 151K | Wikipedia entities, synonyms |
| msmarco_ko | 97K | MS MARCO Korean translation |
| korquad | 54K | KorQuAD question-context pairs |
| korquad_context | 54K | KorQuAD context-question pairs |
| naver_news | 35K | News article pairs |
| klue_nli | 15K | KLUE NLI entailment pairs |
| klue_sts | 11K | KLUE STS similar sentences |
| kobest_copa | 6K | KoBEST COPA reasoning pairs |
| single_term | 448 | Explicit single-term synonyms |

#### Curriculum Learning Phases

v21.4 uses 3-phase curriculum learning:

| Phase | Epochs | Focus | Lambda Self |
|-------|--------|-------|-------------|
| Phase 1 | 1-10 | Single-term pairs (100%) | 8.0 |
| Phase 2 | 11-20 | Balanced mix (50% single-term) | 6.0 |
| Phase 3 | 21-30 | Full dataset | 4.0 |

---

## OpenSearch Integration

### Create Index

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

### Neural Sparse Query

```json
POST /documents/_search
{
    "query": {
        "bool": {
            "should": [
                {"rank_feature": {"field": "sparse_embedding.당뇨병", "boost": 2.5}},
                {"rank_feature": {"field": "sparse_embedding.치료", "boost": 1.8}},
                {"rank_feature": {"field": "sparse_embedding.방법", "boost": 1.2}}
            ]
        }
    }
}
```

---

## Version History

| Version | Description | Status |
|---------|-------------|--------|
| **V25** | XLM-RoBERTa + IDF-aware FLOPS + Korean stopword masking | **Latest** |
| V24 | XLM-RoBERTa + BGE-M3 teacher (IDF config exists but inactive) | Deprecated |
| v21.4 | Curriculum learning, dynamic lambda, min activation (KoBERT) | Stable |
| v19 | Cross-lingual (KO-EN) with XLM-RoBERTa | Legacy |

<details>
<summary><b>v19 Cross-lingual Details</b></summary>

### Model Specification

| Property | Value |
|----------|-------|
| Base Model | `xlm-roberta-large` |
| Parameters | 560M |
| Vocabulary | 250,002 tokens |
| Languages | Korean, English |

### Data Sources

| Source | Description | Pairs |
|--------|-------------|-------|
| MUSE | Facebook bilingual dictionary | ~20,000 |
| Wikidata | Entity labels (ko/en) | ~10,000 |
| IT Terminology | Technical terms | ~80 |

</details>

---

## Requirements

- Python 3.12+
- PyTorch 2.0+
- transformers 4.35+
- CUDA 11.8+ (for GPU training)

```bash
pip install torch transformers sentence-transformers opensearch-py tqdm scikit-learn matplotlib
```

---

## References

- [SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720)
- [OpenSearch Neural Sparse Search](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/)
- [A.X-Encoder](https://huggingface.co/skt/A.X-Encoder-base)

## License

Apache License 2.0
