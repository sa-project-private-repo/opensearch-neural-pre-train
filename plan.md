# Neural Sparse Model Pre-training with Wikipedia Synonym Data

## 📋 Project Overview

**Goal:** Pre-train Neural Sparse model using Korean/English synonym data from Wikipedia

**Key Objectives:**
- Extract Korean and English synonym pairs from Wikipedia dumps
- Build comprehensive bilingual synonym dataset
- Pre-train Neural Sparse encoder for cross-lingual retrieval
- Export model for OpenSearch integration

---

## 🎯 Implementation Plan

### Phase 1: Wikipedia Data Collection

#### Task 1.1: Download Wikipedia Dumps
- [ ] Download Korean Wikipedia dump (kowiki)
- [ ] Download English Wikipedia dump (enwiki) or use subset
- [ ] Store dumps in `dataset/wikipedia/` directory
- [ ] Verify data integrity

**Data Source:**
- Korean: https://dumps.wikimedia.org/kowiki/latest/
- English: https://dumps.wikimedia.org/enwiki/latest/

**Target Files:**
- `kowiki-latest-pages-articles.xml.bz2`
- `enwiki-latest-pages-articles.xml.bz2` (or subset)

#### Task 1.2: Extract Clean Text
- [ ] Parse Wikipedia XML dumps
- [ ] Extract article text (remove markup, templates)
- [ ] Clean and normalize text
- [ ] Save as JSONL format

**Output:**
- `dataset/wikipedia/ko_articles.jsonl`
- `dataset/wikipedia/en_articles.jsonl`

---

### Phase 2: Synonym Data Extraction

#### Task 2.1: Extract Inter-language Links
- [ ] Parse Wikipedia inter-language links
- [ ] Extract Korean ↔ English article mappings
- [ ] Build title-based synonym pairs
- [ ] Filter and validate pairs

**Example:**
```json
{
  "korean": "인공지능",
  "english": "Artificial Intelligence",
  "source": "wikipedia_interlang"
}
```

#### Task 2.2: Extract Entity Synonyms
- [ ] Extract named entities from articles
- [ ] Use language-specific tokenizers
- [ ] Match entities across Korean/English articles
- [ ] Build entity synonym dictionary

#### Task 2.3: Build Comprehensive Synonym Dataset
- [ ] Combine inter-language links and entity synonyms
- [ ] Add existing enhanced_synonyms.json
- [ ] Remove duplicates and validate quality
- [ ] Create train/val/test splits

**Target Size:** 10K+ synonym pairs

**Output:**
- `dataset/synonyms/wiki_synonyms.json`
- `dataset/synonyms/entity_synonyms.json`
- `dataset/synonyms/combined_synonyms.json`

---

### Phase 3: Neural Sparse Model Pre-training

#### Task 3.1: Prepare Training Data
- [ ] Load Wikipedia articles and synonyms
- [ ] Generate query-document pairs from articles
- [ ] Create hard negative samples (BM25-based)
- [ ] Prepare cross-lingual training pairs

**Data Structure:**
```python
{
    "query": "인공지능 모델",
    "positive_doc": "...",
    "negative_docs": [...],
    "synonym_pairs": [("모델", "model"), ("인공지능", "AI")]
}
```

#### Task 3.2: Implement Neural Sparse Encoder
- [ ] Create `src/models/neural_sparse_encoder.py`
- [ ] Base model: `klue/bert-base` or `xlm-roberta-base`
- [ ] Implement sparse projection layer
- [ ] Add FLOPS regularization

#### Task 3.3: Implement Training Pipeline
- [ ] Create loss functions (ranking, cross-lingual, sparsity)
- [ ] Implement data collator and dataset
- [ ] Build trainer with mixed precision
- [ ] Setup logging and checkpointing

#### Task 3.4: Run Pre-training
- [ ] Start with small-scale experiment (10K pairs)
- [ ] Validate loss convergence
- [ ] Run full pre-training on Wikipedia data
- [ ] Monitor metrics (loss, sparsity, synonym alignment)

**Training Config:**
```yaml
model:
  base: "klue/bert-base"
  vocab_size: 30000

training:
  epochs: 10
  batch_size: 32
  learning_rate: 2e-5

  # Loss weights
  alpha_ranking: 1.0
  beta_cross_lingual: 0.3
  gamma_sparsity: 0.001
```

---

### Phase 4: Evaluation & Export

#### Task 4.1: Evaluate Model
- [ ] Test on held-out synonym pairs
- [ ] Measure cross-lingual retrieval accuracy
- [ ] Analyze activated terms for Korean/English queries
- [ ] Compare with baseline (BM25)

#### Task 4.2: Export Model
- [ ] Export to ONNX format
- [ ] Export to TorchScript for OpenSearch
- [ ] Validate exported model
- [ ] Create model card

#### Task 4.3: OpenSearch Integration (Optional)
- [ ] Upload model to OpenSearch
- [ ] Create neural sparse pipeline
- [ ] Test end-to-end search
- [ ] Benchmark performance

---

## 📁 Directory Structure

```
opensearch-neural-pre-train/
├── dataset/
│   ├── wikipedia/           # NEW: Wikipedia dumps and processed data
│   │   ├── dumps/
│   │   ├── ko_articles.jsonl
│   │   └── en_articles.jsonl
│   │
│   ├── synonyms/            # NEW: Synonym datasets
│   │   ├── wiki_synonyms.json
│   │   ├── entity_synonyms.json
│   │   └── combined_synonyms.json
│   │
│   └── training/            # NEW: Pre-training data
│       ├── qd_pairs.pkl
│       └── negative_samples.pkl
│
├── src/
│   ├── data/
│   │   ├── wikipedia_parser.py      # NEW
│   │   ├── synonym_extractor.py     # NEW
│   │   └── training_data_builder.py # NEW
│   │
│   ├── models/
│   │   ├── neural_sparse_encoder.py # NEW
│   │   └── model_config.py          # NEW
│   │
│   └── training/
│       ├── trainer.py               # NEW
│       ├── losses.py                # NEW
│       └── data_collator.py         # NEW
│
├── notebooks/
│   ├── 01_wikipedia_data_extraction.ipynb  # NEW
│   ├── 02_synonym_extraction.ipynb         # NEW
│   ├── 03_model_pretraining.ipynb          # NEW
│   └── 04_evaluation.ipynb                 # NEW
│
├── config/
│   └── training_config.yaml
│
└── plan.md
```

---

## 🗓️ Timeline

### Week 1: Data Collection
- [ ] Task 1.1: Download Wikipedia dumps
- [ ] Task 1.2: Extract and clean text
- [ ] Task 2.1: Extract inter-language links

### Week 2: Synonym Extraction
- [ ] Task 2.2: Extract entity synonyms
- [ ] Task 2.3: Build comprehensive synonym dataset
- [ ] Data quality validation

### Week 3: Model Implementation
- [ ] Task 3.1: Prepare training data
- [ ] Task 3.2: Implement Neural Sparse Encoder
- [ ] Task 3.3: Implement training pipeline

### Week 4: Pre-training
- [ ] Task 3.4: Small-scale experiment
- [ ] Task 3.4: Full pre-training run
- [ ] Monitor and tune

### Week 5: Evaluation & Export
- [ ] Task 4.1: Evaluate model
- [ ] Task 4.2: Export model
- [ ] Task 4.3: OpenSearch integration (optional)
- [ ] Documentation

---

## 🎯 Success Criteria

1. **Data Quality**
   - 10K+ high-quality Korean-English synonym pairs
   - Clean Wikipedia articles for pre-training

2. **Model Performance**
   - Cross-lingual synonym retrieval accuracy > 70%
   - Sparsity: average active terms < 100
   - Better than BM25 baseline on bilingual queries

3. **Deployment Ready**
   - Model exported for OpenSearch
   - Documentation complete
   - Reproducible training pipeline

---

## 📊 Current Status

**Phase:** Planning
**Next Action:** Start Phase 1 - Download Wikipedia dumps
**Last Updated:** 2025-11-17

---

## 📝 Notes

### Wikipedia Dump Processing
- Use `mwparserfromhell` or `wikiextractor` for parsing
- Filter by article quality (length, links, etc.)
- Consider using pre-processed dumps (Hugging Face datasets)

### Synonym Quality
- Validate with existing bilingual dictionaries
- Manual review of sample pairs
- Use confidence scores for filtering

### Model Architecture
- Consider starting with `xlm-roberta-base` for better multilingual support
- Experiment with different sparsity regularization strengths
- Monitor cross-lingual alignment during training
