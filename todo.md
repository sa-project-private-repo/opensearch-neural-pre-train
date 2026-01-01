# v21.4 Korean Neural Sparse Encoder Development Plan

## Overview

Combine the strengths of v21.2 (single-term performance) and v21.3 (data quality, natural language) to create an improved model.

### Target Metrics
| Metric | v21.2 | v21.3 | v21.4 Target |
|--------|-------|-------|--------------|
| Single-term Recall | 77.4% | 63.1% | **80%+** |
| Natural Language | Good | Good | **Good** |
| Garbage Ratio | Low | High | **< 5%** |
| Sparsity | 96.36% | 95.55% | **95%+** |

---

## Phase 1: Data Engineering (data-engineering-professional)

### 1.1 Analyze Current Data Issues
- [ ] Identify terms with garbage outputs in v21.3
- [ ] Check if these terms exist in filtered_synonym_pairs.jsonl
- [ ] Analyze filtering statistics per term category
- [ ] Generate report on data coverage gaps

### 1.2 Single-term Data Augmentation
- [ ] Create identity pairs for problem terms (term → term)
- [ ] Extract single-term synonym pairs from existing data
- [ ] Add explicit synonym pairs for: 추천, 데이터베이스, 증상, 질환, 인슐린
- [ ] Balance single-term vs sentence-level pairs (target: 30% single-term)

### 1.3 Domain-specific Filtering Adjustment
- [ ] Implement per-domain filtering thresholds
  - Legal: 2/3 filters (strict)
  - Medical: 1/3 filters (relaxed for rare terms)
  - General: 2/3 filters (strict)
- [ ] Re-run filtering pipeline with adjusted thresholds
- [ ] Validate medical terminology coverage

### 1.4 Data Quality Validation
- [ ] Create validation set for single-term evaluation
- [ ] Create validation set for sentence-level evaluation
- [ ] Ensure no data leakage between train/validation

**Output Files:**
- `data/v21.4/single_term_pairs.jsonl`
- `data/v21.4/augmented_synonym_pairs.jsonl`
- `data/v21.4/domain_filtered_pairs.jsonl`

---

## Phase 2: ML Architecture Design (ml-architecture-advisor)

### 2.1 Loss Function Redesign
- [ ] Design dynamic lambda_self based on input length
  ```python
  lambda_self = 8.0 if len(tokens) <= 3 else 4.0
  ```
- [ ] Design minimum activation loss
  ```python
  L_min_act = -log(mean(top_k_activations))
  ```
- [ ] Adjust FLOPS regularization (8e-3 → 5e-3)
- [ ] Document loss function changes with mathematical formulation

### 2.2 Curriculum Learning Strategy
- [ ] Design 3-phase training curriculum
  - Phase 1 (Epoch 1-10): Single-term focus, lambda_self=8.0
  - Phase 2 (Epoch 11-20): Mixed data, lambda_self=6.0
  - Phase 3 (Epoch 21-30): Full data, lambda_self=4.0
- [ ] Design batch composition strategy
  - 30% single-term
  - 30% short phrases (2-5 tokens)
  - 40% sentences (6+ tokens)

### 2.3 Evaluation Framework Design
- [ ] Design single-term specific evaluation metrics
- [ ] Design garbage detection metric
- [ ] Design composite score combining all metrics

**Output:**
- `docs/v21.4_architecture_design.md`

---

## Phase 3: Implementation (python-expert)

### 3.1 Data Preparation Module
- [ ] Implement `src/data/single_term_augmentor.py`
  - SingleTermAugmentor class
  - Identity pair generation
  - Domain-aware filtering
- [ ] Implement `src/data/multi_length_sampler.py`
  - Length-balanced batch sampler
  - Curriculum-aware sampling

### 3.2 Training Module Updates
- [ ] Update `src/training/loss.py`
  - Add DynamicLambdaScheduler class
  - Add MinimumActivationLoss class
  - Update total loss computation
- [ ] Update `src/training/trainer.py`
  - Add curriculum learning support
  - Add per-phase configuration

### 3.3 Evaluation Module Updates
- [ ] Update `src/evaluation/metrics.py`
  - Add SingleTermRecall metric
  - Add GarbageRatio metric
  - Add CompositeScore metric
- [ ] Create `src/evaluation/garbage_detector.py`
  - Valid token detection
  - Unicode range validation

### 3.4 Configuration
- [ ] Create `configs/v21.4_config.yaml`
  ```yaml
  model:
    base_model: skt/A.X-Encoder-base
    max_length: 64

  training:
    epochs: 30
    batch_size: 64
    learning_rate: 3e-6

  loss:
    lambda_self_single: 8.0
    lambda_self_sentence: 4.0
    lambda_synonym: 10.0
    lambda_margin: 2.5
    lambda_flops: 5e-3
    lambda_min_activation: 1.0

  curriculum:
    phase1_epochs: 10
    phase2_epochs: 10
    phase3_epochs: 10
  ```

**Output Files:**
- `src/data/single_term_augmentor.py`
- `src/data/multi_length_sampler.py`
- `src/training/loss.py` (updated)
- `src/training/trainer.py` (updated)
- `src/evaluation/metrics.py` (updated)
- `src/evaluation/garbage_detector.py`
- `configs/v21.4_config.yaml`

---

## Phase 4: Training Pipeline (ai-ml-expert-engineer)

### 4.1 Notebook Development
- [ ] Create `notebooks/opensearch-neural-v21.4/01_data_augmentation.ipynb`
  - Load v21.3 filtered data
  - Add single-term augmentation
  - Apply domain-specific filtering
  - Save augmented dataset

- [ ] Create `notebooks/opensearch-neural-v21.4/02_data_preparation.ipynb`
  - Generate training triplets with length balance
  - Create curriculum-aware data splits
  - Validate data distribution

- [ ] Create `notebooks/opensearch-neural-v21.4/03_training.ipynb`
  - Implement curriculum learning
  - Dynamic loss scheduling
  - Checkpoint saving per phase

- [ ] Create `notebooks/opensearch-neural-v21.4/04_evaluation.ipynb`
  - Single-term evaluation
  - Sentence-level evaluation
  - Garbage ratio analysis
  - v21.2 vs v21.3 vs v21.4 comparison

### 4.2 Training Execution
- [ ] Run Phase 1 training (single-term focus)
- [ ] Evaluate Phase 1 checkpoint
- [ ] Run Phase 2 training (mixed)
- [ ] Evaluate Phase 2 checkpoint
- [ ] Run Phase 3 training (full)
- [ ] Final evaluation and comparison

### 4.3 Model Selection
- [ ] Compare all phase checkpoints
- [ ] Select best model based on composite score
- [ ] Validate on held-out test set

**Output Files:**
- `outputs/v21.4_korean_enhanced/`
  - `phase1_checkpoint.pt`
  - `phase2_checkpoint.pt`
  - `best_model.pt`
  - `training_history.json`
  - `evaluation_results.json`

---

## Phase 5: Deployment (opensearch-nlp-expert)

### 5.1 Model Export
- [ ] Convert to HuggingFace format
- [ ] Create model card with v21.4 improvements
- [ ] Upload to HuggingFace Hub

### 5.2 OpenSearch Integration Test
- [ ] Test with OpenSearch neural sparse plugin
- [ ] Benchmark retrieval performance
- [ ] Document integration guide

---

## Timeline

| Phase | Duration | Agent | Status |
|-------|----------|-------|--------|
| Phase 1: Data Engineering | 2 days | data-engineering-professional | Not Started |
| Phase 2: ML Architecture | 1 day | ml-architecture-advisor | Not Started |
| Phase 3: Implementation | 2 days | python-expert | Not Started |
| Phase 4: Training | 3 days | ai-ml-expert-engineer | Not Started |
| Phase 5: Deployment | 1 day | opensearch-nlp-expert | Not Started |

---

## Success Criteria

1. **Single-term Recall**: >= 80% (improvement from 63.1%)
2. **Garbage Ratio**: < 5% on test terms
3. **Natural Language Performance**: Maintain or improve v21.3 level
4. **Sparsity**: >= 95%

---

## Notes

- All code should follow PEP 8 and include type hints
- Each notebook should be independently executable
- Checkpoints should be saved at each phase for analysis
- Document all hyperparameter changes and their rationale
