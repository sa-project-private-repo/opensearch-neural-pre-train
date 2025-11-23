# OpenSearch Neural Sparse Model Loading Guide

## Overview

This guide explains how to load OpenSearch neural sparse models and handle various loading scenarios.

## Problem Background

OpenSearch neural sparse models on HuggingFace Hub do not include a `neural_sparse_head.pt` file. This file contains the projection layer weights that map transformer hidden states to vocabulary-sized sparse representations.

The official OpenSearch models only provide:
- Base transformer weights (BERT/XLM-RoBERTa)
- Model configuration
- Tokenizer

## Solution

The `NeuralSparseEncoder.from_pretrained()` method now handles three scenarios:

### 1. Local Model with Pretrained Head

If you have a locally trained model with `neural_sparse_head.pt`:

```python
from src.models.neural_sparse_encoder import NeuralSparseEncoder

model = NeuralSparseEncoder.from_pretrained("/path/to/local/model")
```

This will load both the base encoder and the pretrained projection layer.

### 2. HuggingFace Hub Model (with fallback)

When loading from HuggingFace Hub, the method will:
1. Try to download `neural_sparse_head.pt`
2. If not found, fall back to initializing from base encoder
3. Initialize projection layer with random weights

```python
# This will load the base encoder and randomly initialize the projection
model = NeuralSparseEncoder.from_pretrained(
    "opensearch-project/opensearch-neural-sparse-encoding-v1"
)

# Warning: The projection layer is randomly initialized!
# You must train the model before using it for retrieval.
```

### 3. Initialize New Model from Base Encoder

For training a new model from scratch:

```python
model = NeuralSparseEncoder(
    model_name="xlm-roberta-base",  # or any HuggingFace model
    max_length=256,
    use_relu=True,
)
```

## Training Workflow

### Option A: Train from Scratch

If you don't have pretrained sparse weights, you must train the model:

```python
# 1. Initialize from base encoder
model = NeuralSparseEncoder.from_pretrained(
    "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
)

# 2. Train with your dataset
# ... training code ...

# 3. Save the trained model
model.save_pretrained("outputs/my_trained_model")

# 4. Load for inference
trained_model = NeuralSparseEncoder.from_pretrained("outputs/my_trained_model")
```

### Option B: Use Pretrained Weights

If you have access to pretrained `neural_sparse_head.pt` weights:

1. Download or obtain the weights file
2. Place it in your model directory with the required structure:

```
my_model/
├── config.json
├── pytorch_model.bin (or model.safetensors)
├── tokenizer_config.json
├── vocab.txt
└── neural_sparse_head.pt  # Your pretrained projection weights
```

3. Load the model:

```python
model = NeuralSparseEncoder.from_pretrained("my_model")
```

## For Notebook Training (02_training_opensearch_neural_v2.ipynb)

The notebook is designed for training, not using pretrained sparse weights. The workflow is:

### 1. Initialize Student Model (Random Projection)

```python
# Initialize model with random projection layer
model = NeuralSparseEncoder(
    model_name=CONFIG['model']['base_model'],
    max_length=CONFIG['model']['max_doc_length'],
    use_relu=CONFIG['model']['use_relu'],
)
```

### 2. Load Teacher Models

For knowledge distillation, load teacher models separately:

```python
# Dense teacher (sentence-transformers)
from sentence_transformers import SentenceTransformer
dense_teacher = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5")

# Sparse teacher - Option 1: Load base encoder if no pretrained head
sparse_teacher = NeuralSparseEncoder.from_pretrained(
    "opensearch-project/opensearch-neural-sparse-encoding-v1"
)

# Sparse teacher - Option 2: Use different architecture
# If opensearch-v1 doesn't have pretrained head, you might want to:
# - Train a sparse teacher first
# - Use a different sparse retrieval model
# - Skip sparse teacher and use only dense teacher
```

### 3. Train Student Model

The student model's projection layer will be learned during training through knowledge distillation and ranking losses.

## Model Architecture Notes

### OpenSearch Neural Sparse Encoder Structure

```
Input Text → Tokenizer → Transformer (BERT/XLM-R)
    → Hidden States [batch, seq_len, hidden_size]
    → Projection Layer [hidden_size, vocab_size]  # This layer is learned
    → ReLU Activation
    → Max Pooling over sequence
    → Sparse Representation [batch, vocab_size]
```

The projection layer is the only component that needs to be trained for neural sparse encoding. The base transformer can be frozen or fine-tuned during training.

## Common Issues and Solutions

### Issue 1: FileNotFoundError for neural_sparse_head.pt

**Cause**: The HuggingFace Hub model doesn't have a pretrained projection layer.

**Solution**: The updated code automatically falls back to random initialization. You must train the model before using it.

### Issue 2: Poor Retrieval Performance with Random Initialization

**Cause**: Random projection weights don't capture meaningful term importance.

**Solution**: Train the model with:
- Knowledge distillation from teacher models
- Ranking loss on query-document pairs
- IDF-aware FLOPS regularization
- Hard negative mining

### Issue 3: Teacher Model Loading Fails

**Cause**: Same issue - teacher model may not have pretrained sparse head.

**Solution**: Choose one of these approaches:

1. **Use only dense teacher**: Skip sparse teacher, use dense model only
2. **Train sparse teacher first**: Train a sparse model on large corpus first
3. **Use alternative sparse model**: Use SPLADE or other available sparse models
4. **Train without distillation**: Use only ranking loss without teacher

## Updated Notebook Configuration

For the training notebook, use this configuration for teacher models:

```python
CONFIG = {
    # ... other configs ...

    "knowledge_distillation": {
        "enabled": True,
        "dense_teacher": "Alibaba-NLP/gte-large-en-v1.5",  # Has pretrained weights
        "sparse_teacher": None,  # Skip if no pretrained sparse weights available
        "teacher_weights": {
            "dense": 1.0,  # Use only dense teacher
            "sparse": 0.0,
        },
    },
}
```

Or train sparse teacher first:

```python
# Phase 1: Train sparse teacher
sparse_teacher = NeuralSparseEncoder(model_name="xlm-roberta-base")
# ... train sparse_teacher ...
sparse_teacher.save_pretrained("outputs/sparse_teacher")

# Phase 2: Use trained sparse teacher
CONFIG["knowledge_distillation"]["sparse_teacher"] = "outputs/sparse_teacher"
```

## Testing Model Loading

Use the provided test script:

```bash
python scripts/test_model_loading.py
```

This will test:
1. Loading from HuggingFace Hub
2. Fallback to base encoder
3. Inference with loaded model
4. Save and load cycle

## References

- [OpenSearch Neural Sparse Encoding v1](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-v1)
- [OpenSearch Neural Sparse Encoding Multilingual v1](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1)
- [SPLADE Paper](https://arxiv.org/abs/2109.10086)
- [Learned Sparse Retrievers Paper](https://arxiv.org/abs/2411.04403v2)
