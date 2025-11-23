# Training Quick Start Guide

## Issue Resolution: KeyError 'queries'

The KeyError has been fixed! The data collator now passes both tokenized inputs and raw text.

## What Was Fixed

### Problem
```python
KeyError: 'queries'  # train_step couldn't find raw text for teacher model
```

### Solution
Modified `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/src/training/data_collator.py` to include raw text in batch:

```python
batch["queries"] = queries
batch["positive_docs"] = pos_docs
batch["negative_docs"] = [f["negative_docs"] for f in features]
```

## Current Batch Structure

Each batch now contains:

### For Student Model (Tokenized)
- `query_input_ids`: torch.Tensor [batch_size, seq_len]
- `query_attention_mask`: torch.Tensor [batch_size, seq_len]
- `pos_doc_input_ids`: torch.Tensor [batch_size, seq_len]
- `pos_doc_attention_mask`: torch.Tensor [batch_size, seq_len]
- `neg_doc_input_ids`: torch.Tensor [batch_size, num_negatives, seq_len]
- `neg_doc_attention_mask`: torch.Tensor [batch_size, num_negatives, seq_len]

### For Teacher Model (Raw Text)
- `queries`: List[str] - Raw query strings
- `positive_docs`: List[str] - Raw positive document strings
- `negative_docs`: List[List[str]] - Raw negative document strings

## How to Resume Training

### Option 1: Run Full Notebook

1. Open notebook:
   ```bash
   jupyter notebook /home/west/Documents/cursor-workspace/opensearch-neural-pre-train/notebooks/opensearch-neural-v2/02_training_opensearch_neural_v2.ipynb
   ```

2. Execute cells in order:
   - Cells 1-25: Setup, data loading, model initialization
   - New validation cell: Verify batch structure
   - Cell 30: Start training loop

3. Monitor output for:
   - Successful teacher score computation
   - Decreasing total loss
   - Reasonable FLOPS regularization values

### Option 2: Quick Test (Skip Teacher)

If you want to test training quickly without teacher model:

```python
# In Cell 5 (CONFIG), modify:
CONFIG['knowledge_distillation']['enabled'] = False
```

This will:
- Skip teacher model loading (faster startup)
- Use standard cross-entropy loss instead of KL divergence
- Still apply IDF-aware FLOPS regularization

### Option 3: Use Only Dense Teacher

If sparse teacher loading fails:

```python
# In Cell 5 (CONFIG), modify:
CONFIG['knowledge_distillation']['teacher_weights']['sparse'] = 0.0
```

This will:
- Use only the dense teacher (Alibaba GTE-large)
- Avoid potential issues with sparse teacher projection head
- Still benefit from knowledge distillation

## Validation Before Training

The new validation cell (after Cell 25) will show:

```
================================================================================
VALIDATION RESULTS
================================================================================

Student model keys (tokenized):
  ✓ query_input_ids
  ✓ query_attention_mask
  ✓ pos_doc_input_ids
  ✓ pos_doc_attention_mask
  ✓ neg_doc_input_ids
  ✓ neg_doc_attention_mask

Teacher model keys (raw text):
  ✓ queries
  ✓ positive_docs
  ✓ negative_docs

Batch size: 40
Number of negatives: 10

Shape validation:
  ✓ query_input_ids              : torch.Size([40, 64])
  ✓ pos_doc_input_ids             : torch.Size([40, 256])
  ✓ neg_doc_input_ids             : torch.Size([40, 10, 256])

================================================================================
SUCCESS: Batch structure is correct!
Ready to start training with knowledge distillation.
================================================================================
```

## Troubleshooting

### If Teacher Model Fails to Load

1. Check CUDA memory:
   ```python
   import torch
   print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
   print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   ```

2. Reduce batch size in CONFIG:
   ```python
   CONFIG['finetuning']['batch_size'] = 20  # Reduced from 40
   ```

3. Disable sparse teacher:
   ```python
   CONFIG['knowledge_distillation']['teacher_weights']['sparse'] = 0.0
   ```

### If Training is Too Slow

1. Reduce number of training steps:
   ```python
   CONFIG['finetuning']['num_steps'] = 10000  # Reduced from 50000
   ```

2. Increase gradient accumulation:
   ```python
   CONFIG['finetuning']['gradient_accumulation_steps'] = 2
   CONFIG['finetuning']['batch_size'] = 20  # Reduce accordingly
   ```

3. Use mixed precision (should be enabled by default):
   ```python
   CONFIG['hardware']['mixed_precision'] = True
   CONFIG['hardware']['precision'] = 'bf16'
   ```

### If You Get OOM (Out of Memory)

1. Reduce batch size:
   ```python
   CONFIG['finetuning']['batch_size'] = 16
   ```

2. Reduce number of negatives:
   ```python
   CONFIG['finetuning']['num_hard_negatives'] = 5
   ```

3. Reduce max lengths:
   ```python
   CONFIG['model']['max_query_length'] = 32
   CONFIG['model']['max_doc_length'] = 128
   ```

## Expected Training Behavior

### Normal Training Output

```
Training...
Step 100: Avg loss = 2.5432
Step 200: Avg loss = 2.1234
Step 300: Avg loss = 1.8765
...
Checkpoint saved to outputs/opensearch-neural-v2/checkpoint-1000
```

### Loss Components

- **Total loss**: Should decrease over time (target: < 1.0)
- **Ranking loss**: Main component, measures retrieval quality
- **FLOPS loss**: Sparsity regularization, should be small (< 0.01)

### Checkpoints

Saved every 1000 steps to:
```
outputs/opensearch-neural-v2/
├── checkpoint-1000/
├── checkpoint-2000/
├── checkpoint-3000/
└── final_model/
```

## Load Your Training Data

Replace the dummy data in Cell 23:

```python
# TODO: Load your training data
import json

# Example: Load from JSONL file
train_data = []
with open('path/to/your/training_data.jsonl', 'r') as f:
    for line in f:
        train_data.append(json.loads(line))

# Extract components
train_queries = [item['query'] for item in train_data]
train_positive_docs = [item['positive_doc'] for item in train_data]
train_negative_docs = [item['negative_docs'] for item in train_data]

print(f"Loaded {len(train_queries)} training samples")
```

### Expected Data Format

```json
{
  "query": "What is machine learning?",
  "positive_doc": "Machine learning is a subset of artificial intelligence...",
  "negative_docs": [
    "This is an irrelevant document about cooking.",
    "Another unrelated document about sports.",
    ...
  ]
}
```

## Next Steps After Training

1. **Evaluate model**:
   - Run inference tests (Cell 36)
   - Check sparsity statistics
   - Measure retrieval quality on validation set

2. **Deploy to OpenSearch**:
   - Export model to ONNX format
   - Upload to OpenSearch model registry
   - Configure neural search pipeline

3. **Fine-tune hyperparameters**:
   - Adjust `lambda_flops` for sparsity-relevance trade-off
   - Tune learning rate and warmup steps
   - Experiment with different teacher weights

## References

- Fix documentation: `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/docs/KEYERROR_FIX_SUMMARY.md`
- Training notebook: `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/notebooks/opensearch-neural-v2/02_training_opensearch_neural_v2.ipynb`
- Data collator: `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/src/training/data_collator.py`
