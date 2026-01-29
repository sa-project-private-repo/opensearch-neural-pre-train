# KeyError Fix Summary - 02_training_opensearch_neural_v2.ipynb

## Problem Description

Training failed with a `KeyError: 'queries'` in Cell 29, line 32 of the training notebook.

### Error Details
```python
KeyError: 'queries'

Cell In[15], line 32, in train_step(batch, model, loss_fn, optimizer, scaler, teacher)
     30 teacher_scores = None
     31 if teacher is not None:
---> 32     queries = batch['queries']  # Assuming raw text is passed
     33     positive_docs = batch['positive_docs']
     34     negative_docs = batch['negative_docs']
```

## Root Cause Analysis

### Batch Interface Mismatch

1. **Data Collator Output** (`NeuralSparseDataCollator`):
   - Returns ONLY tokenized tensors
   - Keys: `query_input_ids`, `query_attention_mask`, `pos_doc_input_ids`, `pos_doc_attention_mask`, `neg_doc_input_ids`, `neg_doc_attention_mask`
   - Does NOT include raw text strings

2. **train_step Function Expectation**:
   - Expects raw text for teacher model (knowledge distillation)
   - Tries to access: `batch['queries']`, `batch['positive_docs']`, `batch['negative_docs']`
   - These keys don't exist in the batch

3. **Teacher Model Requirement**:
   - The `EnsembleTeacher` class needs raw text strings (not tokenized inputs)
   - Teacher models (dense/sparse) encode raw text internally
   - Cannot use pre-tokenized inputs

## Solution Implemented

### Modified: `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/src/training/data_collator.py`

Added raw text pass-through in `NeuralSparseDataCollator.__call__()`:

```python
# Store raw text for teacher model (knowledge distillation)
batch["queries"] = queries
batch["positive_docs"] = pos_docs
batch["negative_docs"] = [f["negative_docs"] for f in features]
```

### Result

The batch now contains BOTH:
1. **Tokenized inputs** (for student model):
   - `query_input_ids`, `query_attention_mask`
   - `pos_doc_input_ids`, `pos_doc_attention_mask`
   - `neg_doc_input_ids`, `neg_doc_attention_mask`

2. **Raw text** (for teacher model):
   - `queries`: List[str]
   - `positive_docs`: List[str]
   - `negative_docs`: List[List[str]]

## Validation

### Test Results

```bash
python -c "from src.training.data_collator import NeuralSparseDataCollator; ..."
```

Output:
```
Batch keys: ['queries', 'positive_docs', 'negative_docs', 'query_input_ids',
             'query_attention_mask', 'pos_doc_input_ids', 'pos_doc_attention_mask',
             'neg_doc_input_ids', 'neg_doc_attention_mask']

Tokenized inputs:
  query_input_ids: torch.Size([2, 7])
  pos_doc_input_ids: torch.Size([2, 10])
  neg_doc_input_ids: torch.Size([2, 3, 7])

Raw text for teacher:
  queries: <class 'list'> (length=2)
  positive_docs: <class 'list'> (length=2)
  negative_docs: <class 'list'> (length=2)

SUCCESS: Data collator now includes both tokenized inputs and raw text!
```

### Added Validation Cell

Inserted a new validation cell in the notebook (after Cell 25) that:
1. Displays all batch keys and their types/shapes
2. Validates student model keys (tokenized inputs)
3. Validates teacher model keys (raw text)
4. Checks shape consistency
5. Reports success/failure status

## Files Modified

1. **Source Code**:
   - `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/src/training/data_collator.py`
     - Modified `NeuralSparseDataCollator.__call__()`
     - Added raw text pass-through (lines 59-62)

2. **Notebook**:
   - `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/notebooks/opensearch-neural-v2/02_training_opensearch_neural_v2.ipynb`
     - Added batch validation cell (after Cell 25)
     - No changes needed to `train_step()` function

## Next Steps

1. **Run the notebook from the beginning**:
   - Execute cells 1-25 to initialize everything
   - Run the new validation cell to confirm batch structure
   - Proceed with training (Cell 30)

2. **Monitor training**:
   - Check that teacher model scores are computed correctly
   - Verify knowledge distillation loss is reasonable
   - Monitor FLOPS regularization values

3. **Alternative: Disable teacher temporarily** (if issues persist):
   ```python
   CONFIG['knowledge_distillation']['enabled'] = False
   ```
   This will skip teacher model creation and train with standard cross-entropy loss only.

## Technical Details

### Why This Fix Works

1. **No Memory Overhead**: Raw text strings are negligible compared to tokenized tensors
2. **Clean Separation**: Student model uses tokenized inputs, teacher uses raw text
3. **Backward Compatible**: Existing code without teacher still works
4. **Type Safe**: Each component gets data in its expected format

### Alternative Approaches Considered

1. **Option B**: Store raw text outside batch
   - Rejected: More complex, harder to maintain

2. **Option C**: Disable teacher model
   - Temporary workaround only, loses knowledge distillation benefit

3. **Option D**: Detokenize in train_step
   - Rejected: Lossy, inefficient, may not recover original text

## References

- Training notebook: `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/notebooks/opensearch-neural-v2/02_training_opensearch_neural_v2.ipynb`
- Data collator: `/home/west/Documents/cursor-workspace/opensearch-neural-pre-train/src/training/data_collator.py`
- Model paper: "Towards Competitive Search Relevance For Inference-Free Learned Sparse Retrievers" (arXiv:2411.04403v2)
