# SPLADE V24 XLM-RoBERTa Checkpoint

## Model Info
- Base model: xlm-roberta-base
- Epoch: 10
- Step: 112340
- Timestamp: 20260118_053617

## Metrics
{
  "train_loss": -1.3336232583389516,
  "val_loss": -1.2612728974742535
}

## Usage
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("huggingface/v24_epoch10")
model = AutoModelForMaskedLM.from_pretrained("huggingface/v24_epoch10")
```
