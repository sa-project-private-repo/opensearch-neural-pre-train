# Korean Neural Sparse Encoder V28

Context-Aware Sparse Vector Model for Korean Language Search

## Model Information

| Property | Value |
|----------|-------|
| Base Model | `xlm-roberta-base` |
| Architecture | SPLADEDocContextGated |
| Parameters | 345M (278M base + 67M context gate) |
| Vocabulary | 250,002 tokens |
| Max Length | 192 tokens |
| Training Epochs | 25 |

## Key Features

### V28 Enhancements over V26

1. **Korean Token Filtering**: Non-Korean token penalty (100.0) to suppress multilingual noise
2. **Context-Aware Sparse Vectors**: Context gate modulates token activations based on document context
3. **Korean Ratio**: 73.20 (vs ~0.03 in V26) - Korean tokens now dominate sparse representations

### Context-Gated Architecture

```
Input -> XLM-R -> MLM Logits
              |
              +-> Context Gate -> Gate [batch, vocab]
                       |
Gated Logits = MLM Logits * Gate.unsqueeze(1)
                       |
Sparse = max_pool(ReLU(log1p(Gated Logits)))
```

The context gate enables different token activations for the same keyword based on surrounding context:
- "출근했는데 점심 메뉴" → 회사, 직장인, 비빔밥
- "학교를 갔는데 점심 메뉴" → 학생, 급식, 도시락

## Training Results

| Metric | Value |
|--------|-------|
| Final Train Loss | 11.8504 |
| Best Val Loss | 7.8480 |
| Korean Ratio | **73.20** |
| Language Penalty | 0.0 |

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load model
tokenizer = AutoTokenizer.from_pretrained("sewoong/korean-neural-sparse-encoder-v28")
model = AutoModelForMaskedLM.from_pretrained("sewoong/korean-neural-sparse-encoder-v28")

def encode(text: str, top_k: int = 50):
    """Encode text to sparse vector."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=192)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # SPLADE transformation: log(1 + ReLU(logits))
    sparse_scores = torch.log1p(torch.relu(logits))

    # Max pooling over sequence
    sparse_repr = sparse_scores.max(dim=1).values.squeeze()

    # Get top-k tokens
    top_values, top_indices = sparse_repr.topk(top_k)

    # Convert to dict
    sparse_dict = {}
    for idx, val in zip(top_indices.tolist(), top_values.tolist()):
        if val > 0:
            token = tokenizer.decode([idx]).strip()
            if token:
                sparse_dict[token] = round(val, 4)

    return sparse_dict

# Example
result = encode("당뇨병 치료 방법")
print(result)
# {'당뇨병': 2.45, '치료': 1.82, '방법': 1.21, '치료법': 0.95, ...}
```

## OpenSearch Integration

```json
{
    "mappings": {
        "properties": {
            "content": {"type": "text"},
            "sparse_embedding": {"type": "rank_features"}
        }
    }
}
```

## Version History

| Version | Description |
|---------|-------------|
| V28 | Context-aware sparse vectors + Korean token filtering |
| V26 | IDF-aware FLOPS + Special token handling |

## Citation

```bibtex
@misc{korean-neural-sparse-v28,
    title={Korean Neural Sparse Encoder V28: Context-Aware Sparse Vectors},
    author={sewoong},
    year={2026},
    publisher={HuggingFace}
}
```

## License

Apache 2.0
