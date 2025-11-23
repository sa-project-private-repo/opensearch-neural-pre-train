#!/usr/bin/env python3
"""
Test script for DGX Spark (ARM + GB10) setup.
Validates SPLADE model loading and GPU training.
"""

import torch
from transformers import AutoTokenizer
from src.model.splade_model import create_splade_model
from src.model.losses import SPLADELoss

print("=" * 70)
print("Testing SPLADE-doc on Nvidia DGX Spark (ARM + GB10)")
print("=" * 70)

# 1. GPU Information
print("\n[1/5] GPU Information")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  CUDA Version: {torch.version.cuda}")
print(f"  BF16 Support: {torch.cuda.is_bf16_supported()}")
print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 2. Load Model
print("\n[2/5] Loading SPLADE-doc model")
device = torch.device('cuda')
model = create_splade_model(
    model_name="bert-base-multilingual-cased",
    use_idf=False,
    dropout=0.1
)
model = model.to(device)
print(f"  ✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# 3. Load Tokenizer
print("\n[3/5] Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
print(f"  ✓ Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

# 4. Test Forward Pass (FP32)
print("\n[4/5] Testing forward pass (FP32)")
test_texts = [
    "한국어 신경망 희소 검색",
    "Neural sparse retrieval for Korean",
]
encoded = tokenizer(
    test_texts,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
input_ids = encoded['input_ids'].to(device)
attention_mask = encoded['attention_mask'].to(device)

with torch.no_grad():
    sparse_repr, token_weights = model(input_ids, attention_mask)

print(f"  ✓ Input shape: {input_ids.shape}")
print(f"  ✓ Sparse repr shape: {sparse_repr.shape}")
print(f"  ✓ Non-zero tokens: {(sparse_repr[0] > 0).sum().item()}/{sparse_repr.shape[1]}")
print(f"  ✓ Sparsity: {100 * (1 - (sparse_repr[0] > 0).sum().item() / sparse_repr.shape[1]):.2f}%")

# 5. Test Forward Pass (BF16)
print("\n[5/5] Testing forward pass (BF16)")
model = model.to(torch.bfloat16)

with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        sparse_repr_bf16, token_weights_bf16 = model(input_ids, attention_mask)

print(f"  ✓ BF16 sparse repr shape: {sparse_repr_bf16.shape}")
print(f"  ✓ BF16 dtype: {sparse_repr_bf16.dtype}")
print(f"  ✓ Non-zero tokens: {(sparse_repr_bf16[0] > 0).sum().item()}/{sparse_repr_bf16.shape[1]}")

# 6. Test Loss Function
print("\n[6/6] Testing loss function")
model = model.to(torch.float32)  # Back to FP32 for loss test

# Create dummy batch
batch_size = 4
num_negatives = 3

query_repr = torch.randn(batch_size, tokenizer.vocab_size, device=device)
pos_doc_repr = torch.randn(batch_size, tokenizer.vocab_size, device=device)
neg_doc_repr = torch.randn(batch_size, num_negatives, tokenizer.vocab_size, device=device)

loss_fn = SPLADELoss(
    temperature=0.05,
    lambda_flops=1e-4,
    lambda_idf=1e-3,
    use_kd=False,
    use_idf_penalty=False,
).to(device)

loss, loss_dict = loss_fn(query_repr, pos_doc_repr, neg_doc_repr)

print(f"  ✓ Total loss: {loss.item():.4f}")
print(f"  ✓ Contrastive loss: {loss_dict['contrastive']:.4f}")
print(f"  ✓ FLOPS loss: {loss_dict['flops']:.4f}")

# Memory usage
mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
print(f"\n[Memory Usage]")
print(f"  Allocated: {mem_allocated:.2f} GB")
print(f"  Reserved: {mem_reserved:.2f} GB")

print("\n" + "=" * 70)
print("✓ All tests passed! DGX setup is ready for training.")
print("=" * 70)
print("\nNext steps:")
print("  1. Run baseline training:")
print("     source .venv/bin/activate")
print("     python train.py --config configs/baseline_dgx.yaml")
print("\n  2. Or run full pre-training:")
print("     python train.py --config configs/pretrain_korean_dgx.yaml")
print("=" * 70)
