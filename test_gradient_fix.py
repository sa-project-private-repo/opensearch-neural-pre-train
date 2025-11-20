#!/usr/bin/env python3
"""
Test script to verify gradient computation works without inplace errors.
"""

import torch
from transformers import AutoTokenizer
from src.model.splade_model import create_splade_model
from src.model.losses import SPLADELoss

print("=" * 70)
print("Testing Gradient Computation (Inplace Operation Fix)")
print("=" * 70)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Load model and tokenizer
print("\n[1/4] Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = create_splade_model(
    model_name="bert-base-multilingual-cased",
    use_idf=False,
    dropout=0.1,
)
model = model.to(device)
print("✓ Model loaded")

# Create loss function
print("\n[2/4] Initializing loss function...")
loss_fn = SPLADELoss(
    temperature=0.05,
    lambda_flops=1e-4,
    lambda_idf=1e-3,
    use_kd=False,
    use_idf_penalty=False,
)
loss_fn = loss_fn.to(device)
print("✓ Loss function initialized")

# Create sample batch
print("\n[3/4] Creating sample batch...")
batch_size = 4
num_negatives = 3

texts = [
    "한국어 신경망 희소 검색 모델",
    "Neural sparse retrieval for Korean",
    "OpenSearch 검색 엔진",
    "SPLADE document encoder",
]

# Tokenize
query_encoded = tokenizer(
    texts,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
pos_encoded = tokenizer(
    texts,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
neg_encoded = tokenizer(
    texts * num_negatives,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# Move to device
query_input_ids = query_encoded['input_ids'].to(device)
query_attention_mask = query_encoded['attention_mask'].to(device)
pos_input_ids = pos_encoded['input_ids'].to(device)
pos_attention_mask = pos_encoded['attention_mask'].to(device)
neg_input_ids = neg_encoded['input_ids'].view(batch_size, num_negatives, -1).to(device)
neg_attention_mask = neg_encoded['attention_mask'].view(batch_size, num_negatives, -1).to(device)

print("✓ Sample batch created")

# Test forward and backward
print("\n[4/4] Testing forward and backward pass...")
try:
    # Forward pass
    query_repr, _ = model(query_input_ids, query_attention_mask)
    pos_doc_repr, _ = model(pos_input_ids, pos_attention_mask)

    # Negative documents
    neg_input_ids_flat = neg_input_ids.view(batch_size * num_negatives, -1)
    neg_attention_mask_flat = neg_attention_mask.view(batch_size * num_negatives, -1)
    neg_doc_repr_flat, _ = model(neg_input_ids_flat, neg_attention_mask_flat)
    neg_doc_repr = neg_doc_repr_flat.view(batch_size, num_negatives, -1)

    print("  ✓ Forward pass successful")

    # Compute loss
    loss, loss_dict = loss_fn(query_repr, pos_doc_repr, neg_doc_repr)
    print(f"  ✓ Loss computed: {loss.item():.4f}")
    print(f"    - Contrastive: {loss_dict['contrastive']:.4f}")
    print(f"    - FLOPS: {loss_dict['flops']:.4f}")

    # Backward pass (THIS IS THE CRITICAL TEST)
    loss.backward()
    print("  ✓ Backward pass successful (NO INPLACE ERROR!)")

    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            break

    if has_grad:
        print("  ✓ Gradients computed successfully")
    else:
        print("  ⚠ Warning: No gradients found")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nThe inplace operation error has been fixed.")
    print("You can now run notebook 06 without errors.")
    print("\nNext steps:")
    print("  1. Restart Jupyter kernel")
    print("  2. Run notebook 06 from the beginning")
    print("=" * 70)

except RuntimeError as e:
    if "inplace operation" in str(e):
        print("\n" + "=" * 70)
        print("❌ INPLACE OPERATION ERROR STILL EXISTS")
        print("=" * 70)
        print(f"\nError: {e}")
        print("\nThe error persists. Further debugging needed.")
    else:
        print(f"\n❌ Different error occurred: {e}")
        raise
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    raise
