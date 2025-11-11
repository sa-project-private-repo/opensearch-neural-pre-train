#!/usr/bin/env python3
"""
Fix vocab_size usage in compute_query_representation function
"""

import json

# Read notebook
with open('korean_neural_sparse_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find compute_query_representation function
fixed = False
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        if 'def compute_query_representation' in source:
            # Replace tokenizer.vocab_size with len(tokenizer)
            if 'vocab_size = tokenizer.vocab_size' in source:
                new_source = source.replace(
                    'vocab_size = tokenizer.vocab_size',
                    'vocab_size = len(tokenizer)  # Use len() to include added special tokens'
                )
                cell['source'] = [new_source]
                fixed = True
                print(f"✓ Fixed compute_query_representation at cell {i}")
                print(f"  Changed: vocab_size = tokenizer.vocab_size")
                print(f"  To:      vocab_size = len(tokenizer)")
                break

if not fixed:
    print("Error: Could not find compute_query_representation function")
    exit(1)

# Save notebook
nb['cells'] = cells
with open('korean_neural_sparse_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✓ Notebook updated successfully!")
print(f"\nExplanation:")
print(f"  • tokenizer.vocab_size returns original BERT vocab size (32000)")
print(f"  • len(tokenizer) returns current vocab size including added tokens (32033)")
print(f"  • This ensures query_sparse and doc_sparse have matching dimensions")
