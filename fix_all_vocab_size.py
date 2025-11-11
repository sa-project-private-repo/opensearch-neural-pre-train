#!/usr/bin/env python3
"""
Fix all vocab_size usages to use len(tokenizer) instead of tokenizer.vocab_size
"""

import json
import re

# Read notebook
with open('korean_neural_sparse_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

fixed_count = 0
cell_indices = []

for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        if 'tokenizer.vocab_size' in source:
            # Replace all occurrences
            # Pattern 1: Direct usage in code
            new_source = re.sub(
                r'tokenizer\.vocab_size',
                'len(tokenizer)',
                source
            )

            if new_source != source:
                cell['source'] = [new_source]
                fixed_count += source.count('tokenizer.vocab_size')
                cell_indices.append(i)
                print(f"✓ Fixed cell {i} ({source.count('tokenizer.vocab_size')} occurrence(s))")

print(f"\n{'='*60}")
print(f"✓ Fixed {fixed_count} occurrences in {len(cell_indices)} cells")
print(f"{'='*60}")
print("\nFixed cells:", cell_indices)

# Save notebook
nb['cells'] = cells
with open('korean_neural_sparse_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✓ Notebook updated successfully!")
print(f"\n💡 Explanation:")
print(f"  tokenizer.vocab_size = 32000 (original BERT vocab)")
print(f"  len(tokenizer)       = 32033 (after adding 33 special tokens)")
print(f"\nThis ensures all tensors use the correct vocabulary size.")
