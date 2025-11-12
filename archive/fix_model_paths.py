#!/usr/bin/env python3
"""
Update model output paths to use models/ directory in all notebooks.
"""

import json
import sys
from pathlib import Path

def fix_model_paths_in_notebook(notebook_path):
    """Replace model output paths with models/ prefix"""

    print(f"\nProcessing: {notebook_path}")

    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    changes = 0

    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                # Replace model output paths
                if 'OUTPUT_DIR = "./opensearch-korean-neural-sparse' in line:
                    original = line
                    line = line.replace(
                        'OUTPUT_DIR = "./opensearch-korean-neural-sparse',
                        'OUTPUT_DIR = "./models/opensearch-korean-neural-sparse'
                    )
                    if line != original:
                        changes += 1
                        print(f"  ✓ Updated OUTPUT_DIR")

                elif 'MODEL_DIR = "./opensearch-korean-neural-sparse' in line:
                    original = line
                    line = line.replace(
                        'MODEL_DIR = "./opensearch-korean-neural-sparse',
                        'MODEL_DIR = "./models/opensearch-korean-neural-sparse'
                    )
                    if line != original:
                        changes += 1
                        print(f"  ✓ Updated MODEL_DIR")

                elif 'from_pretrained("./opensearch-korean-neural-sparse' in line:
                    original = line
                    line = line.replace(
                        'from_pretrained("./opensearch-korean-neural-sparse',
                        'from_pretrained("./models/opensearch-korean-neural-sparse'
                    )
                    if line != original:
                        changes += 1
                        print(f"  ✓ Updated from_pretrained path")

                # Also update cd commands in bash cells
                elif line.strip().startswith('cd opensearch-korean-neural-sparse'):
                    original = line
                    line = line.replace(
                        'cd opensearch-korean-neural-sparse',
                        'cd models/opensearch-korean-neural-sparse'
                    )
                    if line != original:
                        changes += 1
                        print(f"  ✓ Updated cd path")

                new_source.append(line)

            cell['source'] = new_source

    if changes > 0:
        # Save updated notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, ensure_ascii=False, indent=1)
        print(f"  ✓ Saved with {changes} changes")
    else:
        print(f"  ⚠ No changes needed")

    return changes

if __name__ == "__main__":
    print("="*70)
    print("Fixing Model Paths to use models/ Directory")
    print("="*70)

    notebooks = [
        'notebooks/korean_neural_sparse_training.ipynb',
        'notebooks/korean_neural_sparse_training_v0.3.0.ipynb',
        'notebooks/neural_sparse_inference.ipynb',
    ]

    total_changes = 0

    for nb_path in notebooks:
        try:
            changes = fix_model_paths_in_notebook(nb_path)
            total_changes += changes
        except FileNotFoundError:
            print(f"  ⚠ Skipped (not found): {nb_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "="*70)
    print(f"✓ Complete! Total changes: {total_changes}")
    print("="*70)
