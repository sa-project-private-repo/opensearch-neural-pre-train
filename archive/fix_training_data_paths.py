#!/usr/bin/env python3
"""
Update all training data paths (checkpoints, logs, best models) to use models/ directory.
"""

import json
import sys
from pathlib import Path

def fix_training_paths_in_notebook(notebook_path):
    """Replace training data paths with models/ prefix"""

    print(f"\nProcessing: {notebook_path}")

    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    changes = 0

    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                original = line

                # Fix best model checkpoint path
                if 'best_model_path = "best_korean_neural_sparse_encoder.pt"' in line:
                    line = line.replace(
                        'best_model_path = "best_korean_neural_sparse_encoder.pt"',
                        'best_model_path = "./models/best_korean_neural_sparse_encoder.pt"'
                    )
                    if line != original:
                        changes += 1
                        print(f"  ✓ Updated best_model_path")

                # Fix torch.load paths for checkpoints
                elif 'torch.load("best_korean_neural_sparse_encoder.pt"' in line:
                    line = line.replace(
                        'torch.load("best_korean_neural_sparse_encoder.pt"',
                        'torch.load("./models/best_korean_neural_sparse_encoder.pt"'
                    )
                    if line != original:
                        changes += 1
                        print(f"  ✓ Updated torch.load checkpoint path")

                # Fix checkpoint directory references
                elif 'checkpoint = torch.load(best_model_path)' in line:
                    # This is fine, uses the variable
                    pass

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
    print("Fixing Training Data Paths to use models/ Directory")
    print("="*70)

    notebooks = [
        'notebooks/korean_neural_sparse_training.ipynb',
        'notebooks/korean_neural_sparse_training_v0.3.0.ipynb',
    ]

    total_changes = 0

    for nb_path in notebooks:
        try:
            changes = fix_training_paths_in_notebook(nb_path)
            total_changes += changes
        except FileNotFoundError:
            print(f"  ⚠ Skipped (not found): {nb_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "="*70)
    print(f"✓ Complete! Total changes: {total_changes}")
    print("="*70)

    print("\nUpdated paths:")
    print("  • best_korean_neural_sparse_encoder.pt → models/best_korean_neural_sparse_encoder.pt")
    print("  • All checkpoints saved to models/")
