#!/usr/bin/env python3
"""
Remove or comment out ai_domain_terminology imports from notebooks.
This module has been archived and is no longer used.
"""

import json
import sys
from pathlib import Path

def fix_ai_terminology_in_notebook(notebook_path):
    """Remove or comment out cells that import ai_domain_terminology"""

    print(f"\nProcessing: {notebook_path}")

    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    changes = 0
    cells_to_remove = []

    for idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Check if this cell imports ai_domain_terminology
            if 'from ai_domain_terminology import' in source or 'import ai_domain_terminology' in source:
                print(f"  Found ai_domain_terminology import in cell {idx}")

                # Option 1: Comment out the cell
                new_source = []
                for line in cell['source']:
                    if line.strip() and not line.strip().startswith('#'):
                        new_source.append('# ' + line)
                    else:
                        new_source.append(line)

                cell['source'] = new_source
                changes += 1
                print(f"  ✓ Commented out cell {idx}")

            # Also check for usage of AI_TERMINOLOGY, TECHNICAL_SPECIAL_TOKENS, AI_SYNONYMS
            elif any(term in source for term in ['AI_TERMINOLOGY', 'TECHNICAL_SPECIAL_TOKENS', 'AI_SYNONYMS']):
                # Check if it's being used (not just mentioned in comments)
                has_usage = False
                for line in cell['source']:
                    if not line.strip().startswith('#') and any(term in line for term in ['AI_TERMINOLOGY', 'TECHNICAL_SPECIAL_TOKENS', 'AI_SYNONYMS']):
                        has_usage = True
                        break

                if has_usage:
                    print(f"  Found usage of AI terminology variables in cell {idx}")
                    # Comment out
                    new_source = []
                    for line in cell['source']:
                        if line.strip() and not line.strip().startswith('#'):
                            new_source.append('# ' + line)
                        else:
                            new_source.append(line)

                    cell['source'] = new_source
                    changes += 1
                    print(f"  ✓ Commented out cell {idx}")

    if changes > 0:
        # Save updated notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, ensure_ascii=False, indent=1)
        print(f"  ✓ Saved with {changes} cells commented out")
    else:
        print(f"  ⚠ No changes needed")

    return changes

if __name__ == "__main__":
    print("="*70)
    print("Fixing AI Domain Terminology Imports")
    print("="*70)
    print("\nThe ai_domain_terminology module has been archived.")
    print("Commenting out related cells in notebooks...\n")

    notebooks = [
        'notebooks/korean_neural_sparse_training.ipynb',
        'notebooks/korean_neural_sparse_training_v0.3.0.ipynb',
    ]

    total_changes = 0

    for nb_path in notebooks:
        try:
            changes = fix_ai_terminology_in_notebook(nb_path)
            total_changes += changes
        except FileNotFoundError:
            print(f"  ⚠ Skipped (not found): {nb_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "="*70)
    print(f"✓ Complete! Total cells modified: {total_changes}")
    print("="*70)

    if total_changes > 0:
        print("\nNote: Cells have been commented out, not deleted.")
        print("You can manually delete them if not needed, or uncomment if you")
        print("want to restore ai_domain_terminology.py from archive/")
