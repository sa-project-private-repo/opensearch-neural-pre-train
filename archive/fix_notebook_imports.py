#!/usr/bin/env python3
"""
Fix notebook imports after moving to notebooks/ folder.
Adds sys.path modification to make src/ modules accessible.
"""

import json
import sys

def fix_notebook_imports(notebook_path):
    """Add sys.path setup cell at the beginning of notebook"""

    print(f"Processing: {notebook_path}")

    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Check if sys.path cell already exists
    has_sys_path = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'sys.path.append' in source and '..' in source:
                has_sys_path = True
                print("  ✓ sys.path cell already exists")
                break

    if not has_sys_path:
        # Create sys.path setup cell
        sys_path_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Add parent directory to path for src imports\n",
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "# Add project root to path\n",
                "project_root = Path().absolute().parent\n",
                "if str(project_root) not in sys.path:\n",
                "    sys.path.insert(0, str(project_root))\n",
                "\n",
                "print(f\"Project root: {project_root}\")\n"
            ]
        }

        # Insert at the beginning (after title cells if any)
        insert_pos = 0
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                insert_pos = i
                break

        notebook['cells'].insert(insert_pos, sys_path_cell)
        print(f"  ✓ Added sys.path cell at position {insert_pos}")

        # Save updated notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, ensure_ascii=False, indent=1)

        print(f"  ✓ Updated notebook: {len(notebook['cells'])} cells")

    return notebook

if __name__ == "__main__":
    notebooks = [
        'notebooks/korean_neural_sparse_training_v0.3.0.ipynb',
        'notebooks/korean_neural_sparse_training.ipynb',
        'notebooks/neural_sparse_inference.ipynb',
    ]

    print("="*70)
    print("Fixing Notebook Imports")
    print("="*70)

    for nb_path in notebooks:
        try:
            fix_notebook_imports(nb_path)
        except FileNotFoundError:
            print(f"  ⚠ Skipped (not found): {nb_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n✓ All notebooks processed!")
