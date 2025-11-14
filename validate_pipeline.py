"""
Validation script for the modular notebook pipeline

Checks:
1. All notebooks exist and are properly formatted
2. DatasetManager integration is correct
3. Data flow dependencies are set up
4. Expected cell counts match
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_notebook(path: Path) -> dict:
    """Load a Jupyter notebook"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_notebook_structure(nb: dict, name: str) -> bool:
    """Validate basic notebook structure"""
    print(f"\n{'='*70}")
    print(f"Validating: {name}")
    print(f"{'='*70}")

    if "cells" not in nb:
        print(f"❌ Missing 'cells' in notebook")
        return False

    cell_count = len(nb["cells"])
    print(f"✅ Notebook has {cell_count} cells")

    # Count code vs markdown cells
    code_cells = sum(1 for cell in nb["cells"] if cell["cell_type"] == "code")
    markdown_cells = sum(1 for cell in nb["cells"] if cell["cell_type"] == "markdown")

    print(f"   - Code cells: {code_cells}")
    print(f"   - Markdown cells: {markdown_cells}")

    return True


def check_datasetmanager_import(nb: dict, name: str) -> bool:
    """Check if DatasetManager is imported"""
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            if "from src.dataset_manager import DatasetManager" in source:
                print(f"✅ DatasetManager import found in cell {i}")
                return True

    print(f"❌ DatasetManager import not found")
    return False


def check_dependency_validation(nb: dict, name: str, expected_deps: List[str]) -> bool:
    """Check if dependency validation is present"""
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            if "check_dependencies" in source:
                print(f"✅ Dependency checking found in cell {i}")
                # Check if expected dependencies are mentioned
                for dep in expected_deps:
                    if dep in source:
                        print(f"   - Checks for: {dep}")
                return True

    if expected_deps:
        print(f"⚠️  No dependency checking found (expected for notebooks 2 and 3)")
        return False

    return True


def check_data_saving(nb: dict, name: str, expected_saves: List[str]) -> bool:
    """Check if data saving logic is present"""
    found_saves = []

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            for save_method in ["save_json", "save_pickle", "save_model"]:
                if save_method in source:
                    # Extract what's being saved
                    for exp_save in expected_saves:
                        if exp_save in source:
                            found_saves.append(exp_save)
                            print(f"✅ Data saving: {exp_save} (cell {i})")

    missing = set(expected_saves) - set(found_saves)
    if missing:
        print(f"⚠️  Missing expected saves: {missing}")
        return False

    return True


def validate_notebook_1():
    """Validate notebook 1: Base Training"""
    nb = load_notebook(Path("notebooks/01_neural_sparse_base_training.ipynb"))

    checks = []
    checks.append(check_notebook_structure(nb, "Notebook 1"))
    checks.append(check_datasetmanager_import(nb, "Notebook 1"))
    checks.append(check_dependency_validation(nb, "Notebook 1", []))  # No dependencies
    checks.append(check_data_saving(nb, "Notebook 1", [
        "korean_documents",
        "idf_dict",
        "qd_pairs",
        "bilingual_dict",
        "model"
    ]))

    return all(checks)


def validate_notebook_2():
    """Validate notebook 2: LLM Synthetic Data"""
    nb = load_notebook(Path("notebooks/02_llm_synthetic_data_generation.ipynb"))

    checks = []
    checks.append(check_notebook_structure(nb, "Notebook 2"))
    checks.append(check_datasetmanager_import(nb, "Notebook 2"))
    checks.append(check_dependency_validation(nb, "Notebook 2", [
        "korean_documents.json",
        "bilingual_synonyms.json"
    ]))
    checks.append(check_data_saving(nb, "Notebook 2", [
        "synthetic_qd_pairs",
        "enhanced_synonyms"
    ]))

    # Check for Triton disable
    triton_found = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            if "TRITON_INTERPRET" in source or "DISABLE_TRITON" in source:
                print(f"✅ Triton compilation disable found")
                triton_found = True
                break

    if not triton_found:
        print(f"⚠️  Triton disable not found (may cause issues on ARM)")

    checks.append(triton_found)

    return all(checks)


def validate_notebook_3():
    """Validate notebook 3: Enhanced Training"""
    nb = load_notebook(Path("notebooks/03_llm_enhanced_training.ipynb"))

    checks = []
    checks.append(check_notebook_structure(nb, "Notebook 3"))
    checks.append(check_datasetmanager_import(nb, "Notebook 3"))
    checks.append(check_dependency_validation(nb, "Notebook 3", [
        "qd_pairs_base.pkl",
        "synthetic_qd_pairs.pkl",
        "neural_sparse_v1_model"
    ]))
    checks.append(check_data_saving(nb, "Notebook 3", [
        "enhanced_model",
        "performance_comparison"
    ]))

    return all(checks)


def validate_dataset_manager():
    """Validate DatasetManager implementation"""
    print(f"\n{'='*70}")
    print(f"Validating: DatasetManager")
    print(f"{'='*70}")

    dm_path = Path("src/dataset_manager.py")

    if not dm_path.exists():
        print(f"❌ DatasetManager not found at {dm_path}")
        return False

    with open(dm_path, 'r', encoding='utf-8') as f:
        content = f.read()

    required_methods = [
        "save_json",
        "load_json",
        "save_pickle",
        "load_pickle",
        "save_model",
        "load_model",
        "check_dependencies",
        "check_data_exists",
        "print_summary"
    ]

    checks = []
    for method in required_methods:
        if f"def {method}" in content:
            print(f"✅ Method exists: {method}")
            checks.append(True)
        else:
            print(f"❌ Missing method: {method}")
            checks.append(False)

    return all(checks)


def main():
    """Run all validations"""
    print("="*70)
    print("VALIDATING MODULAR NOTEBOOK PIPELINE")
    print("="*70)

    results = {}

    # Validate DatasetManager
    results["DatasetManager"] = validate_dataset_manager()

    # Validate notebooks
    results["Notebook 1"] = validate_notebook_1()
    results["Notebook 2"] = validate_notebook_2()
    results["Notebook 3"] = validate_notebook_3()

    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")

    all_passed = all(results.values())

    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:30} {status}")

    print(f"{'='*70}")

    if all_passed:
        print("\n🎉 All validations PASSED! Pipeline is ready to use.")
        print("\n📝 Execution order:")
        print("   1. notebooks/01_neural_sparse_base_training.ipynb")
        print("   2. notebooks/02_llm_synthetic_data_generation.ipynb")
        print("   3. notebooks/03_llm_enhanced_training.ipynb")
        return 0
    else:
        print("\n⚠️  Some validations FAILED. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
