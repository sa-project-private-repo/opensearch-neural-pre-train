#!/usr/bin/env python3
"""
Test model save/load functionality for custom PyTorch models
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from src.dataset_manager import DatasetManager


class DummyModel(nn.Module):
    """Dummy PyTorch model for testing"""

    def __init__(self, model_name="klue/bert-base"):
        super().__init__()
        self.model_name = model_name
        self.vocab_size = 30000
        self.linear = nn.Linear(768, 100)

    def forward(self, x):
        return self.linear(x)


def test_save_load_custom_model():
    """Test saving and loading custom PyTorch model"""
    print("="*70)
    print("🧪 Testing Custom PyTorch Model Save/Load")
    print("="*70)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n📁 Using temporary directory: {tmpdir}")

        # Initialize DatasetManager
        dm = DatasetManager(base_path=tmpdir)

        # Create dummy model
        print("\n1️⃣ Creating dummy model...")
        model = DummyModel(model_name="klue/bert-base")
        print(f"   Model: {model.__class__.__name__}")
        print(f"   Vocab size: {model.vocab_size}")

        # Create dummy tokenizer
        print("\n2️⃣ Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        print(f"   Tokenizer loaded: {len(tokenizer)} tokens")

        # Save model
        print("\n3️⃣ Saving model...")
        try:
            saved_path = dm.save_model(
                model=model,
                tokenizer=tokenizer,
                model_dir="test_model",
                subdir="test"
            )
            print(f"   ✅ Model saved to: {saved_path}")

            # Check saved files
            saved_files = list(saved_path.rglob("*"))
            print(f"\n   📄 Saved files ({len(saved_files)} total):")
            for f in sorted(saved_files):
                if f.is_file():
                    size_mb = f.stat().st_size / (1024**2)
                    print(f"      - {f.name} ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"   ❌ Failed to save model: {e}")
            return False

        # Load model
        print("\n4️⃣ Loading model...")
        try:
            loaded_model, loaded_tokenizer = dm.load_model(
                model_class=DummyModel,
                model_dir="test_model",
                subdir="test",
                device="cpu"
            )
            print(f"   ✅ Model loaded")
            print(f"   Model type: {loaded_model.__class__.__name__}")
            print(f"   Tokenizer vocab size: {len(loaded_tokenizer)}")
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Verify state dict
        print("\n5️⃣ Verifying state dict...")
        try:
            # Check if weights are the same
            original_weight = model.linear.weight.data
            loaded_weight = loaded_model.linear.weight.data

            if torch.allclose(original_weight, loaded_weight):
                print("   ✅ Weights match!")
            else:
                print("   ❌ Weights don't match!")
                return False
        except Exception as e:
            print(f"   ❌ Failed to verify: {e}")
            return False

        print("\n" + "="*70)
        print("✅ All tests passed!")
        print("="*70)

        return True


if __name__ == "__main__":
    success = test_save_load_custom_model()

    if success:
        print("\n✅ Custom PyTorch model save/load is working correctly!")
        print("\nYou can now use dm.save_model() with custom PyTorch models")
        print("including OpenSearchDocEncoder in the notebooks.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
