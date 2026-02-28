"""Export V33 SPLADE model to HuggingFace format."""
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model.splade_modern import SPLADEModernBERT
from transformers import AutoTokenizer


def main() -> None:
    """Load trained SPLADEModernBERT and export inner MLM model to HF format."""
    output_dir = Path("huggingface/v33")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = "outputs/train_v33/final_model/model.pt"
    base_model_name = "skt/A.X-Encoder-base"

    print(f"Loading SPLADEModernBERT from {model_path} ...")
    model = SPLADEModernBERT(base_model_name)
    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Saving inner MLM model to {output_dir} (safetensors) ...")
    model.model.save_pretrained(str(output_dir), safe_serialization=True)

    print(f"Saving tokenizer from {base_model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(str(output_dir))

    files = sorted(f.name for f in output_dir.iterdir())
    print(f"Export complete. Files in {output_dir}:")
    for f in files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
