"""
Convert training checkpoint to HuggingFace format for evaluation.

Usage:
    python scripts/convert_checkpoint_to_hf.py \
        --checkpoint outputs/train_v24/checkpoint_epoch10_step112340 \
        --output huggingface/v24_epoch10
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.splade_xlmr import SPLADEDocXLMR, SPLADEDocContextGated


def convert_checkpoint(
    checkpoint_path: str,
    output_path: str,
    model_name: str = "xlm-roberta-base",
    model_class: str = "SPLADEDocXLMR",
) -> None:
    """
    Convert training checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to checkpoint directory or model.pt file
        output_path: Output directory for HuggingFace model
        model_name: Base model name (xlm-roberta-base or xlm-roberta-large)
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find model.pt file
    if checkpoint_path.is_dir():
        model_file = checkpoint_path / "model.pt"
    else:
        model_file = checkpoint_path

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    print(f"Loading checkpoint: {model_file}")

    # Create model and load state dict
    if model_class == "SPLADEDocContextGated":
        model = SPLADEDocContextGated(model_name=model_name, use_mlm_head=True)
    else:
        model = SPLADEDocXLMR(model_name=model_name, use_mlm_head=True)

    state_dict = torch.load(model_file, map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Save the underlying MLM model using HuggingFace's save_pretrained
    # This handles shared tensors correctly
    model.model.save_pretrained(output_path, safe_serialization=True)
    print(f"Saved model weights: {output_path / 'model.safetensors'}")
    print(f"Saved config: {output_path / 'config.json'}")

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_path)
    print(f"Saved tokenizer files")

    # Save checkpoint info
    info_file = checkpoint_path / "checkpoint_info.json" if checkpoint_path.is_dir() else None
    if info_file and info_file.exists():
        with open(info_file) as f:
            ckpt_info = json.load(f)

        readme_content = f"""# SPLADE V24 XLM-RoBERTa Checkpoint

## Model Info
- Base model: {model_name}
- Epoch: {ckpt_info.get('epoch', 'N/A')}
- Step: {ckpt_info.get('step', 'N/A')}
- Timestamp: {ckpt_info.get('timestamp', 'N/A')}

## Metrics
{json.dumps(ckpt_info.get('metrics', {}), indent=2)}

## Usage
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("{output_path}")
model = AutoModelForMaskedLM.from_pretrained("{output_path}")
```
"""
        with open(output_path / "README.md", "w") as f:
            f.write(readme_content)

    print(f"\n✓ Conversion complete: {output_path}")
    print(f"  Files created:")
    for f in output_path.iterdir():
        size = f.stat().st_size / (1024 * 1024)
        print(f"    - {f.name}: {size:.1f} MB" if size > 1 else f"    - {f.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert training checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        required=True,
        help="Path to checkpoint directory or model.pt file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for HuggingFace model"
    )
    parser.add_argument(
        "--model-name", "-m",
        default="xlm-roberta-base",
        help="Base model name (default: xlm-roberta-base)"
    )
    parser.add_argument(
        "--model-class",
        default="SPLADEDocXLMR",
        choices=["SPLADEDocXLMR", "SPLADEDocContextGated"],
        help="Model class (default: SPLADEDocXLMR)"
    )

    args = parser.parse_args()
    convert_checkpoint(args.checkpoint, args.output, args.model_name, args.model_class)


if __name__ == "__main__":
    main()
