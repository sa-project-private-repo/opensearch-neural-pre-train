#!/usr/bin/env python3
"""
Export trained V25 model to HuggingFace format for benchmark.

The V25 model (SPLADEDocXLMR) wraps XLMRobertaForMaskedLM.
This script extracts the trained weights and saves in HuggingFace format.

Usage:
    python scripts/export_v25_to_huggingface.py
    python scripts/export_v25_to_huggingface.py --checkpoint outputs/train_v25/best_model
    python scripts/export_v25_to_huggingface.py --output huggingface/v25
"""

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, XLMRobertaForMaskedLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_v25_model(
    checkpoint_path: str = "outputs/train_v25/best_model",
    output_dir: str = "huggingface/v25",
    base_model: str = "xlm-roberta-base",
) -> None:
    """
    Export V25 checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to checkpoint directory containing model.pt
        output_dir: Output directory for HuggingFace model
        base_model: Base model name for config/tokenizer
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    model_file = checkpoint_path / "model.pt"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    logger.info(f"Loading checkpoint from: {model_file}")
    state_dict = torch.load(model_file, map_location="cpu")

    # Create base model
    logger.info(f"Creating base model: {base_model}")
    model = XLMRobertaForMaskedLM.from_pretrained(base_model)

    # The V25 model (SPLADEDocXLMR) wraps XLMRobertaForMaskedLM
    # Checkpoint keys: model.roberta.*, model.lm_head.*
    # HuggingFace keys: roberta.*, lm_head.*
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            # Remove "model." prefix to get HuggingFace keys
            new_key = key[6:]  # len("model.") = 6
            new_state_dict[new_key] = value
        elif key.startswith("transformer."):
            # This is for non-MLM head models, map to roberta.*
            new_key = key.replace("transformer.", "roberta.")
            new_state_dict[new_key] = value
        else:
            # Keep other keys as-is
            new_state_dict[key] = value

    # Load the processed state dict
    logger.info(f"Loading {len(new_state_dict)} parameters into model")
    logger.info(f"Sample keys: {list(new_state_dict.keys())[:5]}")

    # Check for missing/unexpected keys
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(new_state_dict.keys())

    missing = model_keys - loaded_keys
    unexpected = loaded_keys - model_keys

    if missing:
        logger.warning(f"Missing keys: {list(missing)[:5]}... ({len(missing)} total)")
    if unexpected:
        logger.warning(f"Unexpected keys: {list(unexpected)[:5]}... ({len(unexpected)} total)")

    # Load with strict=False to handle minor differences
    model.load_state_dict(new_state_dict, strict=False)

    # Save model
    logger.info(f"Saving model to: {output_dir}")
    model.save_pretrained(output_dir)

    # Copy tokenizer
    logger.info("Copying tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_dir)

    # Verify export
    logger.info("Verifying export...")
    loaded_model = XLMRobertaForMaskedLM.from_pretrained(output_dir)
    logger.info(f"Loaded model vocab size: {loaded_model.config.vocab_size}")

    # Test encoding
    test_text = "당뇨병 치료 방법"
    inputs = tokenizer(test_text, return_tensors="pt")
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        logits = outputs.logits

    logger.info(f"Test encoding shape: {logits.shape}")
    logger.info(f"Export complete: {output_dir}")

    # Print top-10 tokens for test query
    sparse = torch.log1p(torch.relu(logits)).max(dim=1).values[0]
    top_values, top_indices = sparse.topk(10)
    logger.info("Top-10 tokens for test query:")
    for idx, val in zip(top_indices.tolist(), top_values.tolist()):
        token = tokenizer.decode([idx])
        logger.info(f"  {token}: {val:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Export V25 model to HuggingFace format"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/train_v25/best_model",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="huggingface/v25",
        help="Output directory for HuggingFace model",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="xlm-roberta-base",
        help="Base model name",
    )

    args = parser.parse_args()

    export_v25_model(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        base_model=args.base_model,
    )


if __name__ == "__main__":
    main()
