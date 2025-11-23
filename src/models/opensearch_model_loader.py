"""Utilities for loading OpenSearch neural sparse models."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def download_opensearch_sparse_model(
    model_id: str,
    cache_dir: Optional[str] = None,
) -> tuple[str, dict]:
    """
    Download OpenSearch neural sparse model from HuggingFace Hub.

    This function handles the special case where OpenSearch models may not
    have a neural_sparse_head.pt file. It downloads the base transformer
    and returns information needed to initialize the model.

    Args:
        model_id: HuggingFace model ID (e.g., "opensearch-project/...")
        cache_dir: Optional cache directory for downloads

    Returns:
        Tuple of (base_model_path, model_config)
    """
    from huggingface_hub import snapshot_download

    print(f"Downloading OpenSearch model: {model_id}")

    # Download entire model repository
    model_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.json", "*.bin", "*.pt", "*.safetensors"],
    )

    print(f"Model downloaded to: {model_path}")

    # Check if neural_sparse_head.pt exists
    model_path_obj = Path(model_path)
    head_path = model_path_obj / "neural_sparse_head.pt"

    if head_path.exists():
        # Load checkpoint config
        checkpoint = torch.load(head_path, map_location="cpu")
        config = {
            "base_model": checkpoint.get("model_name", model_id),
            "has_pretrained_head": True,
            "head_path": str(head_path),
            "max_length": checkpoint.get("max_length", 256),
            "vocab_size": checkpoint.get("vocab_size"),
            "hidden_size": checkpoint.get("hidden_size"),
        }
    else:
        # No pretrained head - will need to initialize randomly
        print(
            f"Warning: neural_sparse_head.pt not found in {model_id}\n"
            f"The projection layer will be randomly initialized"
        )
        config = {
            "base_model": model_path,
            "has_pretrained_head": False,
            "head_path": None,
            "max_length": 256,
        }

    return model_path, config


def create_neural_sparse_head(
    hidden_size: int,
    vocab_size: int,
) -> nn.Linear:
    """
    Create a neural sparse projection head.

    Args:
        hidden_size: Hidden dimension from base encoder
        vocab_size: Vocabulary size for output

    Returns:
        Linear projection layer
    """
    projection = nn.Linear(hidden_size, vocab_size)

    # Initialize with small random weights
    nn.init.xavier_uniform_(projection.weight, gain=0.01)
    nn.init.zeros_(projection.bias)

    return projection


def verify_opensearch_model(model_id: str) -> dict:
    """
    Verify that an OpenSearch model can be loaded and get its config.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Dictionary with model information
    """
    from transformers import AutoConfig

    print(f"Verifying model: {model_id}")

    # Get model config
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    info = {
        "model_id": model_id,
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
    }

    print("Model information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    return info


if __name__ == "__main__":
    # Test the loader
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="opensearch-project/opensearch-neural-sparse-encoding-v1",
        help="HuggingFace model ID to test",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for downloads",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Testing OpenSearch Model Loader")
    print("=" * 80)

    # Verify model
    print("\n1. Verifying model configuration...")
    info = verify_opensearch_model(args.model_id)

    # Download model
    print("\n2. Downloading model...")
    model_path, config = download_opensearch_sparse_model(
        args.model_id,
        cache_dir=args.cache_dir,
    )

    print("\n3. Model configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Test projection head creation
    print("\n4. Creating projection head...")
    if config["has_pretrained_head"]:
        print("  Loading pretrained head from checkpoint")
        checkpoint = torch.load(config["head_path"], map_location="cpu")
        print(f"  Checkpoint keys: {list(checkpoint.keys())}")
    else:
        print("  Creating randomly initialized head")
        projection = create_neural_sparse_head(
            hidden_size=info["hidden_size"],
            vocab_size=info["vocab_size"],
        )
        print(f"  Projection shape: {projection.weight.shape}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
