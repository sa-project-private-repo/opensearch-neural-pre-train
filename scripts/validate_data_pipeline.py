"""
Validate the complete data pipeline from JSONL to training batch.

This script validates:
1. JSONL file format
2. Dataset loading and output
3. DataCollator batch creation
4. Batch structure for training

Usage:
    python scripts/validate_data_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.data.jsonl_dataset import NeuralSparseJSONLDataset
from src.training.data_collator import NeuralSparseDataCollator


def validate_data_pipeline(
    jsonl_path: str = "dataset/neural_sparse_training/train.jsonl",
    batch_size: int = 8,
    num_negatives: int = 7,
) -> None:
    """
    Validate complete data pipeline.

    Args:
        jsonl_path: Path to JSONL training file
        batch_size: Batch size for testing
        num_negatives: Number of negative samples
    """
    print("=" * 80)
    print("DATA PIPELINE VALIDATION")
    print("=" * 80)

    # Step 1: Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    model_name = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"✓ Loaded tokenizer: {model_name}")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Step 2: Create dataset
    print(f"\n[2/5] Loading dataset from {jsonl_path}...")
    dataset = NeuralSparseJSONLDataset(
        jsonl_path=jsonl_path,
        num_negatives=num_negatives,
        validate_format=True,
    )
    print(f"✓ Dataset loaded successfully")

    # Validate dataset output
    print("\n  Validating dataset output...")
    sample = dataset[0]
    required_keys = ["query", "positive_doc", "negative_docs"]
    for key in required_keys:
        if key not in sample:
            raise ValueError(f"Missing key '{key}' in dataset output")
        print(f"    ✓ {key:20s}: {type(sample[key])}")

    # Validate types
    assert isinstance(sample["query"], str), f"query must be str, got {type(sample['query'])}"
    assert isinstance(
        sample["positive_doc"], str
    ), f"positive_doc must be str, got {type(sample['positive_doc'])}"
    assert isinstance(
        sample["negative_docs"], list
    ), f"negative_docs must be list, got {type(sample['negative_docs'])}"
    assert len(sample["negative_docs"]) == num_negatives, (
        f"Expected {num_negatives} negatives, got {len(sample['negative_docs'])}"
    )

    print(f"    ✓ All types valid")
    print(f"    ✓ Number of negatives: {len(sample['negative_docs'])}")

    # Step 3: Create DataCollator
    print(f"\n[3/5] Creating DataCollator...")
    data_collator = NeuralSparseDataCollator(
        tokenizer=tokenizer,
        query_max_length=64,
        doc_max_length=256,
        num_negatives=num_negatives,
    )
    print(f"✓ DataCollator created")
    print(f"  Query max length: 64")
    print(f"  Doc max length: 256")
    print(f"  Num negatives: {num_negatives}")

    # Step 4: Create DataLoader
    print(f"\n[4/5] Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,  # Use 0 for debugging
    )
    print(f"✓ DataLoader created")
    print(f"  Batch size: {batch_size}")
    print(f"  Num batches: {len(dataloader)}")

    # Step 5: Validate batch structure
    print(f"\n[5/5] Validating batch structure...")
    batch = next(iter(dataloader))

    print("\n  Batch keys:")
    for key in sorted(batch.keys()):
        if isinstance(batch[key], torch.Tensor):
            print(f"    {key:30s}: Tensor {tuple(batch[key].shape)}")
        elif isinstance(batch[key], list):
            if len(batch[key]) > 0:
                if isinstance(batch[key][0], str):
                    print(f"    {key:30s}: List[str] (len={len(batch[key])})")
                elif isinstance(batch[key][0], list):
                    print(
                        f"    {key:30s}: List[List[str]] (len={len(batch[key])}, nested={len(batch[key][0])})"
                    )
                else:
                    print(f"    {key:30s}: List[{type(batch[key][0]).__name__}]")
            else:
                print(f"    {key:30s}: List (empty)")

    # Validate required keys for student model (tokenized inputs)
    print("\n  Student model inputs (tokenized):")
    student_keys = [
        "query_input_ids",
        "query_attention_mask",
        "pos_doc_input_ids",
        "pos_doc_attention_mask",
        "neg_doc_input_ids",
        "neg_doc_attention_mask",
    ]

    all_present = True
    for key in student_keys:
        if key in batch:
            print(f"    ✓ {key}")
        else:
            print(f"    ✗ {key} MISSING!")
            all_present = False

    if not all_present:
        raise ValueError("Missing required student model keys")

    # Validate required keys for teacher model (raw text)
    print("\n  Teacher model inputs (raw text):")
    teacher_keys = ["queries", "positive_docs", "negative_docs"]

    for key in teacher_keys:
        if key in batch:
            print(f"    ✓ {key}")
        else:
            print(f"    ✗ {key} MISSING!")
            all_present = False

    if not all_present:
        raise ValueError("Missing required teacher model keys")

    # Validate shapes
    print("\n  Shape validation:")
    actual_batch_size = len(batch["queries"])
    print(f"    Batch size: {actual_batch_size}")

    # Query shapes
    assert batch["query_input_ids"].shape[0] == actual_batch_size
    assert batch["query_attention_mask"].shape[0] == actual_batch_size
    print(f"    ✓ Query tensors: {batch['query_input_ids'].shape}")

    # Positive doc shapes
    assert batch["pos_doc_input_ids"].shape[0] == actual_batch_size
    assert batch["pos_doc_attention_mask"].shape[0] == actual_batch_size
    print(f"    ✓ Positive doc tensors: {batch['pos_doc_input_ids'].shape}")

    # Negative docs shapes
    assert batch["neg_doc_input_ids"].shape[0] == actual_batch_size
    assert batch["neg_doc_input_ids"].shape[1] == num_negatives
    assert batch["neg_doc_attention_mask"].shape[0] == actual_batch_size
    assert batch["neg_doc_attention_mask"].shape[1] == num_negatives
    print(
        f"    ✓ Negative docs tensors: {batch['neg_doc_input_ids'].shape} (batch_size, num_neg, seq_len)"
    )

    # Validate raw text
    print("\n  Raw text validation:")
    assert len(batch["queries"]) == actual_batch_size
    assert all(isinstance(q, str) for q in batch["queries"])
    print(f"    ✓ queries: {len(batch['queries'])} strings")

    assert len(batch["positive_docs"]) == actual_batch_size
    assert all(isinstance(d, str) for d in batch["positive_docs"])
    print(f"    ✓ positive_docs: {len(batch['positive_docs'])} strings")

    assert len(batch["negative_docs"]) == actual_batch_size
    assert all(isinstance(neg_list, list) for neg_list in batch["negative_docs"])
    assert all(
        len(neg_list) == num_negatives for neg_list in batch["negative_docs"]
    )
    assert all(
        all(isinstance(d, str) for d in neg_list)
        for neg_list in batch["negative_docs"]
    )
    print(
        f"    ✓ negative_docs: {len(batch['negative_docs'])} lists of {num_negatives} strings each"
    )

    # Print sample content
    print("\n  Sample batch content:")
    print(f"    Query 0: {batch['queries'][0][:80]}...")
    print(f"    Pos doc 0: {batch['positive_docs'][0][:80]}...")
    print(f"    Neg doc 0[0]: {batch['negative_docs'][0][0][:80]}...")

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"✓ Dataset: {len(dataset):,} samples loaded")
    print(f"✓ DataLoader: {len(dataloader):,} batches")
    print(f"✓ Batch size: {actual_batch_size}")
    print(f"✓ Num negatives: {num_negatives}")
    print(f"✓ All required keys present")
    print(f"✓ All shapes correct")
    print(f"✓ All types valid")
    print("\n✓✓✓ DATA PIPELINE IS VALID ✓✓✓")
    print("\nReady for training!")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate data pipeline for neural sparse training"
    )
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default="dataset/neural_sparse_training/train.jsonl",
        help="Path to JSONL training file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for validation",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=7,
        help="Number of negative samples",
    )

    args = parser.parse_args()

    try:
        validate_data_pipeline(
            jsonl_path=args.jsonl_path,
            batch_size=args.batch_size,
            num_negatives=args.num_negatives,
        )
    except Exception as e:
        print("\n" + "=" * 80)
        print("VALIDATION FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
