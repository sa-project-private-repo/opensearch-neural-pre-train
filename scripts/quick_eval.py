"""
Quick evaluation script for SPLADE models.

Usage:
    python scripts/quick_eval.py --model huggingface/v24_best
    python scripts/quick_eval.py --model huggingface/v24_epoch10 --compare huggingface/v22.0
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm


def load_model(model_path: str, device: str = "cuda"):
    """Load SPLADE model and tokenizer."""
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def get_special_token_ids(tokenizer, include_common: bool = True) -> set:
    """Get special token IDs to mask during inference."""
    special_ids = set()

    # Standard special tokens
    for attr in ['bos_token_id', 'eos_token_id', 'pad_token_id', 'unk_token_id',
                 'cls_token_id', 'sep_token_id', 'mask_token_id']:
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            special_ids.add(token_id)

    # Additional special tokens from added_tokens
    if hasattr(tokenizer, 'additional_special_tokens_ids'):
        special_ids.update(tokenizer.additional_special_tokens_ids)

    # XLM-RoBERTa specific: 0=<s>, 1=<pad>, 2=</s>, 3=<unk>, 250001=<mask>
    special_ids.update({0, 1, 2, 3})

    # Common non-informative tokens (punctuation, SentencePiece markers)
    if include_common:
        # Token 4: , (comma prefix)
        # Token 5: . (period)
        # Token 6: ▁ (SentencePiece space marker, decodes to empty)
        special_ids.update({4, 5, 6})

    return special_ids


def encode_sparse(
    texts: List[str],
    tokenizer,
    model,
    device: str = "cuda",
    max_length: int = 192,
    mask_special_tokens: bool = True,
) -> List[Dict[str, float]]:
    """Encode texts to sparse vectors."""
    sparse_vectors = []

    # Get special token IDs to mask
    special_ids = get_special_token_ids(tokenizer) if mask_special_tokens else set()

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits

            # SPLADE: max pooling + ReLU + log
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            masked_logits = logits * attention_mask
            sparse = torch.max(masked_logits, dim=1)[0]
            sparse = torch.relu(sparse)
            sparse = torch.log1p(sparse)

            # Mask special tokens
            if mask_special_tokens and special_ids:
                for sid in special_ids:
                    if sid < sparse.shape[1]:
                        sparse[0, sid] = 0.0

            # Extract non-zero values
            sparse_vec = {}
            non_zero = sparse[0].nonzero().squeeze(-1)
            for idx in non_zero.tolist():
                if isinstance(idx, int):
                    token = tokenizer.decode([idx])
                    weight = sparse[0][idx].item()
                    if weight > 0.01:  # Threshold
                        sparse_vec[token] = round(weight, 4)

            sparse_vectors.append(sparse_vec)

    return sparse_vectors


def compute_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Compute sparse vector similarity (dot product)."""
    common_keys = set(vec1.keys()) & set(vec2.keys())
    return sum(vec1[k] * vec2[k] for k in common_keys)


def evaluate_retrieval(
    model_path: str,
    validation_path: str = "data/v24.0/val.jsonl",
    num_samples: int = 100,
    device: str = "cuda",
    mask_special_tokens: bool = True,
) -> Dict[str, float]:
    """
    Evaluate retrieval performance on validation set.

    Returns:
        Dict with MRR@10 and Hit@10 metrics
    """
    tokenizer, model = load_model(model_path, device)

    # Load validation data
    val_path = Path(validation_path)
    if not val_path.exists():
        # Try alternative paths
        alt_paths = [
            "data/v22.0/validation_triplets.jsonl",
            "data/validation.jsonl",
        ]
        for alt in alt_paths:
            if Path(alt).exists():
                val_path = Path(alt)
                break

    print(f"Loading validation data from: {val_path}")
    samples = []
    with open(val_path) as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line))

    print(f"Evaluating on {len(samples)} samples...")

    # Compute metrics
    mrr_sum = 0.0
    hit_at_10 = 0

    for sample in tqdm(samples, desc="Evaluating"):
        query = sample.get("query", sample.get("anchor", ""))
        positive = sample.get("positive", sample.get("pos", ""))
        negatives = sample.get("negatives", sample.get("neg", []))

        if isinstance(negatives, str):
            negatives = [negatives]

        # Encode query and candidates
        query_vec = encode_sparse([query], tokenizer, model, device, mask_special_tokens=mask_special_tokens)[0]
        candidates = [positive] + negatives[:9]  # Top 10 candidates
        candidate_vecs = encode_sparse(candidates, tokenizer, model, device, mask_special_tokens=mask_special_tokens)

        # Rank by similarity
        scores = [(i, compute_similarity(query_vec, cv)) for i, cv in enumerate(candidate_vecs)]
        scores.sort(key=lambda x: x[1], reverse=True)

        # Find positive rank
        for rank, (idx, score) in enumerate(scores, 1):
            if idx == 0:  # Positive is at index 0
                mrr_sum += 1.0 / rank
                if rank <= 10:
                    hit_at_10 += 1
                break

    metrics = {
        "MRR@10": round(mrr_sum / len(samples), 4),
        "Hit@10": round(hit_at_10 / len(samples), 4),
        "num_samples": len(samples),
    }

    return metrics


def analyze_sparsity(model_path: str, device: str = "cuda", mask_special: bool = True):
    """Analyze sparsity characteristics of the model."""
    tokenizer, model = load_model(model_path, device)

    test_texts = [
        "서울에서 맛있는 식당을 추천해주세요",
        "인공지능 기술의 발전 방향은 어떻게 될까요?",
        "What is the capital of France?",
        "How to learn programming efficiently?",
    ]

    mask_status = "ON" if mask_special else "OFF"
    print(f"\n=== Sparsity Analysis (special token mask: {mask_status}) ===")
    for text in test_texts:
        vec = encode_sparse([text], tokenizer, model, device, mask_special_tokens=mask_special)[0]
        print(f"\nText: {text[:50]}...")
        print(f"  Active tokens: {len(vec)}")
        top_tokens = dict(sorted(vec.items(), key=lambda x: -x[1])[:10])
        print(f"  Top 10 tokens: {top_tokens}")


def main():
    parser = argparse.ArgumentParser(description="Quick evaluation for SPLADE models")
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to model directory"
    )
    parser.add_argument(
        "--compare", "-c",
        help="Optional model to compare against"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--sparsity-only",
        action="store_true",
        help="Only run sparsity analysis"
    )
    parser.add_argument(
        "--no-mask-special",
        action="store_true",
        help="Disable special token masking"
    )

    args = parser.parse_args()
    mask_special = not args.no_mask_special

    if args.sparsity_only:
        analyze_sparsity(args.model, args.device, mask_special=mask_special)
        return

    mask_label = "with special token masking" if mask_special else "without masking"
    print(f"\n{'='*60}")
    print(f"Evaluating: {args.model}")
    print(f"Mode: {mask_label}")
    print(f"{'='*60}")

    start = time.time()
    metrics = evaluate_retrieval(
        args.model,
        num_samples=args.num_samples,
        device=args.device,
        mask_special_tokens=mask_special,
    )
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"Results for: {args.model}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"  Time: {elapsed:.1f}s")

    analyze_sparsity(args.model, args.device, mask_special=mask_special)

    if args.compare:
        print(f"\n{'='*60}")
        print(f"Comparing with: {args.compare}")
        print(f"{'='*60}")

        compare_metrics = evaluate_retrieval(
            args.compare,
            num_samples=args.num_samples,
            device=args.device,
            mask_special_tokens=mask_special,
        )

        print(f"\nComparison:")
        print(f"{'Metric':<15} {'Model':<15} {'Compare':<15} {'Diff':<10}")
        print("-" * 55)
        for k in ["MRR@10", "Hit@10"]:
            diff = metrics[k] - compare_metrics[k]
            diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
            print(f"{k:<15} {metrics[k]:<15} {compare_metrics[k]:<15} {diff_str}")


if __name__ == "__main__":
    main()
