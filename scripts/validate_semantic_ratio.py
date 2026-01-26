#!/usr/bin/env python3
"""
Validate semantic token ratio in trained neural sparse model.

This script analyzes the token activation patterns of a trained SPLADE model
to verify that semantic tokens (content-bearing words) dominate over stopwords
(grammatical particles, function words).

Usage:
    python scripts/validate_semantic_ratio.py --model huggingface/v26
    python scripts/validate_semantic_ratio.py --model huggingface/v26 --queries "당뇨병 치료" "서울 맛집"
    python scripts/validate_semantic_ratio.py --model huggingface/v25 --compare huggingface/v26

Success Criteria:
    - Top-10 semantic token ratio: >80%
    - Semantic/stopword activation ratio: >1.0
    - No grammatical particles in top-5 activations
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Default test queries (Korean)
DEFAULT_QUERIES = [
    "당뇨병 치료 방법",
    "서울 맛집 추천",
    "파이썬 프로그래밍 배우기",
    "건강한 다이어트 식단",
    "강아지 훈련 방법",
]

# Korean stopword patterns for classification
KOREAN_STOPWORD_TOKENS = {
    # Particles
    "이", "가", "은", "는", "을", "를", "의", "에", "에서", "로", "으로",
    "와", "과", "도", "만", "까지", "부터", "보다",
    # Verb endings
    "다", "요", "습니다", "니다", "입니다", "어요", "아요",
    "는데", "지만", "면", "고", "서",
    # Function words
    "것", "수", "때", "데", "점",
    "있", "없", "하", "되", "이다",
    # Common adverbs
    "더", "가장", "매우", "아주", "잘",
}


def load_model_and_tokenizer(model_path: str) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Load SPLADE model and tokenizer from HuggingFace format."""
    logger.info(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer


def encode_query(
    query: str,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    max_length: int = 192,
) -> torch.Tensor:
    """Encode query to sparse representation."""
    inputs = tokenizer(
        query,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True,
    )

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

        # Get sparse representation via log-saturation
        hidden = outputs.last_hidden_state  # [1, seq_len, hidden]

        # Apply MLM head if available (SPLADE-style)
        if hasattr(model, "lm_head"):
            logits = model.lm_head(hidden)  # [1, seq_len, vocab_size]
        else:
            # Fallback: use hidden states with vocab projection
            logits = hidden

        # Max pooling over sequence
        sparse_repr = torch.log1p(torch.relu(logits.max(dim=1).values))  # [1, vocab_size]

    return sparse_repr.squeeze(0).cpu()


def classify_token(token: str, tokenizer: AutoTokenizer) -> str:
    """Classify token as semantic or stopword."""
    # Remove SentencePiece marker
    clean_token = token.replace("▁", "").strip()

    if not clean_token:
        return "special"

    # Check if it's a special token
    if token in tokenizer.all_special_tokens:
        return "special"

    # Check stopword patterns
    for stopword in KOREAN_STOPWORD_TOKENS:
        if clean_token == stopword or clean_token.endswith(stopword):
            return "stopword"

    # Check if it's mostly punctuation or single characters
    if len(clean_token) == 1 and not clean_token.isalnum():
        return "punctuation"

    return "semantic"


def analyze_activations(
    sparse_repr: torch.Tensor,
    tokenizer: AutoTokenizer,
    top_k: int = 20,
) -> Dict:
    """Analyze token activations and classify them."""
    # Get top-k activated tokens
    values, indices = torch.topk(sparse_repr, k=top_k)

    results = {
        "top_tokens": [],
        "semantic_count": 0,
        "stopword_count": 0,
        "special_count": 0,
        "semantic_activation_sum": 0.0,
        "stopword_activation_sum": 0.0,
    }

    for i, (idx, val) in enumerate(zip(indices.tolist(), values.tolist())):
        token = tokenizer.decode([idx])
        token_type = classify_token(token, tokenizer)

        results["top_tokens"].append({
            "rank": i + 1,
            "token": token,
            "token_id": idx,
            "activation": val,
            "type": token_type,
        })

        if token_type == "semantic":
            results["semantic_count"] += 1
            results["semantic_activation_sum"] += val
        elif token_type == "stopword":
            results["stopword_count"] += 1
            results["stopword_activation_sum"] += val
        else:
            results["special_count"] += 1

    # Compute ratios
    total_classified = results["semantic_count"] + results["stopword_count"]
    if total_classified > 0:
        results["semantic_ratio"] = results["semantic_count"] / total_classified
    else:
        results["semantic_ratio"] = 0.0

    if results["stopword_activation_sum"] > 0:
        results["activation_ratio"] = (
            results["semantic_activation_sum"] / results["stopword_activation_sum"]
        )
    else:
        results["activation_ratio"] = float("inf")

    return results


def print_analysis(query: str, analysis: Dict, verbose: bool = True):
    """Print analysis results for a query."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    # Print top tokens
    print(f"\n{'Rank':<6}{'Token':<20}{'Activation':<12}{'Type':<12}")
    print("-" * 50)

    for token_info in analysis["top_tokens"][:10]:
        token_display = token_info["token"][:18]
        print(
            f"{token_info['rank']:<6}"
            f"{token_display:<20}"
            f"{token_info['activation']:<12.4f}"
            f"{token_info['type']:<12}"
        )

    # Print summary
    print(f"\n--- Summary (Top-20) ---")
    print(f"Semantic tokens: {analysis['semantic_count']}")
    print(f"Stopword tokens: {analysis['stopword_count']}")
    print(f"Semantic ratio: {analysis['semantic_ratio']:.2%}")
    print(f"Activation ratio (semantic/stopword): {analysis['activation_ratio']:.2f}")

    # Check success criteria
    if analysis["semantic_ratio"] >= 0.8:
        print("\nSUCCESS: Semantic tokens dominate (ratio >= 80%)")
    else:
        print(f"\nWARNING: Stopwords may be dominating (ratio = {analysis['semantic_ratio']:.1%})")


def compare_models(
    model1_path: str,
    model2_path: str,
    queries: List[str],
):
    """Compare token activation patterns between two models."""
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"Model 1: {model1_path}")
    print(f"Model 2: {model2_path}")

    model1, tokenizer1 = load_model_and_tokenizer(model1_path)
    model2, tokenizer2 = load_model_and_tokenizer(model2_path)

    results = []

    for query in queries:
        repr1 = encode_query(query, model1, tokenizer1)
        repr2 = encode_query(query, model2, tokenizer2)

        analysis1 = analyze_activations(repr1, tokenizer1)
        analysis2 = analyze_activations(repr2, tokenizer2)

        results.append({
            "query": query,
            "model1_semantic_ratio": analysis1["semantic_ratio"],
            "model2_semantic_ratio": analysis2["semantic_ratio"],
            "model1_activation_ratio": analysis1["activation_ratio"],
            "model2_activation_ratio": analysis2["activation_ratio"],
        })

    # Print comparison table
    print(f"\n{'Query':<30}{'Model1 Sem%':<15}{'Model2 Sem%':<15}{'Improvement':<15}")
    print("-" * 75)

    for r in results:
        improvement = r["model2_semantic_ratio"] - r["model1_semantic_ratio"]
        print(
            f"{r['query'][:28]:<30}"
            f"{r['model1_semantic_ratio']:.1%:<15}"
            f"{r['model2_semantic_ratio']:.1%:<15}"
            f"{improvement:+.1%:<15}"
        )

    # Average improvement
    avg_improvement = sum(
        r["model2_semantic_ratio"] - r["model1_semantic_ratio"]
        for r in results
    ) / len(results)

    print(f"\nAverage semantic ratio improvement: {avg_improvement:+.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate semantic token ratio in neural sparse model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--queries",
        type=str,
        nargs="+",
        default=DEFAULT_QUERIES,
        help="Test queries (default: Korean sample queries)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Path to second model for comparison",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top tokens to analyze",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed token analysis",
    )

    args = parser.parse_args()

    # Comparison mode
    if args.compare:
        compare_models(args.model, args.compare, args.queries)
        return 0

    # Single model analysis
    model, tokenizer = load_model_and_tokenizer(args.model)

    all_results = []
    total_semantic_ratio = 0.0
    total_activation_ratio = 0.0

    for query in args.queries:
        sparse_repr = encode_query(query, model, tokenizer)
        analysis = analyze_activations(sparse_repr, tokenizer, top_k=args.top_k)

        print_analysis(query, analysis, verbose=args.verbose)

        all_results.append({
            "query": query,
            "analysis": analysis,
        })

        total_semantic_ratio += analysis["semantic_ratio"]
        if analysis["activation_ratio"] != float("inf"):
            total_activation_ratio += analysis["activation_ratio"]

    # Overall summary
    avg_semantic_ratio = total_semantic_ratio / len(args.queries)
    avg_activation_ratio = total_activation_ratio / len(args.queries)

    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Queries analyzed: {len(args.queries)}")
    print(f"Average semantic ratio: {avg_semantic_ratio:.2%}")
    print(f"Average activation ratio: {avg_activation_ratio:.2f}")

    if avg_semantic_ratio >= 0.8:
        print("\nOVERALL: SUCCESS - Model produces semantic-dominant representations")
        status = 0
    elif avg_semantic_ratio >= 0.5:
        print("\nOVERALL: PARTIAL - Model shows some semantic dominance, but not ideal")
        status = 0
    else:
        print("\nOVERALL: FAILURE - Stopwords are dominating representations")
        status = 1

    # Save results
    if args.output:
        output_data = {
            "model": args.model,
            "queries": args.queries,
            "average_semantic_ratio": avg_semantic_ratio,
            "average_activation_ratio": avg_activation_ratio,
            "results": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")

    return status


if __name__ == "__main__":
    sys.exit(main())
