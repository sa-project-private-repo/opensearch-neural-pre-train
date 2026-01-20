#!/usr/bin/env python3
"""
Quick V25 training validation script.

Runs a mini training session to verify IDF-aware FLOPS integration
before committing to a full training run.

Usage:
    python scripts/quick_train_v25.py
    python scripts/quick_train_v25.py --samples 1000 --epochs 2
    python scripts/quick_train_v25.py --model huggingface/v24_best
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.losses import SPLADELossV25
from src.train.idf import (
    IDFComputer,
    compute_idf_from_corpus,
    create_stopword_mask,
    get_korean_stopword_ids,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Quick V25 training validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of training samples",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to pre-trained model checkpoint",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="xlm-roberta-base",
        help="Tokenizer name or path",
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify IDF computation, skip training",
    )

    return parser.parse_args()


def create_sample_corpus() -> List[str]:
    """Create a sample Korean corpus for IDF testing."""
    return [
        "서울에서 맛있는 식당 추천해주세요",
        "강남역 근처 맛집 알려주세요",
        "부산 해운대 여행 코스 추천",
        "제주도 가볼만한 곳",
        "대전 명물 음식 뭐가 있나요",
        "인천공항 면세점 추천",
        "경주 역사 유적지 투어",
        "전주 한옥마을 맛집",
        "속초 바다 횟집 추천",
        "광주 무등산 등산 코스",
        "대구 동성로 쇼핑",
        "울산 현대 박물관",
        "수원 화성 관광",
        "성남 판교 카페 추천",
        "용인 에버랜드 팁",
        "천안 아산 맛집",
        "청주 시내 관광",
        "원주 치악산 등산",
        "춘천 닭갈비 맛집",
        "강릉 커피 거리",
    ]


def verify_idf_computation(tokenizer: AutoTokenizer) -> torch.Tensor:
    """
    Verify IDF computation from sample corpus.

    Args:
        tokenizer: XLM-RoBERTa tokenizer

    Returns:
        Computed IDF weights
    """
    logger.info("=" * 60)
    logger.info("Step 1: Verify IDF Computation")
    logger.info("=" * 60)

    corpus = create_sample_corpus()
    logger.info(f"Sample corpus: {len(corpus)} documents")

    # Compute IDF
    idf_weights = compute_idf_from_corpus(
        corpus=corpus,
        tokenizer=tokenizer,
        smoothing="bm25",
        show_progress=True,
    )

    logger.info(f"IDF weights shape: {idf_weights.shape}")
    logger.info(f"IDF min: {idf_weights.min():.4f}")
    logger.info(f"IDF max: {idf_weights.max():.4f}")
    logger.info(f"IDF mean: {idf_weights.mean():.4f}")

    # Check some semantic vs stopword tokens
    semantic_tokens = ["서울", "맛있는", "추천", "식당", "여행"]
    stopword_tokens = ["을", "를", "에", "에서", "가"]

    logger.info("\nSemantic token IDF values:")
    for token in semantic_tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if token_ids:
            token_id = token_ids[0]
            logger.info(f"  {token}: ID={token_id}, IDF={idf_weights[token_id]:.4f}")

    logger.info("\nStopword token IDF values:")
    for token in stopword_tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if token_ids:
            token_id = token_ids[0]
            logger.info(f"  {token}: ID={token_id}, IDF={idf_weights[token_id]:.4f}")

    return idf_weights


def verify_stopword_masking(tokenizer: AutoTokenizer) -> torch.Tensor:
    """
    Verify Korean stopword masking.

    Args:
        tokenizer: XLM-RoBERTa tokenizer

    Returns:
        Stopword mask tensor
    """
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Verify Stopword Masking")
    logger.info("=" * 60)

    # Get stopword IDs
    stopword_ids = get_korean_stopword_ids(tokenizer)
    logger.info(f"Total stopword IDs: {len(stopword_ids)}")

    # Show some examples
    logger.info("\nSample stopword tokens:")
    sample_ids = list(stopword_ids)[:15]
    for token_id in sample_ids:
        if token_id < tokenizer.vocab_size:
            token = tokenizer.decode([token_id])
            logger.info(f"  ID={token_id}: '{token}'")

    # Create mask
    mask = create_stopword_mask(tokenizer)
    masked_count = (mask == 0).sum().item()
    kept_count = (mask == 1).sum().item()

    logger.info(f"\nStopword mask:")
    logger.info(f"  Masked tokens: {masked_count}")
    logger.info(f"  Kept tokens: {kept_count}")
    logger.info(f"  Mask ratio: {100 * masked_count / len(mask):.2f}%")

    return mask


def verify_loss_creation(
    idf_weights: torch.Tensor,
    stopword_mask: torch.Tensor,
) -> SPLADELossV25:
    """
    Verify SPLADELossV25 creation with IDF weights.

    Args:
        idf_weights: Computed IDF weights
        stopword_mask: Stopword mask

    Returns:
        Created loss function
    """
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Verify SPLADELossV25 Creation")
    logger.info("=" * 60)

    loss_fn = SPLADELossV25(
        idf_weights=idf_weights,
        lambda_infonce=3.0,
        lambda_self=0.5,
        lambda_positive=2.0,
        lambda_flops=0.002,
        lambda_min_act=1.0,
        lambda_kd=0.0,  # No KD for quick test
        temperature=0.07,
        idf_alpha=2.5,
        stopword_mask=stopword_mask,
        stopword_penalty=5.0,
    )

    logger.info("SPLADELossV25 created successfully")
    logger.info(f"  lambda_infonce: {loss_fn.lambda_infonce}")
    logger.info(f"  lambda_flops: {loss_fn.lambda_flops}")
    logger.info(f"  stopword_mask: {'Yes' if loss_fn.stopword_mask is not None else 'No'}")

    return loss_fn


def run_quick_training(
    loss_fn: SPLADELossV25,
    tokenizer: AutoTokenizer,
    samples: int = 500,
    epochs: int = 2,
    batch_size: int = 16,
) -> None:
    """
    Run quick training to verify loss computation.

    Args:
        loss_fn: SPLADELossV25 instance
        tokenizer: XLM-RoBERTa tokenizer
        samples: Number of samples
        epochs: Number of epochs
        batch_size: Batch size
    """
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Run Quick Training")
    logger.info("=" * 60)

    vocab_size = tokenizer.vocab_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Move loss to device
    loss_fn = loss_fn.to(device)

    # Create synthetic data
    logger.info(f"Creating {samples} synthetic samples...")

    for epoch in range(1, epochs + 1):
        logger.info(f"\nEpoch {epoch}/{epochs}")

        epoch_loss = 0.0
        epoch_semantic_ratio = 0.0
        num_batches = samples // batch_size

        loss_fn.reset_metrics()

        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch}"):
            # Create synthetic sparse representations
            # Simulate sparse vectors with most values being 0
            anchor_repr = torch.zeros(batch_size, vocab_size, device=device)
            positive_repr = torch.zeros(batch_size, vocab_size, device=device)
            negative_repr = torch.zeros(batch_size, vocab_size, device=device)

            # Add some random activations (simulate SPLADE output)
            for i in range(batch_size):
                # Activate ~100 random tokens per sample
                active_indices = torch.randint(0, vocab_size, (100,))
                anchor_repr[i, active_indices] = torch.rand(100, device=device) * 5
                positive_repr[i, active_indices] = torch.rand(100, device=device) * 5
                negative_repr[i, active_indices] = torch.rand(100, device=device) * 5

            # Create input_ids and attention_mask
            seq_len = 64
            anchor_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            anchor_attention_mask = torch.ones(batch_size, seq_len, device=device)
            positive_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            positive_attention_mask = torch.ones(batch_size, seq_len, device=device)

            # Compute loss
            loss, loss_dict = loss_fn(
                anchor_repr=anchor_repr,
                positive_repr=positive_repr,
                negative_repr=negative_repr,
                anchor_input_ids=anchor_input_ids,
                anchor_attention_mask=anchor_attention_mask,
                positive_input_ids=positive_input_ids,
                positive_attention_mask=positive_attention_mask,
            )

            epoch_loss += loss_dict["total"]
            epoch_semantic_ratio += loss_dict["semantic_ratio"]

        # Log epoch results
        avg_loss = epoch_loss / num_batches
        avg_semantic_ratio = loss_fn.get_average_semantic_ratio()

        logger.info(f"  Average loss: {avg_loss:.4f}")
        logger.info(f"  Components:")
        logger.info(f"    infonce: {loss_dict['infonce']:.4f}")
        logger.info(f"    self: {loss_dict['self']:.4f}")
        logger.info(f"    positive: {loss_dict['positive']:.4f}")
        logger.info(f"    flops: {loss_dict['flops']:.4f}")
        logger.info(f"    min_act: {loss_dict['min_act']:.4f}")
        logger.info(f"  Semantic ratio: {avg_semantic_ratio:.2f}x")

    logger.info("\n" + "=" * 60)
    logger.info("Quick training completed successfully!")
    logger.info("=" * 60)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("V25 Quick Training Validation")
    logger.info("=" * 60)

    # Load tokenizer
    logger.info(f"\nLoading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")

    # Step 1: Verify IDF computation
    idf_weights = verify_idf_computation(tokenizer)

    # Step 2: Verify stopword masking
    stopword_mask = verify_stopword_masking(tokenizer)

    # Step 3: Verify loss creation
    loss_fn = verify_loss_creation(idf_weights, stopword_mask)

    if args.verify_only:
        logger.info("\nVerification complete (--verify-only mode)")
        return 0

    # Step 4: Run quick training
    run_quick_training(
        loss_fn=loss_fn,
        tokenizer=tokenizer,
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
