#!/usr/bin/env python3
"""
Test SPLADE model inference with Korean samples.

This script loads a trained SPLADE model and performs inference on Korean text samples,
displaying the top-k weighted tokens for each input.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from src.model.splade_model import create_splade_model


# Korean test samples
KOREAN_TEST_SAMPLES = [
    {
        "title": "인공지능",
        "text": "인공지능(人工知能, 영어: artificial intelligence, AI)은 인간의 학습능력, 추론능력, 지각능력을 인공적으로 구현한 컴퓨터 시스템이다."
    },
    {
        "title": "딥러닝",
        "text": "딥러닝은 여러 비선형 변환기법의 조합을 통해 높은 수준의 추상화를 시도하는 기계학습 알고리즘의 집합이다."
    },
    {
        "title": "자연어 처리",
        "text": "자연어 처리는 인간이 사용하는 자연어를 컴퓨터가 이해하고 처리할 수 있도록 하는 인공지능의 한 분야이다."
    },
    {
        "title": "검색 엔진",
        "text": "검색 엔진은 인터넷에서 정보를 찾아주는 프로그램 또는 시스템을 말한다. 사용자가 입력한 검색어를 기반으로 관련된 웹페이지를 찾아서 보여준다."
    },
    {
        "title": "기계학습",
        "text": "기계학습은 경험을 통해 자동으로 개선하는 컴퓨터 알고리즘의 연구이다. 인공지능의 한 분야로 간주된다."
    },
]


class SPLADEInference:
    """SPLADE model inference wrapper."""

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "bert-base-multilingual-cased",
        device: str = None,
    ):
        """
        Initialize SPLADE inference.

        Args:
            checkpoint_path: Path to model checkpoint
            model_name: Base model name
            device: Device to run inference on
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load tokenizer
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        print(f"Loading model from: {checkpoint_path}")
        self.model = self._load_model(checkpoint_path, model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        print("✓ Model loaded successfully")

    def _load_model(self, checkpoint_path: str, model_name: str):
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        # Create model
        model = create_splade_model(
            model_name=model_name,
            use_idf=False,
            dropout=0.1,
        )

        # Load checkpoint
        if (checkpoint_path / 'checkpoint.pt').exists():
            checkpoint = torch.load(
                checkpoint_path / 'checkpoint.pt',
                map_location=self.device
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"  Global step: {checkpoint.get('global_step', 'unknown')}")
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        return model

    def merge_subword_tokens_from_sequence(
        self,
        text: str,
        sparse_repr: torch.Tensor,
        top_k: int = 50,
    ) -> Dict[str, float]:
        """
        Merge WordPiece subword tokens based on input sequence.

        This method tokenizes the input text and merges consecutive
        subword tokens (starting with ##) into complete words.

        Args:
            text: Input text
            sparse_repr: Sparse representation tensor [vocab_size]
            top_k: Number of top tokens to return

        Returns:
            Dictionary of merged tokens and their weights
        """
        # Tokenize input to get the actual token sequence
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Get weights for tokens in the sequence
        token_to_weight = {}
        merged_tokens = []

        current_word = ""
        current_weight = 0.0

        for token, token_id in zip(tokens, token_ids):
            weight = sparse_repr[token_id].item()

            if token.startswith("##"):
                # Continuation of previous word
                current_word += token[2:]  # Remove ##
                current_weight = max(current_weight, weight)  # Use max weight
            else:
                # New word - save previous if exists
                if current_word and current_weight > 0:
                    if current_word not in token_to_weight:
                        token_to_weight[current_word] = current_weight
                    else:
                        token_to_weight[current_word] = max(
                            token_to_weight[current_word],
                            current_weight
                        )

                # Start new word
                current_word = token
                current_weight = weight

        # Add last word
        if current_word and current_weight > 0:
            if current_word not in token_to_weight:
                token_to_weight[current_word] = current_weight
            else:
                token_to_weight[current_word] = max(
                    token_to_weight[current_word],
                    current_weight
                )

        # Also include high-weight tokens not in sequence (for expansion)
        top_vocab_tokens = self.model.get_top_k_tokens(
            sparse_repr,
            self.tokenizer,
            k=top_k * 2,
        )

        for token, weight in top_vocab_tokens.items():
            token = token.strip()
            # Skip subword tokens when adding from vocab
            if not token.startswith("##"):
                if token not in token_to_weight:
                    token_to_weight[token] = weight
                else:
                    token_to_weight[token] = max(token_to_weight[token], weight)

        # Sort by weight and return top-k
        sorted_tokens = sorted(
            token_to_weight.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return dict(sorted_tokens)

    @torch.no_grad()
    def encode(
        self,
        text: str,
        max_length: int = 512,
        merge_subwords: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Encode text to sparse representation.

        Args:
            text: Input text
            max_length: Maximum sequence length
            merge_subwords: Whether to merge WordPiece subword tokens

        Returns:
            sparse_repr: Sparse representation tensor
            top_tokens: Dictionary of top tokens and their weights
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Encode
        sparse_repr, _ = self.model(
            inputs['input_ids'],
            inputs['attention_mask'],
        )

        # Get tokens (merged or raw)
        if merge_subwords:
            top_tokens = self.merge_subword_tokens_from_sequence(
                text,
                sparse_repr[0],
                top_k=50,
            )
        else:
            top_tokens = self.model.get_top_k_tokens(
                sparse_repr[0],
                self.tokenizer,
                k=50,
            )

        return sparse_repr[0], top_tokens

    def print_top_tokens(
        self,
        title: str,
        text: str,
        top_tokens: Dict[str, float],
        top_k: int = 20,
    ):
        """
        Print top-k tokens in a formatted way.

        Args:
            title: Document title
            text: Document text
            top_tokens: Dictionary of tokens and weights
            top_k: Number of top tokens to display
        """
        print("\n" + "=" * 80)
        print(f"Title: {title}")
        print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
        print("-" * 80)
        print(f"Top-{top_k} weighted tokens:")
        print("-" * 80)

        # Sort by weight and take top-k
        sorted_tokens = sorted(
            top_tokens.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Print in a nice format
        for i, (token, weight) in enumerate(sorted_tokens, 1):
            # Clean token display
            token_display = token.strip()
            if not token_display:
                token_display = "[SPACE]"

            print(f"{i:2d}. {token_display:20s} {weight:8.4f}")

        print("=" * 80)


def main():
    """Main inference testing function."""
    parser = argparse.ArgumentParser(
        description='Test SPLADE model inference with Korean samples'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/baseline_dgx/best_model',
        help='Path to model checkpoint directory'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='bert-base-multilingual-cased',
        help='Base model name'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        help='Number of top tokens to display'
    )
    parser.add_argument(
        '--custom-text',
        type=str,
        default=None,
        help='Custom text to test (optional)'
    )
    parser.add_argument(
        '--save-results',
        type=str,
        default=None,
        help='Path to save results JSON (optional)'
    )
    parser.add_argument(
        '--no-merge-subwords',
        action='store_true',
        help='Disable subword token merging (show raw WordPiece tokens)'
    )

    args = parser.parse_args()

    # Initialize inference
    print("=" * 80)
    print("SPLADE Model Inference Test - Korean Samples")
    print("=" * 80)

    inference = SPLADEInference(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
    )

    # Test samples
    test_samples = KOREAN_TEST_SAMPLES.copy()

    # Add custom text if provided
    if args.custom_text:
        test_samples.append({
            "title": "Custom Input",
            "text": args.custom_text
        })

    # Run inference
    results = []

    print(f"\nTesting {len(test_samples)} samples...")
    print()

    for sample in tqdm(test_samples, desc="Processing"):
        # Encode
        sparse_repr, top_tokens = inference.encode(
            sample['text'],
            merge_subwords=not args.no_merge_subwords
        )

        # Print results
        inference.print_top_tokens(
            title=sample['title'],
            text=sample['text'],
            top_tokens=top_tokens,
            top_k=args.top_k,
        )

        # Store results
        results.append({
            'title': sample['title'],
            'text': sample['text'],
            'top_tokens': [
                {'token': token, 'weight': float(weight)}
                for token, weight in sorted(
                    top_tokens.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:args.top_k]
            ],
            'num_nonzero_tokens': int((sparse_repr > 0).sum().item()),
            'sparsity': float((sparse_repr == 0).sum().item() / sparse_repr.shape[0]),
        })

    # Save results if requested
    if args.save_results:
        output_path = Path(args.save_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Total samples: {len(results)}")
    print(f"Average non-zero tokens: {sum(r['num_nonzero_tokens'] for r in results) / len(results):.1f}")
    print(f"Average sparsity: {sum(r['sparsity'] for r in results) / len(results):.2%}")
    print("=" * 80)


if __name__ == '__main__':
    main()
