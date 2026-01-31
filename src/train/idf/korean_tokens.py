"""
Korean and English token identification for XLM-RoBERTa vocabulary.

Used by V28 for language-aware token filtering:
- Korean tokens: preserved (no penalty)
- Non-Korean tokens: suppressed (high penalty)

This addresses multilingual token leakage where non-Korean tokens
inappropriately activate in Korean-only sparse representations.
"""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import torch
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


# Korean Unicode ranges
KOREAN_RANGES = [
    (0xAC00, 0xD7A3),   # Hangul Syllables
    (0x1100, 0x11FF),   # Hangul Jamo
    (0x3130, 0x318F),   # Hangul Compatibility Jamo
    (0xA960, 0xA97F),   # Hangul Jamo Extended-A
    (0xD7B0, 0xD7FF),   # Hangul Jamo Extended-B
]

# English/Latin Unicode ranges
ENGLISH_RANGES = [
    (0x0041, 0x005A),   # A-Z
    (0x0061, 0x007A),   # a-z
    (0x00C0, 0x00FF),   # Latin Extended-A (accented)
]


def is_korean_char(char: str) -> bool:
    """Check if a single character is Korean."""
    if len(char) != 1:
        return False
    code = ord(char)
    for start, end in KOREAN_RANGES:
        if start <= code <= end:
            return True
    return False


def is_english_char(char: str) -> bool:
    """Check if a single character is English/Latin."""
    if len(char) != 1:
        return False
    code = ord(char)
    for start, end in ENGLISH_RANGES:
        if start <= code <= end:
            return True
    return False


def get_token_language(token: str) -> str:
    """
    Determine the primary language of a token.

    Args:
        token: Token string (may include SentencePiece marker ▁)

    Returns:
        "korean", "english", "mixed", "punct", "number", or "other"
    """
    # Remove SentencePiece marker
    clean_token = token.replace("▁", "").strip()

    if not clean_token:
        return "other"

    # Count character types
    korean_count = sum(1 for c in clean_token if is_korean_char(c))
    english_count = sum(1 for c in clean_token if is_english_char(c))
    digit_count = sum(1 for c in clean_token if c.isdigit())
    punct_count = sum(1 for c in clean_token if unicodedata.category(c).startswith('P'))

    total_alpha = korean_count + english_count

    # Pure punctuation
    if punct_count == len(clean_token):
        return "punct"

    # Pure numbers
    if digit_count == len(clean_token):
        return "number"

    # No alphabetic characters
    if total_alpha == 0:
        return "other"

    # Determine primary language
    if korean_count > 0 and english_count == 0:
        return "korean"
    elif english_count > 0 and korean_count == 0:
        return "english"
    elif korean_count > english_count:
        return "korean"  # Majority Korean
    elif english_count > korean_count:
        return "english"  # Majority English
    else:
        return "mixed"


def build_korean_token_ids(
    tokenizer: PreTrainedTokenizer,
    include_mixed: bool = True,
    include_numbers: bool = True,
    include_punct: bool = True,
) -> Set[int]:
    """
    Build set of Korean token IDs from XLM-RoBERTa vocabulary.

    Args:
        tokenizer: XLM-RoBERTa tokenizer
        include_mixed: Include tokens with both Korean and other chars
        include_numbers: Include pure number tokens
        include_punct: Include punctuation tokens

    Returns:
        Set of token IDs considered "Korean-safe"
    """
    korean_ids: Set[int] = set()
    vocab = tokenizer.get_vocab()

    for token, token_id in vocab.items():
        lang = get_token_language(token)

        if lang == "korean":
            korean_ids.add(token_id)
        elif lang == "mixed" and include_mixed:
            korean_ids.add(token_id)
        elif lang == "number" and include_numbers:
            korean_ids.add(token_id)
        elif lang == "punct" and include_punct:
            korean_ids.add(token_id)

    # Always include special tokens (they're handled separately)
    special_ids = _get_special_token_ids(tokenizer)
    korean_ids.update(special_ids)

    logger.info(f"Built Korean token set: {len(korean_ids)} tokens")
    return korean_ids


def build_non_korean_token_ids(
    tokenizer: PreTrainedTokenizer,
    exclude_numbers: bool = True,
    exclude_punct: bool = True,
) -> Set[int]:
    """
    Build set of non-Korean token IDs from XLM-RoBERTa vocabulary.

    These tokens should be suppressed in Korean sparse representations.

    Args:
        tokenizer: XLM-RoBERTa tokenizer
        exclude_numbers: Don't include numbers in non-Korean set
        exclude_punct: Don't include punctuation in non-Korean set

    Returns:
        Set of token IDs to suppress (non-Korean)
    """
    non_korean_ids: Set[int] = set()
    vocab = tokenizer.get_vocab()

    for token, token_id in vocab.items():
        lang = get_token_language(token)

        if lang == "english":
            non_korean_ids.add(token_id)
        elif lang == "other":
            non_korean_ids.add(token_id)
        elif lang == "number" and not exclude_numbers:
            non_korean_ids.add(token_id)
        elif lang == "punct" and not exclude_punct:
            non_korean_ids.add(token_id)

    # Never include special tokens
    special_ids = _get_special_token_ids(tokenizer)
    non_korean_ids -= special_ids

    logger.info(f"Built non-Korean token set: {len(non_korean_ids)} tokens")
    return non_korean_ids


def build_korean_english_token_ids(
    tokenizer: PreTrainedTokenizer,
) -> Tuple[Set[int], Set[int]]:
    """
    Build both Korean and English token ID sets.

    Args:
        tokenizer: XLM-RoBERTa tokenizer

    Returns:
        Tuple of (korean_ids, english_ids)
    """
    korean_ids: Set[int] = set()
    english_ids: Set[int] = set()
    vocab = tokenizer.get_vocab()

    for token, token_id in vocab.items():
        lang = get_token_language(token)
        if lang == "korean":
            korean_ids.add(token_id)
        elif lang == "english":
            english_ids.add(token_id)

    logger.info(f"Korean tokens: {len(korean_ids)}, English tokens: {len(english_ids)}")
    return korean_ids, english_ids


def _get_special_token_ids(tokenizer: PreTrainedTokenizer) -> Set[int]:
    """Get IDs for special tokens."""
    special_ids: Set[int] = set()

    if tokenizer.pad_token_id is not None:
        special_ids.add(tokenizer.pad_token_id)
    if tokenizer.bos_token_id is not None:
        special_ids.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        special_ids.add(tokenizer.eos_token_id)
    if tokenizer.unk_token_id is not None:
        special_ids.add(tokenizer.unk_token_id)
    if tokenizer.cls_token_id is not None:
        special_ids.add(tokenizer.cls_token_id)
    if tokenizer.sep_token_id is not None:
        special_ids.add(tokenizer.sep_token_id)
    if tokenizer.mask_token_id is not None:
        special_ids.add(tokenizer.mask_token_id)

    # XLM-RoBERTa: IDs 0-6 are typically special
    for i in range(7):
        special_ids.add(i)

    return special_ids


def create_language_penalty_mask(
    tokenizer: PreTrainedTokenizer,
    non_korean_penalty: float = 100.0,
    korean_penalty: float = 0.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create penalty mask for language filtering.

    Returns a tensor where:
    - Korean tokens have korean_penalty (default 0.0 = no penalty)
    - Non-Korean tokens have non_korean_penalty (default 100.0 = suppress)
    - Special tokens have 0.0 (handled separately)
    - Punctuation/numbers have 0.0 (neutral)

    Args:
        tokenizer: XLM-RoBERTa tokenizer
        non_korean_penalty: Penalty for non-Korean tokens
        korean_penalty: Penalty for Korean tokens
        device: Target device

    Returns:
        Penalty mask [vocab_size]
    """
    vocab_size = tokenizer.vocab_size
    mask = torch.zeros(vocab_size, device=device)  # Default: no penalty

    vocab = tokenizer.get_vocab()

    for token, token_id in vocab.items():
        lang = get_token_language(token)

        if lang == "korean":
            mask[token_id] = korean_penalty
        elif lang == "english" or lang == "other":
            mask[token_id] = non_korean_penalty
        # mixed, number, punct: keep at 0 (neutral)

    # Ensure special tokens have no penalty
    special_ids = _get_special_token_ids(tokenizer)
    for tid in special_ids:
        if 0 <= tid < vocab_size:
            mask[tid] = 0.0

    non_zero = (mask > 0).sum().item()
    logger.info(f"Language penalty mask: {non_zero} tokens penalized")

    return mask


def save_korean_token_ids(
    korean_ids: Set[int],
    output_path: str,
) -> None:
    """
    Save Korean token IDs to JSON file.

    Args:
        korean_ids: Set of Korean token IDs
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(sorted(korean_ids), f)

    logger.info(f"Saved {len(korean_ids)} Korean token IDs to {output_path}")


def load_korean_token_ids(input_path: str) -> Set[int]:
    """
    Load Korean token IDs from JSON file.

    Args:
        input_path: Input file path

    Returns:
        Set of Korean token IDs
    """
    with open(input_path) as f:
        ids = json.load(f)

    return set(ids)


def load_or_compute_korean_tokens(
    cache_path: str,
    tokenizer: PreTrainedTokenizer,
    recompute: bool = False,
) -> Set[int]:
    """
    Load Korean token IDs from cache or compute from tokenizer.

    Args:
        cache_path: Path to cache file
        tokenizer: XLM-RoBERTa tokenizer
        recompute: Force recomputation

    Returns:
        Set of Korean token IDs
    """
    cache_path = Path(cache_path)

    if cache_path.exists() and not recompute:
        logger.info(f"Loading Korean tokens from cache: {cache_path}")
        return load_korean_token_ids(str(cache_path))

    logger.info("Computing Korean token IDs from vocabulary...")
    korean_ids = build_korean_token_ids(tokenizer)
    save_korean_token_ids(korean_ids, str(cache_path))

    return korean_ids


def analyze_token_language_distribution(
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, int]:
    """
    Analyze language distribution in tokenizer vocabulary.

    Args:
        tokenizer: XLM-RoBERTa tokenizer

    Returns:
        Dictionary of {language: count}
    """
    distribution: Dict[str, int] = {
        "korean": 0,
        "english": 0,
        "mixed": 0,
        "number": 0,
        "punct": 0,
        "other": 0,
    }

    vocab = tokenizer.get_vocab()
    for token in vocab:
        lang = get_token_language(token)
        distribution[lang] += 1

    total = sum(distribution.values())
    logger.info("Vocabulary language distribution:")
    for lang, count in distribution.items():
        pct = 100 * count / total
        logger.info(f"  {lang}: {count:,} ({pct:.1f}%)")

    return distribution
