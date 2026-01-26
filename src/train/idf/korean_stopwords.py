"""
Korean stopword handling for XLM-RoBERTa SPLADE training.

Provides lists of Korean particles, endings, and common words that
should have reduced weights in sparse representations. These tokens
are grammatically important but not semantically discriminative.
"""

import logging
from typing import List, Optional, Set

import torch
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


# Korean postpositional particles (조사)
KOREAN_PARTICLES = [
    # Subject markers
    "이", "가", "께서",
    # Object markers
    "을", "를",
    # Topic markers
    "은", "는",
    # Possessive/attributive
    "의",
    # Locative particles
    "에", "에서", "에게", "한테", "께",
    # Direction/goal
    "로", "으로",
    # Instrumental/means
    "로써", "으로써",
    # Comitative (with)
    "와", "과", "랑", "이랑",
    # Comparative
    "보다", "처럼", "같이", "만큼",
    # Plural marker
    "들",
    # Only/just
    "만", "뿐",
    # Also/too
    "도",
    # From
    "부터", "에서부터",
    # Until
    "까지",
    # Or
    "나", "이나",
    # Even
    "조차", "마저",
]

# Korean verb/adjective endings (어미)
KOREAN_ENDINGS = [
    # Declarative endings
    "다", "습니다", "ㅂ니다", "니다", "입니다",
    "요", "어요", "아요", "죠", "지요",
    "야", "이야", "야",
    # Interrogative endings
    "까", "습니까", "ㅂ니까", "니까",
    "나요", "을까요", "ㄹ까요",
    # Imperative endings
    "세요", "십시오", "어라", "아라",
    # Connective endings
    "고", "서", "며", "면서",
    "지만", "는데", "ㄴ데", "은데",
    "니까", "으니까",
    "면", "으면",
    "려고", "으려고",
    # Nominalization
    "는것", "은것", "ㄴ것", "기", "음",
    # Modifiers
    "는", "은", "ㄴ", "을", "ㄹ",
]

# Common Korean function words
KOREAN_FUNCTION_WORDS = [
    # Copula
    "이다", "아니다",
    # Auxiliary verbs
    "있다", "없다", "하다", "되다",
    # Demonstratives
    "이", "그", "저",
    # Pronouns
    "나", "너", "우리", "저희",
    # Common particles/adverbs
    "매우", "아주", "정말", "진짜",
    "좀", "많이", "조금",
    # Conjunctions
    "그리고", "그러나", "하지만", "그래서",
    # Question words (less discriminative in search)
    "무엇", "뭐", "어디", "언제", "왜", "어떻게",
]

# V26: Additional high-frequency stopwords observed in V25 analysis
# These verb endings and functional words dominated top activations
KOREAN_ADDITIONAL_STOPWORDS = [
    # Declarative verb endings (formal/informal)
    "있습니다", "합니다", "입니다", "됩니다", "했습니다",
    "있어요", "해요", "이에요", "되요", "했어요",
    "있어", "해", "이야", "돼", "했어",
    # Nominalizing patterns
    "것입니다", "것이다", "것은", "것을", "것이",
    "수", "때", "것", "데",
    # Connective patterns
    "그런데", "따라서", "그러므로", "그래서", "하지만",
    "그러나", "그리고", "또한", "또는", "및",
    # Auxiliary expressions
    "있는", "하는", "되는", "하게", "되게",
    "할", "될", "있을", "없을",
    # Common adverbs that don't carry semantic meaning
    "더", "가장", "매우", "아주", "잘",
    "바로", "이미", "아직", "다시", "모두",
    # Modal/aspectual markers
    "수있", "수없", "겠", "어야", "어도",
    # Common predicative endings
    "한다", "한", "하고", "해서", "하면",
]

# Combined stopword list
KOREAN_STOPWORDS: List[str] = (
    KOREAN_PARTICLES + KOREAN_ENDINGS + KOREAN_FUNCTION_WORDS
)

# V26 Extended stopword list with additional high-frequency terms
KOREAN_STOPWORDS_V26: List[str] = (
    KOREAN_PARTICLES + KOREAN_ENDINGS + KOREAN_FUNCTION_WORDS
    + KOREAN_ADDITIONAL_STOPWORDS
)


def get_special_token_ids_only(tokenizer: PreTrainedTokenizer) -> Set[int]:
    """
    Get ONLY special token IDs (not stopwords).

    This is used for V26 to exclude special tokens from IDF normalization
    without conflating them with linguistic stopwords.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        Set of special token IDs
    """
    return _get_special_token_ids(tokenizer)


def get_korean_stopword_ids(
    tokenizer: PreTrainedTokenizer,
    include_subwords: bool = True,
    additional_stopwords: Optional[List[str]] = None,
) -> Set[int]:
    """
    Get token IDs for Korean stopwords.

    For XLM-RoBERTa's SentencePiece tokenizer, stopwords may be tokenized
    with leading space markers (▁) or as subwords. This function handles
    both cases.

    Args:
        tokenizer: HuggingFace tokenizer (XLM-RoBERTa)
        include_subwords: Include subword variants (with/without ▁)
        additional_stopwords: Extra stopwords to include

    Returns:
        Set of token IDs for stopwords
    """
    stopword_ids: Set[int] = set()

    # Combine default and additional stopwords
    stopwords = KOREAN_STOPWORDS.copy()
    if additional_stopwords:
        stopwords.extend(additional_stopwords)

    for word in stopwords:
        # Direct encoding
        tokens = tokenizer.encode(word, add_special_tokens=False)
        stopword_ids.update(tokens)

        if include_subwords:
            # Try with leading space (SentencePiece format)
            space_word = f" {word}"
            space_tokens = tokenizer.encode(space_word, add_special_tokens=False)
            stopword_ids.update(space_tokens)

            # Also try to find the word directly in vocabulary
            # XLM-RoBERTa uses ▁ as space marker
            sp_word = f"▁{word}"
            if sp_word in tokenizer.get_vocab():
                stopword_ids.add(tokenizer.convert_tokens_to_ids(sp_word))

            # Try without marker too
            if word in tokenizer.get_vocab():
                stopword_ids.add(tokenizer.convert_tokens_to_ids(word))

    # Add special tokens (always mask these)
    special_ids = _get_special_token_ids(tokenizer)
    stopword_ids.update(special_ids)

    logger.info(f"Identified {len(stopword_ids)} stopword/special token IDs")
    return stopword_ids


def _get_special_token_ids(tokenizer: PreTrainedTokenizer) -> Set[int]:
    """
    Get IDs for special tokens that should always be masked.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        Set of special token IDs
    """
    special_ids: Set[int] = set()

    # Standard special tokens
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

    # XLM-RoBERTa specific: IDs 0-6 are typically special
    # <s>, </s>, <pad>, <unk>, , ., (space marker)
    for i in range(7):
        special_ids.add(i)

    # Common punctuation in early vocab positions
    punct_tokens = [".", ",", "!", "?", ":", ";", "-", "'", '"', "(", ")", "[", "]"]
    for punct in punct_tokens:
        if punct in tokenizer.get_vocab():
            special_ids.add(tokenizer.convert_tokens_to_ids(punct))
        # With space marker
        sp_punct = f"▁{punct}"
        if sp_punct in tokenizer.get_vocab():
            special_ids.add(tokenizer.convert_tokens_to_ids(sp_punct))

    return special_ids


def create_stopword_mask(
    tokenizer: PreTrainedTokenizer,
    include_subwords: bool = True,
    additional_stopwords: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a binary mask for stopword tokens.

    Returns a tensor where:
    - 1.0 = regular token (keep)
    - 0.0 = stopword/special token (mask)

    Args:
        tokenizer: HuggingFace tokenizer
        include_subwords: Include subword variants
        additional_stopwords: Extra stopwords to include
        device: Target device for tensor

    Returns:
        Binary mask tensor [vocab_size]
    """
    vocab_size = tokenizer.vocab_size
    mask = torch.ones(vocab_size, device=device)

    stopword_ids = get_korean_stopword_ids(
        tokenizer,
        include_subwords=include_subwords,
        additional_stopwords=additional_stopwords,
    )

    for token_id in stopword_ids:
        if 0 <= token_id < vocab_size:
            mask[token_id] = 0.0

    masked_count = (mask == 0).sum().item()
    logger.info(f"Created stopword mask: {masked_count} tokens masked")

    return mask


def get_korean_stopword_ids_v26(
    tokenizer: PreTrainedTokenizer,
    include_subwords: bool = True,
) -> Set[int]:
    """
    Get token IDs for V26 extended Korean stopwords.

    Uses the expanded KOREAN_STOPWORDS_V26 list which includes
    additional high-frequency verb endings and functional words
    observed in V25 analysis.

    Args:
        tokenizer: HuggingFace tokenizer (XLM-RoBERTa)
        include_subwords: Include subword variants (with/without ▁)

    Returns:
        Set of token IDs for stopwords (NOT including special tokens)
    """
    stopword_ids: Set[int] = set()

    for word in KOREAN_STOPWORDS_V26:
        # Direct encoding
        tokens = tokenizer.encode(word, add_special_tokens=False)
        stopword_ids.update(tokens)

        if include_subwords:
            # Try with leading space (SentencePiece format)
            space_word = f" {word}"
            space_tokens = tokenizer.encode(space_word, add_special_tokens=False)
            stopword_ids.update(space_tokens)

            # Also try to find the word directly in vocabulary
            sp_word = f"▁{word}"
            if sp_word in tokenizer.get_vocab():
                stopword_ids.add(tokenizer.convert_tokens_to_ids(sp_word))

            if word in tokenizer.get_vocab():
                stopword_ids.add(tokenizer.convert_tokens_to_ids(word))

    logger.info(f"V26: Identified {len(stopword_ids)} stopword token IDs")
    return stopword_ids


def create_stopword_mask_v26(
    tokenizer: PreTrainedTokenizer,
    include_subwords: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a binary mask for V26 extended stopword tokens.

    Uses the expanded stopword list but does NOT include special tokens
    in the mask (they are handled separately via special_penalty).

    Returns a tensor where:
    - 1.0 = regular token (keep)
    - 0.0 = stopword token (mask)

    Args:
        tokenizer: HuggingFace tokenizer
        include_subwords: Include subword variants
        device: Target device for tensor

    Returns:
        Binary mask tensor [vocab_size]
    """
    vocab_size = tokenizer.vocab_size
    mask = torch.ones(vocab_size, device=device)

    # Get V26 stopword IDs (without special tokens)
    stopword_ids = get_korean_stopword_ids_v26(
        tokenizer,
        include_subwords=include_subwords,
    )

    for token_id in stopword_ids:
        if 0 <= token_id < vocab_size:
            mask[token_id] = 0.0

    masked_count = (mask == 0).sum().item()
    logger.info(f"V26: Created stopword mask: {masked_count} tokens masked")

    return mask


def create_stopword_penalty_weights(
    tokenizer: PreTrainedTokenizer,
    stopword_penalty: float = 5.0,
    special_penalty: float = 10.0,
    include_subwords: bool = True,
    additional_stopwords: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create penalty weights for stopword tokens.

    Returns a tensor where:
    - 1.0 = regular token (normal penalty)
    - stopword_penalty = Korean stopword (higher penalty)
    - special_penalty = special token (highest penalty)

    This provides soft penalization during training while
    create_stopword_mask provides hard masking at inference.

    Args:
        tokenizer: HuggingFace tokenizer
        stopword_penalty: Penalty multiplier for Korean stopwords
        special_penalty: Penalty multiplier for special tokens
        include_subwords: Include subword variants
        additional_stopwords: Extra stopwords to include
        device: Target device for tensor

    Returns:
        Penalty weight tensor [vocab_size]
    """
    vocab_size = tokenizer.vocab_size
    weights = torch.ones(vocab_size, device=device)

    # Get special tokens (highest penalty)
    special_ids = _get_special_token_ids(tokenizer)
    for token_id in special_ids:
        if 0 <= token_id < vocab_size:
            weights[token_id] = special_penalty

    # Get stopword IDs (medium penalty, excluding special)
    all_stopword_ids = get_korean_stopword_ids(
        tokenizer,
        include_subwords=include_subwords,
        additional_stopwords=additional_stopwords,
    )

    for token_id in all_stopword_ids:
        if 0 <= token_id < vocab_size and token_id not in special_ids:
            weights[token_id] = stopword_penalty

    logger.info(
        f"Created penalty weights: {len(special_ids)} special tokens, "
        f"{len(all_stopword_ids) - len(special_ids)} stopwords"
    )

    return weights
