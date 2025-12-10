"""
Term Expander Module for Korean-English Neural Sparse Models.

This module provides functionality to expand Korean terms into
both Korean subwords and English translation terms using the
trained neural sparse model.

Example:
    >>> expander = TermExpander.from_checkpoint("outputs/v13_nouns/best_model.pt")
    >>> result = expander.expand("머신러닝")
    >>> print(result)
    ['머신러닝', '머신', '러', '닝', 'learning', 'machine']
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoTokenizer

from src.model.splade_model import create_splade_model


# Default noise tokens to filter out
DEFAULT_NOISE_TOKENS = {
    # Short meaningless subwords
    "con", "pre", "sub", "les", "pro", "com", "dis", "mis", "non", "uni",
    "tri", "bio", "geo", "neo", "eco", "iso", "hyp", "syn", "ant", "mid",
    # Semantically irrelevant words that may appear due to model bias
    "dark", "snow", "bright", "light", "white", "black", "red", "blue",
    "green", "yellow", "brown", "pink", "gray", "grey", "gold", "silver",
    # Common non-technical words
    "the", "and", "for", "with", "from", "that", "this", "have", "has",
    "was", "were", "been", "being", "are", "will", "would", "could",
    "should", "may", "might", "must", "can", "get", "got", "let", "set",
}


# Target words to reconstruct from subwords
# Maps (start_token, continuation_tokens) -> full_word
# BERT-style tokenization (uses ## prefix for continuations)
RECONSTRUCTABLE_WORDS_BERT = {
    # Problematic terms that BERT splits into subwords
    ("re", ("##com", "##mend")): "recommend",
    ("re", ("##com", "##mend", "##ation")): "recommendation",
    ("re", ("##com", "##mend", "##er")): "recommender",
    ("neu", ("##ral",)): "neural",
    ("rein", ("##forcement",)): "reinforcement",
    # Common AI/ML terms
    ("trans", ("##former",)): "transformer",
    ("transform", ("##er",)): "transformer",
    ("class", ("##ification",)): "classification",
    ("seg", ("##mentation",)): "segmentation",
    ("embed", ("##ding",)): "embedding",
    ("process", ("##ing",)): "processing",
    ("learn", ("##ing",)): "learning",
    ("train", ("##ing",)): "training",
    ("cluster", ("##ing",)): "clustering",
    ("rank", ("##ing",)): "ranking",
    ("re", ("##trieval",)): "retrieval",
    ("gen", ("##eration",)): "generation",
    ("re", ("##cognition",)): "recognition",
    ("detect", ("##ion",)): "detection",
    ("art", ("##ificial",)): "artificial",
    ("intelli", ("##gence",)): "intelligence",
}

# XLM-RoBERTa style tokenization (uses ▁ prefix for word starts)
RECONSTRUCTABLE_WORDS_ROBERTA = {
    # XLM-RoBERTa splits these words differently
    ("▁ne", ("ural",)): "neural",
    ("▁rein", ("force", "ment")): "reinforcement",
    ("▁recommend", ("ation",)): "recommendation",
    ("▁transform", ("er",)): "transformer",
    ("▁class", ("ification",)): "classification",
    ("▁segment", ("ation",)): "segmentation",
    ("▁embed", ("ding",)): "embedding",
    ("▁process", ("ing",)): "processing",
    ("▁learn", ("ing",)): "learning",
    ("▁train", ("ing",)): "training",
    ("▁cluster", ("ing",)): "clustering",
    ("▁rank", ("ing",)): "ranking",
    ("▁retriev", ("al",)): "retrieval",
    ("▁gener", ("ation",)): "generation",
    ("▁recogn", ("ition",)): "recognition",
    ("▁detect", ("ion",)): "detection",
    ("▁art", ("ificial",)): "artificial",
    ("▁intelli", ("gence",)): "intelligence",
}


@dataclass
class ExpansionResult:
    """Result of term expansion."""

    original: str
    subwords: list[str]
    korean_tokens: list[tuple[str, float]]
    english_tokens: list[tuple[str, float]]

    def to_list(
        self,
        include_original: bool = True,
        include_subwords: bool = True,
        max_english: int = 5,
    ) -> list[str]:
        """Convert to flat list of terms.

        Args:
            include_original: Include the original input term.
            include_subwords: Include Korean subword tokens.
            max_english: Maximum number of English terms to include.

        Returns:
            List of expanded terms.
        """
        terms = []

        if include_original:
            terms.append(self.original)

        if include_subwords:
            terms.extend(self.subwords)

        # Add English terms
        terms.extend([t for t, _ in self.english_tokens[:max_english]])

        return terms

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "original": self.original,
            "subwords": self.subwords,
            "korean": [{"token": t, "score": s} for t, s in self.korean_tokens],
            "english": [{"token": t, "score": s} for t, s in self.english_tokens],
        }


class TermExpander:
    """Expands Korean terms to include English translations.

    This class loads a trained neural sparse model and uses it to
    expand Korean input terms into:
    1. Original term
    2. Korean subword tokens
    3. English translation terms

    Attributes:
        model: The loaded SPLADE model.
        tokenizer: The BERT tokenizer.
        device: The device to run inference on.
        config: Model configuration from checkpoint.
        noise_tokens: Set of tokens to filter out.
        filter_noise: Whether to filter noise tokens.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        device: torch.device,
        config: dict,
        noise_tokens: Optional[set[str]] = None,
        filter_noise: bool = True,
    ):
        """Initialize TermExpander.

        Args:
            model: Loaded SPLADE model.
            tokenizer: Tokenizer (BERT or RoBERTa).
            device: Device for inference.
            config: Model configuration.
            noise_tokens: Custom set of noise tokens to filter.
            filter_noise: Whether to filter noise tokens (default True).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.filter_noise = filter_noise
        self.noise_tokens = noise_tokens if noise_tokens is not None else DEFAULT_NOISE_TOKENS
        self.model.eval()

        # Detect tokenizer type based on model name
        model_name = config.get("model_name", "")
        self.is_roberta = "roberta" in model_name.lower() or "xlm" in model_name.lower()
        self.reconstructable_words = (
            RECONSTRUCTABLE_WORDS_ROBERTA if self.is_roberta else RECONSTRUCTABLE_WORDS_BERT
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        noise_tokens: Optional[set[str]] = None,
        filter_noise: bool = True,
    ) -> "TermExpander":
        """Load TermExpander from a trained checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint file.
            device: Device to load model on. Auto-detected if None.
            noise_tokens: Custom set of noise tokens to filter.
            filter_noise: Whether to filter noise tokens (default True).

        Returns:
            Initialized TermExpander instance.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint["config"]

        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

        model = create_splade_model(
            model_name=config["model_name"],
            use_idf=False,
            use_expansion=True,
            expansion_mode="mlm",
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        return cls(model, tokenizer, device, config, noise_tokens, filter_noise)

    def _is_korean_char(self, c: str) -> bool:
        """Check if character is Korean."""
        return (
            "\uac00" <= c <= "\ud7a3"
            or "\u1100" <= c <= "\u11ff"
            or "\u3130" <= c <= "\u318f"
        )

    def _classify_token(self, token: str) -> str:
        """Classify token as korean, english, or other."""
        # Handle both BERT (##) and RoBERTa (▁) tokenization
        clean = token.replace("##", "").replace("▁", "")
        if not clean:
            return "other"

        has_korean = any(self._is_korean_char(c) for c in clean)
        has_english = any(c.isalpha() and c.isascii() for c in clean)

        if has_korean:
            return "korean"
        elif has_english:
            return "english"
        return "other"

    def _is_noise_token(self, token: str) -> bool:
        """Check if token should be filtered as noise.

        Args:
            token: Token to check.

        Returns:
            True if token is noise and should be filtered.
        """
        if not self.filter_noise:
            return False

        # Handle both BERT (##) and RoBERTa (▁) tokenization
        clean = token.replace("##", "").replace("▁", "").lower()
        return clean in self.noise_tokens

    def add_noise_tokens(self, tokens: set[str]) -> None:
        """Add tokens to the noise filter list.

        Args:
            tokens: Set of tokens to add to noise filter.
        """
        self.noise_tokens = self.noise_tokens | tokens

    def remove_noise_tokens(self, tokens: set[str]) -> None:
        """Remove tokens from the noise filter list.

        Args:
            tokens: Set of tokens to remove from noise filter.
        """
        self.noise_tokens = self.noise_tokens - tokens

    def _reconstruct_words_from_subwords(
        self,
        token_scores: dict[str, float],
        min_score: float = 3.5,
    ) -> tuple[list[tuple[str, float]], set[str]]:
        """Reconstruct full words from activated subword tokens.

        When BERT tokenizes words like "recommend" → ['re', '##com', '##mend'],
        this method checks if all subwords are activated and reconstructs the
        original word.

        Args:
            token_scores: Dictionary mapping tokens to their activation scores.
            min_score: Minimum score threshold for subword activation.

        Returns:
            Tuple of:
                - List of (word, score) tuples for successfully reconstructed words.
                - Set of component tokens used in reconstruction (to be filtered).
        """
        reconstructed = []
        used_tokens: set[str] = set()

        for (start_token, continuations), full_word in self.reconstructable_words.items():
            # Check if start token is activated
            if start_token not in token_scores:
                continue
            if token_scores[start_token] < min_score:
                continue

            # Check if all continuation tokens are activated
            all_continuations_active = True
            scores = [token_scores[start_token]]

            for cont in continuations:
                if cont not in token_scores or token_scores[cont] < min_score:
                    all_continuations_active = False
                    break
                scores.append(token_scores[cont])

            if all_continuations_active:
                # Use minimum score as the word's score
                word_score = min(scores)
                reconstructed.append((full_word, word_score))
                # Track used tokens
                used_tokens.add(start_token)
                used_tokens.update(continuations)

        # Sort by score descending
        reconstructed.sort(key=lambda x: -x[1])
        return reconstructed, used_tokens

    def expand(
        self,
        text: str,
        top_k: int = 100,
        min_score: float = 3.5,
        min_token_length: int = 4,
    ) -> ExpansionResult:
        """Expand a Korean term into Korean and English tokens.

        Args:
            text: Korean input text to expand.
            top_k: Number of top tokens to consider (default 100 for subword reconstruction).
            min_score: Minimum activation score threshold.
            min_token_length: Minimum token length (default 4, excludes short noise).

        Returns:
            ExpansionResult with original, subwords, and translations.
        """
        # Tokenize input
        encoding = self.tokenizer(
            text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Get sparse representation
        with torch.no_grad():
            sparse_rep, _ = self.model(
                encoding["input_ids"].to(self.device),
                encoding["attention_mask"].to(self.device),
            )

        sparse_rep = sparse_rep[0].cpu()
        top_values, top_indices = torch.topk(sparse_rep, k=top_k)

        # Get subwords from input (handle both BERT ## and RoBERTa ▁)
        input_tokens = self.tokenizer.tokenize(text)
        subwords = [t.replace("##", "").replace("▁", "") for t in input_tokens]

        # Build token_scores dict for subword reconstruction
        token_scores: dict[str, float] = {}
        for idx, score in zip(top_indices.tolist(), top_values.tolist()):
            token = self.tokenizer.convert_ids_to_tokens(idx)
            token_scores[token] = score

        # Reconstruct full words from subword tokens
        reconstructed_words, used_tokens = self._reconstruct_words_from_subwords(
            token_scores, min_score
        )

        # Collect Korean and English tokens
        korean_tokens = []
        english_tokens = []

        for idx, score in zip(top_indices.tolist(), top_values.tolist()):
            if score < min_score:
                continue

            token = self.tokenizer.convert_ids_to_tokens(idx)
            clean = token.replace("##", "").replace("▁", "")

            # Skip tokens that were used in word reconstruction
            if token in used_tokens:
                continue

            # Skip short tokens
            if len(clean) < min_token_length:
                continue

            # Skip noise tokens
            if self._is_noise_token(token):
                continue

            token_type = self._classify_token(token)

            if token_type == "korean":
                if clean not in subwords:
                    korean_tokens.append((clean, score))
            elif token_type == "english":
                # Only include full words, not subword pieces
                # BERT: continuation tokens start with ##
                # RoBERTa: word-starting tokens start with ▁, others are continuations
                is_subword = token.startswith("##") or (
                    self.is_roberta and not token.startswith("▁") and clean != token
                )
                if not is_subword:
                    english_tokens.append((clean, score))

        # Add reconstructed words to english_tokens (at the beginning for priority)
        # Deduplicate while preserving order
        seen = set()
        deduplicated = []
        for word, score in reconstructed_words + english_tokens:
            if word not in seen:
                seen.add(word)
                deduplicated.append((word, score))
        english_tokens = deduplicated

        return ExpansionResult(
            original=text,
            subwords=subwords,
            korean_tokens=korean_tokens,
            english_tokens=english_tokens,
        )

    def expand_batch(
        self,
        texts: list[str],
        top_k: int = 50,
        min_score: float = 3.5,
        min_token_length: int = 3,
    ) -> list[ExpansionResult]:
        """Expand multiple Korean terms.

        Args:
            texts: List of Korean input texts.
            top_k: Number of top tokens to consider.
            min_score: Minimum activation score threshold.
            min_token_length: Minimum token length.

        Returns:
            List of ExpansionResult objects.
        """
        return [
            self.expand(text, top_k, min_score, min_token_length)
            for text in texts
        ]

    def expand_to_list(
        self,
        text: str,
        include_original: bool = True,
        include_subwords: bool = True,
        max_english: int = 5,
        **kwargs,
    ) -> list[str]:
        """Expand a term and return as flat list.

        Convenience method that combines expand() and to_list().

        Args:
            text: Korean input text.
            include_original: Include original term in output.
            include_subwords: Include subword tokens in output.
            max_english: Maximum English terms to include.
            **kwargs: Additional arguments for expand().

        Returns:
            Flat list of expanded terms.

        Example:
            >>> expander.expand_to_list("머신러닝")
            ['머신러닝', '머신', '러', '닝', 'learning', 'machine']
        """
        result = self.expand(text, **kwargs)
        return result.to_list(
            include_original=include_original,
            include_subwords=include_subwords,
            max_english=max_english,
        )

    def get_sparse_vector(
        self,
        text: str,
        top_k: int = 100,
    ) -> dict[str, float]:
        """Get sparse vector representation for OpenSearch.

        Returns a dictionary mapping tokens to their activation scores,
        suitable for use with OpenSearch neural sparse queries.

        Args:
            text: Input text.
            top_k: Number of top tokens to include.

        Returns:
            Dictionary of token -> score mappings.
        """
        encoding = self.tokenizer(
            text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            sparse_rep, _ = self.model(
                encoding["input_ids"].to(self.device),
                encoding["attention_mask"].to(self.device),
            )

        sparse_rep = sparse_rep[0].cpu()
        top_values, top_indices = torch.topk(sparse_rep, k=top_k)

        sparse_vector = {}
        for idx, score in zip(top_indices.tolist(), top_values.tolist()):
            if score > 0:
                token = self.tokenizer.convert_ids_to_tokens(idx)
                sparse_vector[token] = float(score)

        return sparse_vector
