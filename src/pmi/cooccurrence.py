"""Co-occurrence matrix construction for PMI calculation.

This module implements efficient co-occurrence matrix building from text corpora
using sparse matrix representations for memory efficiency with large vocabularies.

Window Types:
- SENTENCE: Terms co-occur if they appear in the same sentence
- PARAGRAPH: Terms co-occur if they appear in the same paragraph
- SLIDING: Terms co-occur within a sliding window of fixed size

Performance Considerations:
- Uses scipy.sparse.lil_matrix for incremental construction
- Converts to csr_matrix for efficient arithmetic operations
- Supports parallel processing for large corpora
"""

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
from scipy import sparse
from tqdm import tqdm


class WindowType(Enum):
    """Co-occurrence window type definitions."""

    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SLIDING = "sliding"


@dataclass
class CooccurrenceConfig:
    """Configuration for co-occurrence matrix construction.

    Attributes:
        window_type: Type of co-occurrence window (sentence, paragraph, sliding)
        window_size: Size of sliding window (only used for SLIDING type)
        min_term_freq: Minimum frequency threshold for terms to be included
        max_vocab_size: Maximum vocabulary size (keeps most frequent terms)
        symmetric: Whether to make co-occurrence matrix symmetric
        normalize: Whether to normalize counts by window size
    """

    window_type: WindowType = WindowType.SENTENCE
    window_size: int = 10
    min_term_freq: int = 5
    max_vocab_size: int = 120000
    symmetric: bool = True
    normalize: bool = False


@dataclass
class CooccurrenceStats:
    """Statistics from co-occurrence matrix construction.

    Attributes:
        vocab_size: Number of terms in vocabulary
        total_documents: Total number of documents processed
        total_windows: Total number of co-occurrence windows
        total_cooccurrences: Total number of co-occurrence pairs
        sparsity: Sparsity ratio of the matrix
    """

    vocab_size: int = 0
    total_documents: int = 0
    total_windows: int = 0
    total_cooccurrences: int = 0
    sparsity: float = 0.0


class CooccurrenceMatrixBuilder:
    """Builds sparse co-occurrence matrices from text corpora.

    This class efficiently constructs co-occurrence matrices using scipy sparse
    matrices, suitable for vocabularies up to 120K terms with ~640K documents.

    Memory Estimation:
    - 120K vocab, 10% density: ~1.4GB (using float32)
    - With actual sparsity (~0.1%): ~140MB

    Example:
        >>> builder = CooccurrenceMatrixBuilder(config)
        >>> builder.fit(texts)
        >>> cooc_matrix = builder.get_cooccurrence_matrix()
        >>> term_freq = builder.get_term_frequencies()
    """

    def __init__(self, config: Optional[CooccurrenceConfig] = None):
        """Initialize the co-occurrence matrix builder.

        Args:
            config: Configuration for matrix construction. Uses defaults if None.
        """
        self.config = config or CooccurrenceConfig()
        self._vocab: Dict[str, int] = {}
        self._reverse_vocab: Dict[int, str] = {}
        self._term_freq: Counter = Counter()
        self._cooccurrence_matrix: Optional[sparse.csr_matrix] = None
        self._doc_freq: Counter = Counter()
        self._stats = CooccurrenceStats()

    def fit(
        self,
        documents: List[str],
        tokenizer: Optional[callable] = None,
        show_progress: bool = True,
    ) -> "CooccurrenceMatrixBuilder":
        """Build co-occurrence matrix from documents.

        Args:
            documents: List of text documents
            tokenizer: Optional tokenizer function. If None, uses simple whitespace.
            show_progress: Whether to show progress bar

        Returns:
            Self for method chaining
        """
        # Step 1: Build vocabulary
        self._build_vocabulary(documents, tokenizer, show_progress)

        # Step 2: Count co-occurrences
        self._count_cooccurrences(documents, tokenizer, show_progress)

        # Step 3: Compute statistics
        self._compute_stats()

        return self

    def _build_vocabulary(
        self,
        documents: List[str],
        tokenizer: Optional[callable],
        show_progress: bool,
    ) -> None:
        """Build vocabulary from documents with frequency filtering.

        Args:
            documents: List of text documents
            tokenizer: Tokenizer function
            show_progress: Whether to show progress
        """
        self._term_freq = Counter()
        self._doc_freq = Counter()

        iterator = tqdm(documents, desc="Building vocabulary") if show_progress else documents

        for doc in iterator:
            tokens = self._tokenize(doc, tokenizer)
            self._term_freq.update(tokens)
            unique_tokens = set(tokens)
            self._doc_freq.update(unique_tokens)

        # Filter by minimum frequency
        filtered_terms = [
            term
            for term, freq in self._term_freq.items()
            if freq >= self.config.min_term_freq
        ]

        # Sort by frequency and limit vocabulary size
        filtered_terms.sort(key=lambda t: -self._term_freq[t])
        if len(filtered_terms) > self.config.max_vocab_size:
            filtered_terms = filtered_terms[: self.config.max_vocab_size]

        # Create vocabulary mappings
        self._vocab = {term: idx for idx, term in enumerate(filtered_terms)}
        self._reverse_vocab = {idx: term for term, idx in self._vocab.items()}

        self._stats.vocab_size = len(self._vocab)
        self._stats.total_documents = len(documents)

    def _count_cooccurrences(
        self,
        documents: List[str],
        tokenizer: Optional[callable],
        show_progress: bool,
    ) -> None:
        """Count co-occurrences in documents.

        Uses lil_matrix for efficient incremental updates, then converts to csr.

        Args:
            documents: List of text documents
            tokenizer: Tokenizer function
            show_progress: Whether to show progress
        """
        vocab_size = len(self._vocab)
        if vocab_size == 0:
            return

        # Use lil_matrix for efficient incremental construction
        cooc_lil = sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float32)

        iterator = tqdm(documents, desc="Counting co-occurrences") if show_progress else documents
        total_windows = 0

        for doc in iterator:
            windows = self._get_windows(doc, tokenizer)
            total_windows += len(windows)

            for window_tokens in windows:
                # Get valid vocabulary indices
                indices = [
                    self._vocab[t]
                    for t in window_tokens
                    if t in self._vocab
                ]

                if len(indices) < 2:
                    continue

                # Count co-occurrences within window
                for i, idx_i in enumerate(indices):
                    for idx_j in indices[i + 1:]:
                        weight = 1.0
                        if self.config.normalize:
                            weight = 1.0 / len(indices)

                        cooc_lil[idx_i, idx_j] += weight
                        if self.config.symmetric:
                            cooc_lil[idx_j, idx_i] += weight

        self._stats.total_windows = total_windows

        # Convert to csr for efficient operations
        self._cooccurrence_matrix = cooc_lil.tocsr()
        self._stats.total_cooccurrences = int(self._cooccurrence_matrix.nnz)

    def _get_windows(
        self,
        document: str,
        tokenizer: Optional[callable],
    ) -> List[List[str]]:
        """Extract co-occurrence windows from document.

        Args:
            document: Text document
            tokenizer: Tokenizer function

        Returns:
            List of token lists, one per window
        """
        if self.config.window_type == WindowType.SENTENCE:
            return self._get_sentence_windows(document, tokenizer)
        elif self.config.window_type == WindowType.PARAGRAPH:
            return self._get_paragraph_windows(document, tokenizer)
        else:
            return self._get_sliding_windows(document, tokenizer)

    def _get_sentence_windows(
        self,
        document: str,
        tokenizer: Optional[callable],
    ) -> List[List[str]]:
        """Split document into sentences as windows.

        Args:
            document: Text document
            tokenizer: Tokenizer function

        Returns:
            List of token lists per sentence
        """
        # Simple sentence splitting using common delimiters
        sentences = []
        current = []

        for char in document:
            if char in ".!?\n":
                if current:
                    sentences.append("".join(current).strip())
                current = []
            else:
                current.append(char)

        if current:
            sentences.append("".join(current).strip())

        return [self._tokenize(sent, tokenizer) for sent in sentences if sent]

    def _get_paragraph_windows(
        self,
        document: str,
        tokenizer: Optional[callable],
    ) -> List[List[str]]:
        """Split document into paragraphs as windows.

        Args:
            document: Text document
            tokenizer: Tokenizer function

        Returns:
            List of token lists per paragraph
        """
        paragraphs = document.split("\n\n")
        return [
            self._tokenize(para, tokenizer)
            for para in paragraphs
            if para.strip()
        ]

    def _get_sliding_windows(
        self,
        document: str,
        tokenizer: Optional[callable],
    ) -> List[List[str]]:
        """Create sliding windows over tokens.

        Args:
            document: Text document
            tokenizer: Tokenizer function

        Returns:
            List of token lists per sliding window
        """
        tokens = self._tokenize(document, tokenizer)
        window_size = self.config.window_size

        if len(tokens) <= window_size:
            return [tokens] if tokens else []

        windows = []
        for i in range(len(tokens) - window_size + 1):
            windows.append(tokens[i : i + window_size])

        return windows

    def _tokenize(
        self,
        text: str,
        tokenizer: Optional[callable],
    ) -> List[str]:
        """Tokenize text using provided or default tokenizer.

        Args:
            text: Input text
            tokenizer: Optional tokenizer function

        Returns:
            List of tokens
        """
        if tokenizer is not None:
            return tokenizer(text)
        return text.split()

    def _compute_stats(self) -> None:
        """Compute final statistics."""
        if self._cooccurrence_matrix is None:
            return

        vocab_size = len(self._vocab)
        total_possible = vocab_size * vocab_size
        if total_possible > 0:
            self._stats.sparsity = 1.0 - (
                self._cooccurrence_matrix.nnz / total_possible
            )

    def get_cooccurrence_matrix(self) -> sparse.csr_matrix:
        """Get the sparse co-occurrence matrix.

        Returns:
            Sparse CSR matrix of co-occurrence counts

        Raises:
            ValueError: If fit() has not been called
        """
        if self._cooccurrence_matrix is None:
            raise ValueError("Matrix not built. Call fit() first.")
        return self._cooccurrence_matrix

    def get_term_frequencies(self) -> Dict[str, int]:
        """Get term frequency dictionary (filtered by vocabulary).

        Returns:
            Dictionary mapping terms to their frequencies
        """
        return {term: self._term_freq[term] for term in self._vocab}

    def get_document_frequencies(self) -> Dict[str, int]:
        """Get document frequency dictionary.

        Returns:
            Dictionary mapping terms to number of documents containing them
        """
        return {term: self._doc_freq[term] for term in self._vocab}

    def get_vocabulary(self) -> Dict[str, int]:
        """Get vocabulary mapping.

        Returns:
            Dictionary mapping terms to indices
        """
        return self._vocab.copy()

    def get_term_by_index(self, index: int) -> Optional[str]:
        """Get term by vocabulary index.

        Args:
            index: Vocabulary index

        Returns:
            Term string or None if index not found
        """
        return self._reverse_vocab.get(index)

    def get_index_by_term(self, term: str) -> Optional[int]:
        """Get vocabulary index for term.

        Args:
            term: Term string

        Returns:
            Index or None if term not in vocabulary
        """
        return self._vocab.get(term)

    def get_cooccurrence_count(self, term1: str, term2: str) -> float:
        """Get co-occurrence count for a pair of terms.

        Args:
            term1: First term
            term2: Second term

        Returns:
            Co-occurrence count (0 if either term not in vocabulary)
        """
        if self._cooccurrence_matrix is None:
            return 0.0

        idx1 = self._vocab.get(term1)
        idx2 = self._vocab.get(term2)

        if idx1 is None or idx2 is None:
            return 0.0

        return float(self._cooccurrence_matrix[idx1, idx2])

    def get_stats(self) -> CooccurrenceStats:
        """Get construction statistics.

        Returns:
            Statistics about the built matrix
        """
        return self._stats

    def save(self, path: Union[str, Path]) -> None:
        """Save co-occurrence matrix and vocabulary to disk.

        Args:
            path: Directory path to save files
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save sparse matrix
        if self._cooccurrence_matrix is not None:
            sparse.save_npz(path / "cooccurrence_matrix.npz", self._cooccurrence_matrix)

        # Save vocabulary
        import json

        with open(path / "vocabulary.json", "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)

        # Save term frequencies
        with open(path / "term_frequencies.json", "w", encoding="utf-8") as f:
            term_freq_dict = {term: self._term_freq[term] for term in self._vocab}
            json.dump(term_freq_dict, f, ensure_ascii=False, indent=2)

        # Save config and stats
        config_dict = {
            "window_type": self.config.window_type.value,
            "window_size": self.config.window_size,
            "min_term_freq": self.config.min_term_freq,
            "max_vocab_size": self.config.max_vocab_size,
            "symmetric": self.config.symmetric,
            "normalize": self.config.normalize,
        }
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        stats_dict = {
            "vocab_size": self._stats.vocab_size,
            "total_documents": self._stats.total_documents,
            "total_windows": self._stats.total_windows,
            "total_cooccurrences": self._stats.total_cooccurrences,
            "sparsity": self._stats.sparsity,
        }
        with open(path / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats_dict, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CooccurrenceMatrixBuilder":
        """Load co-occurrence matrix and vocabulary from disk.

        Args:
            path: Directory path containing saved files

        Returns:
            Loaded CooccurrenceMatrixBuilder instance
        """
        import json

        path = Path(path)

        # Load config
        with open(path / "config.json", "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        config = CooccurrenceConfig(
            window_type=WindowType(config_dict["window_type"]),
            window_size=config_dict["window_size"],
            min_term_freq=config_dict["min_term_freq"],
            max_vocab_size=config_dict["max_vocab_size"],
            symmetric=config_dict["symmetric"],
            normalize=config_dict["normalize"],
        )

        builder = cls(config)

        # Load vocabulary
        with open(path / "vocabulary.json", "r", encoding="utf-8") as f:
            builder._vocab = json.load(f)
        builder._reverse_vocab = {idx: term for term, idx in builder._vocab.items()}

        # Load term frequencies
        with open(path / "term_frequencies.json", "r", encoding="utf-8") as f:
            term_freq_dict = json.load(f)
        builder._term_freq = Counter(term_freq_dict)

        # Load sparse matrix
        builder._cooccurrence_matrix = sparse.load_npz(
            path / "cooccurrence_matrix.npz"
        )

        # Load stats
        with open(path / "stats.json", "r", encoding="utf-8") as f:
            stats_dict = json.load(f)

        builder._stats = CooccurrenceStats(**stats_dict)

        return builder
