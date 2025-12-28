"""PMI (Pointwise Mutual Information) calculation with smoothing techniques.

This module implements PMI computation for synonym validation with support for:
- Standard PMI: log(P(x,y) / (P(x) * P(y)))
- PPMI (Positive PMI): max(0, PMI) - handles negative values
- Laplace smoothing: Adds pseudo-counts to handle zero co-occurrences
- Context distribution smoothing: Adjusts for frequency bias

PMI Interpretation:
- PMI > 0: Terms co-occur more often than expected by chance
- PMI = 0: Terms co-occur exactly as expected if independent
- PMI < 0: Terms co-occur less often than expected

For synonym validation:
- High PMI indicates genuine semantic relationship
- Low/negative PMI suggests spurious embedding similarity
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from tqdm import tqdm


@dataclass
class PMIConfig:
    """Configuration for PMI calculation.

    Attributes:
        laplace_smoothing: Pseudo-count for Laplace smoothing (0 = no smoothing)
        context_smoothing_alpha: Power for context distribution smoothing (0.75 typical)
        use_ppmi: Whether to use Positive PMI (max(0, PMI))
        log_base: Base for logarithm (2 for bits, e for nats)
        min_cooccurrence: Minimum co-occurrence count to compute PMI
    """

    laplace_smoothing: float = 1.0
    context_smoothing_alpha: float = 0.75
    use_ppmi: bool = True
    log_base: float = 2.0
    min_cooccurrence: int = 1


class PMICalculator:
    """Compute PMI scores for term pairs using co-occurrence statistics.

    This class provides efficient PMI computation with various smoothing
    techniques to handle zero co-occurrences and frequency bias.

    Example:
        >>> from src.pmi import CooccurrenceMatrixBuilder, PMICalculator
        >>> builder = CooccurrenceMatrixBuilder(config)
        >>> builder.fit(documents)
        >>> pmi_calc = PMICalculator(
        ...     cooccurrence_matrix=builder.get_cooccurrence_matrix(),
        ...     term_frequencies=builder.get_term_frequencies(),
        ...     vocabulary=builder.get_vocabulary(),
        ...     total_windows=builder.get_stats().total_windows,
        ... )
        >>> pmi_score = pmi_calc.compute_pmi("machine", "learning")
    """

    def __init__(
        self,
        cooccurrence_matrix: sparse.csr_matrix,
        term_frequencies: Dict[str, int],
        vocabulary: Dict[str, int],
        total_windows: int,
        config: Optional[PMIConfig] = None,
    ):
        """Initialize PMI calculator.

        Args:
            cooccurrence_matrix: Sparse co-occurrence count matrix
            term_frequencies: Dictionary mapping terms to frequencies
            vocabulary: Dictionary mapping terms to indices
            total_windows: Total number of co-occurrence windows in corpus
            config: PMI calculation configuration
        """
        self.cooc_matrix = cooccurrence_matrix
        self.term_freq = term_frequencies
        self.vocab = vocabulary
        self.reverse_vocab = {idx: term for term, idx in vocabulary.items()}
        self.total_windows = total_windows
        self.config = config or PMIConfig()

        # Precompute marginal probabilities
        self._compute_marginals()

    def _compute_marginals(self) -> None:
        """Precompute marginal probabilities for efficiency."""
        vocab_size = len(self.vocab)

        # Compute smoothed frequencies if context smoothing is enabled
        if self.config.context_smoothing_alpha != 1.0:
            freqs = np.zeros(vocab_size, dtype=np.float64)
            for term, idx in self.vocab.items():
                freqs[idx] = self.term_freq.get(term, 0)

            # Apply context distribution smoothing: f^alpha / sum(f^alpha)
            alpha = self.config.context_smoothing_alpha
            smoothed_freqs = np.power(freqs + 1e-10, alpha)
            self._marginal_probs = smoothed_freqs / smoothed_freqs.sum()
        else:
            # Standard marginal probabilities
            total_freq = sum(self.term_freq.values())
            self._marginal_probs = np.zeros(vocab_size, dtype=np.float64)
            for term, idx in self.vocab.items():
                self._marginal_probs[idx] = self.term_freq.get(term, 0) / total_freq

        # Total co-occurrence for joint probability
        self._total_cooc = float(self.cooc_matrix.sum())
        if self._total_cooc == 0:
            self._total_cooc = 1.0  # Avoid division by zero

    def compute_pmi(self, term1: str, term2: str) -> float:
        """Compute PMI score for a term pair.

        PMI(x, y) = log(P(x, y) / (P(x) * P(y)))

        With Laplace smoothing:
        PMI(x, y) = log((C(x,y) + k) / (N + k*V^2)) - log(P(x) * P(y))

        Args:
            term1: First term
            term2: Second term

        Returns:
            PMI score (or PPMI if configured)
        """
        idx1 = self.vocab.get(term1)
        idx2 = self.vocab.get(term2)

        # OOV handling
        if idx1 is None or idx2 is None:
            return float("-inf") if not self.config.use_ppmi else 0.0

        return self._compute_pmi_by_index(idx1, idx2)

    def _compute_pmi_by_index(self, idx1: int, idx2: int) -> float:
        """Compute PMI score using vocabulary indices.

        Args:
            idx1: Index of first term
            idx2: Index of second term

        Returns:
            PMI score
        """
        # Get co-occurrence count
        cooc_count = float(self.cooc_matrix[idx1, idx2])

        # Apply minimum co-occurrence threshold
        if cooc_count < self.config.min_cooccurrence:
            if self.config.laplace_smoothing > 0:
                cooc_count = self.config.laplace_smoothing
            else:
                return float("-inf") if not self.config.use_ppmi else 0.0

        # Apply Laplace smoothing
        k = self.config.laplace_smoothing
        vocab_size = len(self.vocab)

        # Smoothed joint probability
        smoothed_cooc = cooc_count + k
        smoothed_total = self._total_cooc + k * vocab_size * vocab_size
        p_joint = smoothed_cooc / smoothed_total

        # Marginal probabilities
        p1 = self._marginal_probs[idx1]
        p2 = self._marginal_probs[idx2]

        # Avoid division by zero
        if p1 == 0 or p2 == 0:
            return float("-inf") if not self.config.use_ppmi else 0.0

        # Compute PMI
        p_independent = p1 * p2

        if self.config.log_base == 2.0:
            pmi = np.log2(p_joint / p_independent)
        elif self.config.log_base == np.e:
            pmi = np.log(p_joint / p_independent)
        else:
            pmi = np.log(p_joint / p_independent) / np.log(self.config.log_base)

        # Apply PPMI if configured
        if self.config.use_ppmi:
            pmi = max(0.0, pmi)

        return float(pmi)

    def compute_pmi_batch(
        self,
        term_pairs: List[Tuple[str, str]],
        show_progress: bool = True,
    ) -> List[float]:
        """Compute PMI scores for multiple term pairs efficiently.

        Args:
            term_pairs: List of (term1, term2) tuples
            show_progress: Whether to show progress bar

        Returns:
            List of PMI scores corresponding to input pairs
        """
        iterator = tqdm(term_pairs, desc="Computing PMI") if show_progress else term_pairs

        results = []
        for term1, term2 in iterator:
            results.append(self.compute_pmi(term1, term2))

        return results

    def compute_pmi_matrix(self) -> sparse.csr_matrix:
        """Compute full PMI matrix from co-occurrence matrix.

        This is memory-efficient using sparse operations.

        Returns:
            Sparse PMI matrix
        """
        vocab_size = len(self.vocab)
        k = self.config.laplace_smoothing

        # Get non-zero entries
        rows, cols = self.cooc_matrix.nonzero()
        data = []

        for i, (r, c) in enumerate(zip(rows, cols)):
            pmi = self._compute_pmi_by_index(r, c)
            if not np.isinf(pmi):
                data.append(pmi)
            else:
                data.append(0.0)

        pmi_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(vocab_size, vocab_size),
            dtype=np.float32,
        )

        return pmi_matrix

    def get_pmi_percentile(
        self,
        term_pairs: List[Tuple[str, str]],
        percentile: float,
    ) -> float:
        """Compute PMI threshold at given percentile.

        Args:
            term_pairs: List of term pairs to compute distribution
            percentile: Percentile value (0-100)

        Returns:
            PMI value at the given percentile
        """
        pmi_scores = self.compute_pmi_batch(term_pairs, show_progress=False)

        # Filter out infinite values
        valid_scores = [s for s in pmi_scores if not np.isinf(s)]

        if not valid_scores:
            return 0.0

        return float(np.percentile(valid_scores, percentile))

    def filter_by_pmi_threshold(
        self,
        term_pairs: List[Tuple[str, str]],
        threshold: Optional[float] = None,
        percentile: Optional[float] = None,
        show_progress: bool = True,
    ) -> Tuple[List[Tuple[str, str]], List[float]]:
        """Filter term pairs by PMI threshold.

        Args:
            term_pairs: List of term pairs to filter
            threshold: Absolute PMI threshold (pairs with PMI >= threshold kept)
            percentile: Percentile threshold (pairs above percentile kept)
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (filtered_pairs, pmi_scores)

        Raises:
            ValueError: If neither threshold nor percentile is provided
        """
        if threshold is None and percentile is None:
            raise ValueError("Either threshold or percentile must be provided")

        pmi_scores = self.compute_pmi_batch(term_pairs, show_progress)

        # Determine threshold
        if threshold is None:
            valid_scores = [s for s in pmi_scores if not np.isinf(s)]
            if valid_scores:
                threshold = np.percentile(valid_scores, percentile)
            else:
                threshold = 0.0

        # Filter pairs
        filtered_pairs = []
        filtered_scores = []

        for pair, score in zip(term_pairs, pmi_scores):
            if score >= threshold:
                filtered_pairs.append(pair)
                filtered_scores.append(score)

        return filtered_pairs, filtered_scores

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get PMI calculation statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "vocab_size": len(self.vocab),
            "total_windows": self.total_windows,
            "total_cooccurrences": self._total_cooc,
            "laplace_smoothing": self.config.laplace_smoothing,
            "context_smoothing_alpha": self.config.context_smoothing_alpha,
            "use_ppmi": self.config.use_ppmi,
        }


class PPMICalculator(PMICalculator):
    """Convenience class for Positive PMI calculation.

    PPMI(x, y) = max(0, PMI(x, y))

    This is commonly used in distributional semantics to avoid
    negative associations which can be unreliable.
    """

    def __init__(
        self,
        cooccurrence_matrix: sparse.csr_matrix,
        term_frequencies: Dict[str, int],
        vocabulary: Dict[str, int],
        total_windows: int,
        laplace_smoothing: float = 1.0,
        context_smoothing_alpha: float = 0.75,
    ):
        """Initialize PPMI calculator.

        Args:
            cooccurrence_matrix: Sparse co-occurrence count matrix
            term_frequencies: Dictionary mapping terms to frequencies
            vocabulary: Dictionary mapping terms to indices
            total_windows: Total number of co-occurrence windows
            laplace_smoothing: Pseudo-count for Laplace smoothing
            context_smoothing_alpha: Power for context distribution smoothing
        """
        config = PMIConfig(
            laplace_smoothing=laplace_smoothing,
            context_smoothing_alpha=context_smoothing_alpha,
            use_ppmi=True,
            log_base=2.0,
        )
        super().__init__(
            cooccurrence_matrix,
            term_frequencies,
            vocabulary,
            total_windows,
            config,
        )


def compute_npmi(
    pmi_score: float,
    p_joint: float,
    log_base: float = 2.0,
) -> float:
    """Compute Normalized PMI (NPMI).

    NPMI(x, y) = PMI(x, y) / -log(P(x, y))

    NPMI ranges from -1 (never co-occur) to +1 (always co-occur),
    with 0 indicating independence.

    Args:
        pmi_score: Pre-computed PMI score
        p_joint: Joint probability P(x, y)
        log_base: Base for logarithm

    Returns:
        Normalized PMI score in range [-1, 1]
    """
    if p_joint <= 0:
        return 0.0

    if log_base == 2.0:
        h_joint = -np.log2(p_joint)
    else:
        h_joint = -np.log(p_joint) / np.log(log_base)

    if h_joint == 0:
        return 0.0

    return pmi_score / h_joint
