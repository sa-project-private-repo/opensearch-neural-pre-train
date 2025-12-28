"""Synonym validation using PMI scores.

This module provides functionality to validate synonym pairs by combining
embedding similarity with co-occurrence statistics (PMI).

The key insight is that high embedding similarity alone does not guarantee
true synonymy. Terms must also co-occur in similar contexts to be considered
genuine synonyms.

Validation Strategy:
1. Embedding similarity: Terms are semantically similar in vector space
2. PMI validation: Terms actually co-occur in the corpus (not just similar)
3. Threshold filtering: Use percentile-based thresholds to remove low-quality pairs

OOV Handling:
- Terms not in corpus vocabulary are flagged
- BPE expansion pairs with OOV components are separately tracked
- Configurable treatment: remove, keep, or replace with smoothed estimate
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy import sparse
from tqdm import tqdm

from src.pmi.cooccurrence import CooccurrenceMatrixBuilder
from src.pmi.pmi_calculator import PMICalculator, PMIConfig


class OOVStrategy(Enum):
    """Strategy for handling Out-of-Vocabulary terms."""

    REMOVE = "remove"  # Remove pairs with OOV terms
    KEEP = "keep"  # Keep pairs with OOV terms (assign PMI=0)
    SMOOTH = "smooth"  # Use smoothed PMI estimate for OOV


@dataclass
class SynonymPair:
    """Represents a synonym pair with validation scores.

    Attributes:
        source: Source term
        target: Target term
        embedding_similarity: Similarity score from embeddings
        pmi_score: PMI score from co-occurrence
        is_valid: Whether pair passes validation
        category: Original category (e.g., "cluster", "BPE")
        oov_status: OOV status ("both_in_vocab", "source_oov", "target_oov", "both_oov")
    """

    source: str
    target: str
    embedding_similarity: float = 0.0
    pmi_score: float = 0.0
    is_valid: bool = True
    category: str = ""
    oov_status: str = "both_in_vocab"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON-serializable values."""
        return {
            "source": self.source,
            "target": self.target,
            "embedding_similarity": float(self.embedding_similarity),
            "pmi_score": float(self.pmi_score) if not np.isinf(self.pmi_score) else 0.0,
            "is_valid": bool(self.is_valid),
            "category": self.category,
            "oov_status": self.oov_status,
        }


@dataclass
class ValidationConfig:
    """Configuration for synonym validation.

    Attributes:
        pmi_percentile_threshold: Percentile threshold for PMI filtering (0-100)
        pmi_absolute_threshold: Absolute PMI threshold (overrides percentile if set)
        min_embedding_similarity: Minimum embedding similarity to consider
        oov_strategy: How to handle OOV terms
        separate_bpe_validation: Whether to validate BPE pairs separately
    """

    pmi_percentile_threshold: float = 10.0  # Remove bottom 10%
    pmi_absolute_threshold: Optional[float] = None
    min_embedding_similarity: float = 0.5
    oov_strategy: OOVStrategy = OOVStrategy.KEEP
    separate_bpe_validation: bool = True


@dataclass
class ValidationResult:
    """Results from synonym validation.

    Attributes:
        total_pairs: Total number of input pairs
        valid_pairs: Number of pairs that passed validation
        removed_pairs: Number of pairs removed
        oov_pairs: Number of pairs with OOV terms
        pmi_threshold: PMI threshold used for filtering
        stats: Additional statistics
    """

    total_pairs: int = 0
    valid_pairs: int = 0
    removed_pairs: int = 0
    oov_pairs: int = 0
    pmi_threshold: float = 0.0
    stats: Dict[str, Any] = field(default_factory=dict)


class SynonymValidator:
    """Validate synonym pairs using PMI scores.

    This class combines embedding-based similarity with co-occurrence-based
    PMI to filter out spurious synonym pairs.

    Example:
        >>> validator = SynonymValidator(pmi_calculator, config)
        >>> validated_pairs, result = validator.validate(synonym_pairs)
    """

    def __init__(
        self,
        pmi_calculator: PMICalculator,
        config: Optional[ValidationConfig] = None,
    ):
        """Initialize synonym validator.

        Args:
            pmi_calculator: PMI calculator with loaded co-occurrence data
            config: Validation configuration
        """
        self.pmi_calc = pmi_calculator
        self.config = config or ValidationConfig()
        self._vocabulary = set(pmi_calculator.vocab.keys())

    def validate(
        self,
        pairs: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> Tuple[List[SynonymPair], ValidationResult]:
        """Validate synonym pairs using PMI scores.

        Args:
            pairs: List of synonym pair dictionaries with keys:
                   - source: Source term
                   - target: Target term
                   - similarity: Embedding similarity score
                   - category: Pair category (optional)
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (validated_pairs, validation_result)
        """
        result = ValidationResult(total_pairs=len(pairs))

        # Convert to SynonymPair objects and check OOV status
        synonym_pairs = []
        for pair in pairs:
            sp = self._create_synonym_pair(pair)
            synonym_pairs.append(sp)

        # Separate by category if configured
        if self.config.separate_bpe_validation:
            cluster_pairs = [p for p in synonym_pairs if p.category != "BPE"]
            bpe_pairs = [p for p in synonym_pairs if p.category == "BPE"]

            validated_cluster = self._validate_batch(cluster_pairs, show_progress)
            validated_bpe = self._validate_batch(
                bpe_pairs, show_progress, desc="Validating BPE pairs"
            )

            validated_pairs = validated_cluster + validated_bpe
        else:
            validated_pairs = self._validate_batch(synonym_pairs, show_progress)

        # Compute final statistics
        result.valid_pairs = sum(1 for p in validated_pairs if p.is_valid)
        result.removed_pairs = result.total_pairs - result.valid_pairs
        result.oov_pairs = sum(1 for p in validated_pairs if p.oov_status != "both_in_vocab")
        result.stats = self._compute_validation_stats(validated_pairs)

        return validated_pairs, result

    def _create_synonym_pair(self, pair: Dict[str, Any]) -> SynonymPair:
        """Create SynonymPair from dictionary with OOV status.

        Args:
            pair: Dictionary with pair information

        Returns:
            SynonymPair object with OOV status
        """
        source = pair.get("source", "")
        target = pair.get("target", "")

        source_in_vocab = source in self._vocabulary
        target_in_vocab = target in self._vocabulary

        if source_in_vocab and target_in_vocab:
            oov_status = "both_in_vocab"
        elif not source_in_vocab and not target_in_vocab:
            oov_status = "both_oov"
        elif not source_in_vocab:
            oov_status = "source_oov"
        else:
            oov_status = "target_oov"

        return SynonymPair(
            source=source,
            target=target,
            embedding_similarity=pair.get("similarity", 0.0),
            pmi_score=0.0,  # Will be computed later
            is_valid=True,  # Will be updated during validation
            category=pair.get("category", ""),
            oov_status=oov_status,
        )

    def _validate_batch(
        self,
        pairs: List[SynonymPair],
        show_progress: bool,
        desc: str = "Validating pairs",
    ) -> List[SynonymPair]:
        """Validate a batch of synonym pairs.

        Args:
            pairs: List of SynonymPair objects
            show_progress: Whether to show progress
            desc: Description for progress bar

        Returns:
            List of validated SynonymPair objects
        """
        if not pairs:
            return []

        # Compute PMI scores for all pairs
        term_pairs = [(p.source, p.target) for p in pairs]
        pmi_scores = self.pmi_calc.compute_pmi_batch(term_pairs, show_progress)

        # Update pairs with PMI scores
        for pair, pmi in zip(pairs, pmi_scores):
            pair.pmi_score = pmi

        # Determine threshold
        if self.config.pmi_absolute_threshold is not None:
            threshold = self.config.pmi_absolute_threshold
        else:
            # Compute percentile threshold from in-vocab pairs only
            in_vocab_scores = [
                p.pmi_score
                for p in pairs
                if p.oov_status == "both_in_vocab" and not np.isinf(p.pmi_score)
            ]
            if in_vocab_scores:
                threshold = np.percentile(
                    in_vocab_scores, self.config.pmi_percentile_threshold
                )
            else:
                threshold = 0.0

        # Apply validation
        for pair in pairs:
            pair.is_valid = self._is_pair_valid(pair, threshold)

        return pairs

    def _is_pair_valid(self, pair: SynonymPair, pmi_threshold: float) -> bool:
        """Check if a synonym pair is valid.

        Args:
            pair: SynonymPair to validate
            pmi_threshold: PMI threshold for filtering

        Returns:
            True if pair is valid
        """
        # Check embedding similarity
        if pair.embedding_similarity < self.config.min_embedding_similarity:
            return False

        # Handle OOV cases
        if pair.oov_status != "both_in_vocab":
            if self.config.oov_strategy == OOVStrategy.REMOVE:
                return False
            elif self.config.oov_strategy == OOVStrategy.KEEP:
                return True  # Keep OOV pairs without PMI validation
            else:
                # SMOOTH: Use PMI=0 as neutral estimate
                pair.pmi_score = 0.0
                return True

        # Check PMI threshold
        if np.isinf(pair.pmi_score) and pair.pmi_score < 0:
            return False

        return pair.pmi_score >= pmi_threshold

    def _compute_validation_stats(
        self, pairs: List[SynonymPair]
    ) -> Dict[str, Any]:
        """Compute validation statistics.

        Args:
            pairs: List of validated pairs

        Returns:
            Dictionary of statistics
        """
        valid_pairs = [p for p in pairs if p.is_valid]
        invalid_pairs = [p for p in pairs if not p.is_valid]

        # PMI distribution for valid pairs
        valid_pmi_scores = [
            p.pmi_score for p in valid_pairs if not np.isinf(p.pmi_score)
        ]
        invalid_pmi_scores = [
            p.pmi_score for p in invalid_pairs if not np.isinf(p.pmi_score)
        ]

        stats = {
            "total": len(pairs),
            "valid": len(valid_pairs),
            "invalid": len(invalid_pairs),
            "by_category": {},
            "by_oov_status": {},
            "pmi_stats": {},
        }

        # By category
        for category in set(p.category for p in pairs):
            cat_pairs = [p for p in pairs if p.category == category]
            cat_valid = sum(1 for p in cat_pairs if p.is_valid)
            stats["by_category"][category] = {
                "total": len(cat_pairs),
                "valid": cat_valid,
                "valid_ratio": cat_valid / len(cat_pairs) if cat_pairs else 0,
            }

        # By OOV status
        for status in set(p.oov_status for p in pairs):
            status_pairs = [p for p in pairs if p.oov_status == status]
            status_valid = sum(1 for p in status_pairs if p.is_valid)
            stats["by_oov_status"][status] = {
                "total": len(status_pairs),
                "valid": status_valid,
                "valid_ratio": status_valid / len(status_pairs) if status_pairs else 0,
            }

        # PMI statistics
        if valid_pmi_scores:
            stats["pmi_stats"]["valid"] = {
                "min": float(np.min(valid_pmi_scores)),
                "max": float(np.max(valid_pmi_scores)),
                "mean": float(np.mean(valid_pmi_scores)),
                "median": float(np.median(valid_pmi_scores)),
                "std": float(np.std(valid_pmi_scores)),
            }

        if invalid_pmi_scores:
            stats["pmi_stats"]["invalid"] = {
                "min": float(np.min(invalid_pmi_scores)),
                "max": float(np.max(invalid_pmi_scores)),
                "mean": float(np.mean(invalid_pmi_scores)),
                "median": float(np.median(invalid_pmi_scores)),
            }

        return stats

    def get_oov_terms(self, pairs: List[Dict[str, Any]]) -> Set[str]:
        """Get set of OOV terms from synonym pairs.

        Args:
            pairs: List of synonym pair dictionaries

        Returns:
            Set of terms not in vocabulary
        """
        oov_terms = set()
        for pair in pairs:
            source = pair.get("source", "")
            target = pair.get("target", "")
            if source not in self._vocabulary:
                oov_terms.add(source)
            if target not in self._vocabulary:
                oov_terms.add(target)
        return oov_terms

    def save_validation_report(
        self,
        pairs: List[SynonymPair],
        result: ValidationResult,
        path: Union[str, Path],
    ) -> None:
        """Save validation report to files.

        Args:
            pairs: Validated pairs
            result: Validation result
            path: Output directory path
        """
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save validated pairs as JSONL
        valid_pairs_path = path / "validated_pairs.jsonl"
        invalid_pairs_path = path / "invalid_pairs.jsonl"

        with open(valid_pairs_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                if pair.is_valid:
                    f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

        with open(invalid_pairs_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                if not pair.is_valid:
                    f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

        # Save summary report
        report = {
            "total_pairs": result.total_pairs,
            "valid_pairs": result.valid_pairs,
            "removed_pairs": result.removed_pairs,
            "oov_pairs": result.oov_pairs,
            "pmi_threshold": result.pmi_threshold,
            "validation_ratio": result.valid_pairs / result.total_pairs
            if result.total_pairs > 0
            else 0,
            "stats": result.stats,
            "config": {
                "pmi_percentile_threshold": self.config.pmi_percentile_threshold,
                "pmi_absolute_threshold": self.config.pmi_absolute_threshold,
                "min_embedding_similarity": self.config.min_embedding_similarity,
                "oov_strategy": self.config.oov_strategy.value,
                "separate_bpe_validation": self.config.separate_bpe_validation,
            },
        }

        with open(path / "validation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


def create_pmi_pipeline(
    documents: List[str],
    tokenizer: Optional[callable] = None,
    cooc_config: Optional[Any] = None,
    pmi_config: Optional[PMIConfig] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_progress: bool = True,
) -> Tuple[CooccurrenceMatrixBuilder, PMICalculator]:
    """Create complete PMI calculation pipeline.

    Convenience function to build co-occurrence matrix and PMI calculator
    in one step.

    Args:
        documents: List of text documents
        tokenizer: Optional tokenizer function
        cooc_config: Configuration for co-occurrence matrix
        pmi_config: Configuration for PMI calculation
        save_path: Optional path to save intermediate results
        show_progress: Whether to show progress

    Returns:
        Tuple of (CooccurrenceMatrixBuilder, PMICalculator)
    """
    from src.pmi.cooccurrence import CooccurrenceConfig

    # Build co-occurrence matrix
    cooc_config = cooc_config or CooccurrenceConfig()
    builder = CooccurrenceMatrixBuilder(cooc_config)
    builder.fit(documents, tokenizer, show_progress)

    if save_path:
        builder.save(save_path)

    # Create PMI calculator
    pmi_config = pmi_config or PMIConfig()
    pmi_calc = PMICalculator(
        cooccurrence_matrix=builder.get_cooccurrence_matrix(),
        term_frequencies=builder.get_term_frequencies(),
        vocabulary=builder.get_vocabulary(),
        total_windows=builder.get_stats().total_windows,
        config=pmi_config,
    )

    return builder, pmi_calc
