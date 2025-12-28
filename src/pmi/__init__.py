"""PMI (Pointwise Mutual Information) calculation module for synonym validation.

This module provides efficient PMI computation for validating synonym pairs
based on co-occurrence statistics in a text corpus.

Key Components:
- CooccurrenceMatrixBuilder: Builds sparse co-occurrence matrices from text
- PMICalculator: Computes PMI scores with various smoothing techniques
- SynonymValidator: Validates synonym pairs using PMI thresholds
"""

from src.pmi.cooccurrence import CooccurrenceMatrixBuilder
from src.pmi.pmi_calculator import PMICalculator, PPMICalculator
from src.pmi.synonym_validator import SynonymValidator

__all__ = [
    "CooccurrenceMatrixBuilder",
    "PMICalculator",
    "PPMICalculator",
    "SynonymValidator",
]
