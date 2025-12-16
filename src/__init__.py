"""
OpenSearch Neural Sparse Training - Source Package

This package contains the SPLADE model for neural sparse retrieval.
"""

__version__ = "0.4.0"
__author__ = "OpenSearch Neural Sparse Team"

# Only export the model - other modules have been consolidated into notebooks
from src.model.splade_model import (
    create_splade_model,
    SPLADEDoc,
    SPLADEDocWithIDF,
    SPLADEDocExpansion,
)

__all__ = [
    "__version__",
    "__author__",
    "create_splade_model",
    "SPLADEDoc",
    "SPLADEDocWithIDF",
    "SPLADEDocExpansion",
]
