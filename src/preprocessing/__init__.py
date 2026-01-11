"""
Korean dataset preprocessing for Neural Sparse training.

This module provides tools for:
- Downloading Korean NLP datasets from HuggingFace
- Converting various task formats to triplets
- Hard negative mining with BGE-M3
- Text cleaning and deduplication
"""

from src.preprocessing.config import PipelineConfig
from src.preprocessing.pipeline import PreprocessingPipeline

__all__ = ["PipelineConfig", "PreprocessingPipeline"]
