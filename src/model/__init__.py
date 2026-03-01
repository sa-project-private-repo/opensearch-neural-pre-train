"""Model modules for sparse retrieval (V33 SPLADEModernBERT)."""

from src.model.splade_modern import SPLADEModernBERT
from src.model.teachers import BGEM3Teacher, create_bge_m3_teacher
from src.model.losses import SPLADELossV33

__all__ = [
    "SPLADEModernBERT",
    "BGEM3Teacher",
    "create_bge_m3_teacher",
    "SPLADELossV33",
]
