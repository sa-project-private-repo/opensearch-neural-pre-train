"""Model modules for sparse retrieval."""

from src.model.splade_model import (
    SPLADEDoc,
    SPLADEDocExpansion,
    SPLADEDocWithIDF,
    create_splade_model,
)
from src.model.splade_xlmr import (
    SPLADEDocXLMR,
    SPLADEDocXLMRWithIDF,
    create_splade_xlmr,
    load_splade_xlmr,
)
from src.model.splade_v3 import SPLADEv3, SparseEmbedding, load_splade_v3
from src.model.teachers import BGEM3Teacher, create_bge_m3_teacher
from src.model.losses import (
    InfoNCELoss,
    SelfReconstructionLoss,
    PositiveActivationLoss,
    TripletMarginLoss,
    FLOPSLoss,
    MinimumActivationLoss,
    SPLADELossV22,
    IDFAwareFLOPSLoss,
    KnowledgeDistillationLoss,
    DenseTeacherScorer,
    SPLADELossV23,
    SPLADELossV25,
)

__all__ = [
    # Models - KoBERT/mBERT
    "SPLADEDoc",
    "SPLADEDocExpansion",
    "SPLADEDocWithIDF",
    "create_splade_model",
    # Models - XLM-RoBERTa
    "SPLADEDocXLMR",
    "SPLADEDocXLMRWithIDF",
    "create_splade_xlmr",
    "load_splade_xlmr",
    # Models - v3
    "SPLADEv3",
    "SparseEmbedding",
    "load_splade_v3",
    # Teachers
    "BGEM3Teacher",
    "create_bge_m3_teacher",
    # Losses
    "InfoNCELoss",
    "SelfReconstructionLoss",
    "PositiveActivationLoss",
    "TripletMarginLoss",
    "FLOPSLoss",
    "MinimumActivationLoss",
    "SPLADELossV22",
    # V23 Enhanced Losses
    "IDFAwareFLOPSLoss",
    "KnowledgeDistillationLoss",
    "DenseTeacherScorer",
    "SPLADELossV23",
    # V25 IDF-Aware Loss
    "SPLADELossV25",
]
