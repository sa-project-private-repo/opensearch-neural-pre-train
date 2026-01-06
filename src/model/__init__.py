"""Model modules for sparse retrieval."""

from src.model.splade_model import (
    SPLADEDoc,
    SPLADEDocExpansion,
    SPLADEDocWithIDF,
    create_splade_model,
)
from src.model.splade_v3 import SPLADEv3, SparseEmbedding, load_splade_v3
from src.model.losses import (
    InfoNCELoss,
    SelfReconstructionLoss,
    PositiveActivationLoss,
    TripletMarginLoss,
    FLOPSLoss,
    MinimumActivationLoss,
    SPLADELossV22,
)

__all__ = [
    # Models
    "SPLADEDoc",
    "SPLADEDocExpansion",
    "SPLADEDocWithIDF",
    "create_splade_model",
    "SPLADEv3",
    "SparseEmbedding",
    "load_splade_v3",
    # Losses
    "InfoNCELoss",
    "SelfReconstructionLoss",
    "PositiveActivationLoss",
    "TripletMarginLoss",
    "FLOPSLoss",
    "MinimumActivationLoss",
    "SPLADELossV22",
]
