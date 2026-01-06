"""
Loss functions for SPLADE neural sparse training.

Includes:
- InfoNCE contrastive loss with in-batch negatives
- Self-reconstruction loss
- Positive activation loss
- Triplet margin loss
- FLOPS regularization loss
- Minimum activation loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss with in-batch negatives.

    Computes: -log(exp(sim(q, p+) / τ) / Σ exp(sim(q, p_i) / τ))

    This loss treats other samples in the batch as negatives,
    providing more training signal per batch.
    """

    def __init__(
        self,
        temperature: float = 0.05,
        similarity: str = "cosine",
    ):
        """
        Initialize InfoNCE loss.

        Args:
            temperature: Temperature for softmax scaling (lower = sharper)
            similarity: Similarity function ("cosine" or "dot")
        """
        super().__init__()
        self.temperature = temperature
        self.similarity = similarity

    def forward(
        self,
        anchor_repr: torch.Tensor,
        positive_repr: torch.Tensor,
        negative_repr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            anchor_repr: Anchor representations [batch_size, dim]
            positive_repr: Positive representations [batch_size, dim]
            negative_repr: Optional explicit negatives [batch_size, dim]
                          If None, uses in-batch negatives only

        Returns:
            Scalar loss value
        """
        batch_size = anchor_repr.shape[0]

        # Normalize for cosine similarity
        if self.similarity == "cosine":
            anchor_repr = F.normalize(anchor_repr, p=2, dim=-1)
            positive_repr = F.normalize(positive_repr, p=2, dim=-1)
            if negative_repr is not None:
                negative_repr = F.normalize(negative_repr, p=2, dim=-1)

        # Compute similarity matrix between anchors and all positives
        # [batch_size, batch_size] - each row is anchor vs all positives
        sim_matrix = torch.mm(anchor_repr, positive_repr.t()) / self.temperature

        # If explicit negatives provided, add them
        if negative_repr is not None:
            # Similarity with explicit negatives [batch_size, batch_size]
            neg_sim = torch.mm(anchor_repr, negative_repr.t()) / self.temperature
            # Concatenate: [batch_size, 2*batch_size]
            sim_matrix = torch.cat([sim_matrix, neg_sim], dim=1)

        # Labels: diagonal elements are positives (index i for sample i)
        labels = torch.arange(batch_size, device=anchor_repr.device)

        # Cross entropy loss with softmax over all samples
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class SelfReconstructionLoss(nn.Module):
    """
    Self-reconstruction loss for SPLADE.

    Encourages the model to activate the actual input tokens.
    This is important for inference-free sparse retrieval where
    only input tokens should be activated.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        sparse_repr: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute self-reconstruction loss.

        Args:
            sparse_repr: Sparse representation [batch_size, vocab_size]
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Scalar loss value
        """
        batch_size, vocab_size = sparse_repr.shape

        # Create target: one-hot encoding of input tokens
        # For each token in input, we want high activation
        target = torch.zeros_like(sparse_repr)
        target.scatter_add_(
            1,
            input_ids,
            attention_mask.float(),
        )
        # Normalize by number of tokens
        target = target.clamp(max=1.0)

        # Binary cross entropy: encourage activation of input tokens
        # and penalize activation of other tokens
        loss = F.binary_cross_entropy_with_logits(
            sparse_repr,
            target,
            reduction=self.reduction,
        )

        return loss


class PositiveActivationLoss(nn.Module):
    """
    Positive activation loss for synonym learning.

    Encourages anchor to activate tokens from positive document,
    promoting cross-document term alignment.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        anchor_repr: torch.Tensor,
        positive_input_ids: torch.Tensor,
        positive_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute positive activation loss.

        Args:
            anchor_repr: Anchor sparse representation [batch_size, vocab_size]
            positive_input_ids: Positive document tokens [batch_size, seq_len]
            positive_attention_mask: Positive attention mask [batch_size, seq_len]

        Returns:
            Scalar loss value
        """
        batch_size, vocab_size = anchor_repr.shape

        # Create mask for positive tokens
        positive_mask = torch.zeros_like(anchor_repr)
        positive_mask.scatter_add_(
            1,
            positive_input_ids,
            positive_attention_mask.float(),
        )
        positive_mask = (positive_mask > 0).float()

        # We want anchor to have HIGH activation for positive tokens
        # Loss: -log(sigmoid(anchor_repr)) for positive token positions
        positive_activations = anchor_repr * positive_mask
        num_positive_tokens = positive_mask.sum(dim=1).clamp(min=1)

        # Mean activation for positive tokens (higher is better)
        mean_positive_activation = positive_activations.sum(dim=1) / num_positive_tokens

        # Loss: we want high activation, so minimize -activation
        loss = -mean_positive_activation.mean()

        return loss


class TripletMarginLoss(nn.Module):
    """
    Triplet margin loss using cosine similarity.

    Ensures: sim(anchor, positive) > sim(anchor, negative) + margin
    """

    def __init__(
        self,
        margin: float = 1.0,
        similarity: str = "cosine",
        reduction: str = "mean",
    ):
        super().__init__()
        self.margin = margin
        self.similarity = similarity
        self.reduction = reduction

    def forward(
        self,
        anchor_repr: torch.Tensor,
        positive_repr: torch.Tensor,
        negative_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet margin loss.

        Args:
            anchor_repr: Anchor representations [batch_size, dim]
            positive_repr: Positive representations [batch_size, dim]
            negative_repr: Negative representations [batch_size, dim]

        Returns:
            Scalar loss value
        """
        if self.similarity == "cosine":
            # Normalize
            anchor_repr = F.normalize(anchor_repr, p=2, dim=-1)
            positive_repr = F.normalize(positive_repr, p=2, dim=-1)
            negative_repr = F.normalize(negative_repr, p=2, dim=-1)

            # Cosine similarity
            pos_sim = (anchor_repr * positive_repr).sum(dim=-1)
            neg_sim = (anchor_repr * negative_repr).sum(dim=-1)
        else:
            # Dot product similarity
            pos_sim = (anchor_repr * positive_repr).sum(dim=-1)
            neg_sim = (anchor_repr * negative_repr).sum(dim=-1)

        # Triplet loss: max(0, margin - pos_sim + neg_sim)
        loss = F.relu(self.margin - pos_sim + neg_sim)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FLOPSLoss(nn.Module):
    """
    FLOPS regularization loss for sparsity.

    Penalizes the average activation across the vocabulary,
    encouraging sparse representations.

    Based on SPLADE paper: L_FLOPS = Σ(mean_activation_j)^2
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, sparse_repr: torch.Tensor) -> torch.Tensor:
        """
        Compute FLOPS loss.

        Args:
            sparse_repr: Sparse representations [batch_size, vocab_size]

        Returns:
            Scalar loss value
        """
        # Mean activation per vocabulary token across batch
        mean_activation = sparse_repr.mean(dim=0)  # [vocab_size]

        # L2 penalty on mean activations
        loss = (mean_activation ** 2).sum()

        return loss


class MinimumActivationLoss(nn.Module):
    """
    Minimum activation loss to prevent garbage outputs.

    Ensures the top-k activations stay above a threshold,
    preventing the model from producing near-zero outputs.
    """

    def __init__(
        self,
        top_k: int = 5,
        min_activation: float = 0.5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.top_k = top_k
        self.min_activation = min_activation
        self.reduction = reduction

    def forward(self, sparse_repr: torch.Tensor) -> torch.Tensor:
        """
        Compute minimum activation loss.

        Args:
            sparse_repr: Sparse representations [batch_size, vocab_size]

        Returns:
            Scalar loss value
        """
        # Get top-k activations per sample
        top_k_values, _ = torch.topk(sparse_repr, k=self.top_k, dim=-1)

        # Mean of top-k activations
        mean_top_k = top_k_values.mean(dim=-1)  # [batch_size]

        # Penalize if mean top-k is below threshold
        loss = F.relu(self.min_activation - mean_top_k)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SPLADELossV22(nn.Module):
    """
    Combined loss function for SPLADE v22.0 training.

    Combines all loss components with configurable weights:
    - InfoNCE contrastive loss (NEW in v22.0)
    - Self-reconstruction loss
    - Positive activation loss
    - Triplet margin loss
    - FLOPS regularization
    - Minimum activation loss
    """

    def __init__(
        self,
        lambda_infonce: float = 2.0,
        lambda_self: float = 4.0,
        lambda_positive: float = 10.0,
        lambda_margin: float = 2.5,
        lambda_flops: float = 5e-3,
        lambda_min_act: float = 1.0,
        temperature: float = 0.05,
        margin: float = 1.5,
        top_k: int = 5,
        min_activation: float = 0.5,
    ):
        """
        Initialize combined loss.

        Args:
            lambda_infonce: Weight for InfoNCE loss
            lambda_self: Weight for self-reconstruction loss
            lambda_positive: Weight for positive activation loss
            lambda_margin: Weight for triplet margin loss
            lambda_flops: Weight for FLOPS regularization
            lambda_min_act: Weight for minimum activation loss
            temperature: Temperature for InfoNCE
            margin: Margin for triplet loss
            top_k: Top-k for minimum activation
            min_activation: Minimum activation threshold
        """
        super().__init__()

        # Loss weights
        self.lambda_infonce = lambda_infonce
        self.lambda_self = lambda_self
        self.lambda_positive = lambda_positive
        self.lambda_margin = lambda_margin
        self.lambda_flops = lambda_flops
        self.lambda_min_act = lambda_min_act

        # Individual loss modules
        self.infonce_loss = InfoNCELoss(temperature=temperature)
        self.self_loss = SelfReconstructionLoss()
        self.positive_loss = PositiveActivationLoss()
        self.margin_loss = TripletMarginLoss(margin=margin)
        self.flops_loss = FLOPSLoss()
        self.min_act_loss = MinimumActivationLoss(
            top_k=top_k, min_activation=min_activation
        )

    def forward(
        self,
        anchor_repr: torch.Tensor,
        positive_repr: torch.Tensor,
        negative_repr: torch.Tensor,
        anchor_input_ids: torch.Tensor,
        anchor_attention_mask: torch.Tensor,
        positive_input_ids: torch.Tensor,
        positive_attention_mask: torch.Tensor,
        dynamic_lambda_self: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            anchor_repr: Anchor sparse representations [batch_size, vocab_size]
            positive_repr: Positive sparse representations [batch_size, vocab_size]
            negative_repr: Negative sparse representations [batch_size, vocab_size]
            anchor_input_ids: Anchor token IDs [batch_size, seq_len]
            anchor_attention_mask: Anchor attention mask [batch_size, seq_len]
            positive_input_ids: Positive token IDs [batch_size, seq_len]
            positive_attention_mask: Positive attention mask [batch_size, seq_len]
            dynamic_lambda_self: Optional dynamic weight for self loss

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Use dynamic lambda_self if provided (for curriculum learning)
        lambda_self = dynamic_lambda_self if dynamic_lambda_self else self.lambda_self

        # Compute individual losses
        loss_infonce = self.infonce_loss(anchor_repr, positive_repr, negative_repr)

        loss_self = self.self_loss(
            anchor_repr, anchor_input_ids, anchor_attention_mask
        )

        loss_positive = self.positive_loss(
            anchor_repr, positive_input_ids, positive_attention_mask
        )

        loss_margin = self.margin_loss(anchor_repr, positive_repr, negative_repr)

        loss_flops = self.flops_loss(anchor_repr)

        loss_min_act = self.min_act_loss(anchor_repr)

        # Combine losses
        total_loss = (
            self.lambda_infonce * loss_infonce
            + lambda_self * loss_self
            + self.lambda_positive * loss_positive
            + self.lambda_margin * loss_margin
            + self.lambda_flops * loss_flops
            + self.lambda_min_act * loss_min_act
        )

        # Return loss components for logging
        loss_dict = {
            "total": total_loss.item(),
            "infonce": loss_infonce.item(),
            "self": loss_self.item(),
            "positive": loss_positive.item(),
            "margin": loss_margin.item(),
            "flops": loss_flops.item(),
            "min_act": loss_min_act.item(),
        }

        return total_loss, loss_dict

    def update_temperature(self, temperature: float):
        """Update InfoNCE temperature (for annealing)."""
        self.infonce_loss.temperature = temperature

    def update_weights(
        self,
        lambda_infonce: Optional[float] = None,
        lambda_flops: Optional[float] = None,
        lambda_min_act: Optional[float] = None,
    ):
        """Update loss weights (for curriculum learning)."""
        if lambda_infonce is not None:
            self.lambda_infonce = lambda_infonce
        if lambda_flops is not None:
            self.lambda_flops = lambda_flops
        if lambda_min_act is not None:
            self.lambda_min_act = lambda_min_act
