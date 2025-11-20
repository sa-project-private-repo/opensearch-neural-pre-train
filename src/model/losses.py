"""Loss functions for SPLADE-doc training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss (InfoNCE) for sparse retrieval.

    Given a query, positive document, and negative documents,
    maximize similarity with positive and minimize with negatives.
    """

    def __init__(self, temperature: float = 0.05):
        """
        Initialize contrastive loss.

        Args:
            temperature: Temperature for softmax scaling
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_repr: torch.Tensor,
        pos_doc_repr: torch.Tensor,
        neg_doc_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            query_repr: Query sparse representations [batch_size, vocab_size]
            pos_doc_repr: Positive document representations [batch_size, vocab_size]
            neg_doc_repr: Negative document representations [batch_size, num_negs, vocab_size]

        Returns:
            Contrastive loss scalar
        """
        batch_size = query_repr.shape[0]
        num_negs = neg_doc_repr.shape[1]

        # Compute similarity scores (dot product for sparse representations)
        # Positive scores: [batch_size]
        pos_scores = (query_repr * pos_doc_repr).sum(dim=1) / self.temperature

        # Negative scores: [batch_size, num_negs]
        neg_scores = torch.bmm(
            neg_doc_repr,  # [batch_size, num_negs, vocab_size]
            query_repr.unsqueeze(-1)  # [batch_size, vocab_size, 1]
        ).squeeze(-1) / self.temperature  # [batch_size, num_negs]

        # Concatenate positive and negative scores
        # [batch_size, 1 + num_negs]
        all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        # Compute log softmax (positive is at index 0)
        log_probs = F.log_softmax(all_scores, dim=1)

        # Loss is negative log probability of positive
        loss = -log_probs[:, 0].mean()

        return loss


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge distillation loss from teacher models.

    Distills knowledge from dense and/or sparse teacher models.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize KD loss.

        Args:
            temperature: Temperature for softening distributions
        """
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(
        self,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between student and teacher distributions.

        Args:
            student_scores: Student similarity scores [batch_size, num_docs]
            teacher_scores: Teacher similarity scores [batch_size, num_docs]

        Returns:
            KD loss scalar
        """
        # Apply temperature scaling
        student_log_probs = F.log_softmax(student_scores / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_scores / self.temperature, dim=1)

        # Compute KL divergence
        loss = self.kl_div(student_log_probs, teacher_probs) * (self.temperature ** 2)

        return loss


class FLOPSRegularization(nn.Module):
    """
    FLOPS regularization for sparsity control.

    Penalizes the number of non-zero activations to enforce sparsity.
    """

    def __init__(self, lambda_flops: float = 1e-4):
        """
        Initialize FLOPS regularization.

        Args:
            lambda_flops: Regularization coefficient
        """
        super().__init__()
        self.lambda_flops = lambda_flops

    def forward(self, sparse_repr: torch.Tensor) -> torch.Tensor:
        """
        Compute FLOPS penalty.

        Args:
            sparse_repr: Sparse representations [batch_size, vocab_size]

        Returns:
            FLOPS penalty scalar
        """
        # L1 norm of sparse representations (sum of absolute values)
        # Encourages sparsity
        flops_penalty = torch.sum(torch.abs(sparse_repr), dim=1).mean()

        return self.lambda_flops * flops_penalty


class IDFAwarePenalty(nn.Module):
    """
    IDF-aware penalty to preserve informative tokens.

    Applies stronger penalty to common tokens (low IDF)
    and weaker penalty to rare tokens (high IDF).
    """

    def __init__(
        self,
        idf_weights: torch.Tensor,
        lambda_idf: float = 1e-3,
    ):
        """
        Initialize IDF-aware penalty.

        Args:
            idf_weights: IDF weights [vocab_size]
            lambda_idf: Regularization coefficient
        """
        super().__init__()
        self.register_buffer('idf_weights', idf_weights)
        self.lambda_idf = lambda_idf

    def forward(self, sparse_repr: torch.Tensor) -> torch.Tensor:
        """
        Compute IDF-aware penalty.

        Args:
            sparse_repr: Sparse representations [batch_size, vocab_size]

        Returns:
            IDF penalty scalar
        """
        # Inverse IDF weighting (penalize low-IDF tokens more)
        # Higher IDF = more informative = lower penalty
        inverse_idf = 1.0 / (self.idf_weights + 1e-8)

        # Weighted L1 penalty
        weighted_penalty = (sparse_repr * inverse_idf.unsqueeze(0)).sum(dim=1).mean()

        return self.lambda_idf * weighted_penalty


class SPLADELoss(nn.Module):
    """
    Combined loss for SPLADE-doc training.

    Combines:
    1. Contrastive loss
    2. Knowledge distillation (optional)
    3. FLOPS regularization
    4. IDF-aware penalty (optional)
    """

    def __init__(
        self,
        temperature: float = 0.05,
        lambda_flops: float = 1e-4,
        lambda_idf: float = 1e-3,
        lambda_kd: float = 0.5,
        idf_weights: Optional[torch.Tensor] = None,
        use_kd: bool = False,
        use_idf_penalty: bool = True,
    ):
        """
        Initialize combined SPLADE loss.

        Args:
            temperature: Temperature for contrastive loss
            lambda_flops: FLOPS regularization coefficient
            lambda_idf: IDF penalty coefficient
            lambda_kd: Knowledge distillation weight
            idf_weights: IDF weights for penalty
            use_kd: Whether to use knowledge distillation
            use_idf_penalty: Whether to use IDF-aware penalty
        """
        super().__init__()

        self.contrastive_loss = ContrastiveLoss(temperature)
        self.flops_reg = FLOPSRegularization(lambda_flops)

        self.use_kd = use_kd
        if use_kd:
            self.kd_loss = KnowledgeDistillationLoss(temperature=1.0)

        self.use_idf_penalty = use_idf_penalty
        if use_idf_penalty and idf_weights is not None:
            self.idf_penalty = IDFAwarePenalty(idf_weights, lambda_idf)
        else:
            self.use_idf_penalty = False

        self.lambda_kd = lambda_kd

    def forward(
        self,
        query_repr: torch.Tensor,
        pos_doc_repr: torch.Tensor,
        neg_doc_repr: torch.Tensor,
        teacher_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            query_repr: Query sparse representations
            pos_doc_repr: Positive document representations
            neg_doc_repr: Negative document representations
            teacher_scores: Optional teacher model scores for KD

        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary of individual loss components
        """
        # Contrastive loss
        contrastive = self.contrastive_loss(query_repr, pos_doc_repr, neg_doc_repr)

        # FLOPS regularization on documents
        flops_pos = self.flops_reg(pos_doc_repr)
        flops_neg = torch.tensor(0.0, device=query_repr.device)
        if neg_doc_repr is not None:
            batch_size, num_negs, vocab_size = neg_doc_repr.shape
            flops_neg = self.flops_reg(
                neg_doc_repr.view(batch_size * num_negs, vocab_size)
            )
        flops_doc = flops_pos + flops_neg

        # Loss dictionary for logging
        loss_dict = {
            'contrastive': contrastive.item(),
            'flops': flops_doc.item(),
        }

        # Collect all loss components
        loss_components = [contrastive, flops_doc]

        # Knowledge distillation
        if self.use_kd and teacher_scores is not None:
            # Compute student scores
            batch_size = query_repr.shape[0]
            num_negs = neg_doc_repr.shape[1]

            pos_scores = (query_repr * pos_doc_repr).sum(dim=1)
            neg_scores = torch.bmm(
                neg_doc_repr,
                query_repr.unsqueeze(-1)
            ).squeeze(-1)
            student_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

            kd = self.kd_loss(student_scores, teacher_scores)
            loss_components.append(self.lambda_kd * kd)
            loss_dict['kd'] = kd.item()

        # IDF-aware penalty
        if self.use_idf_penalty:
            idf_pen_pos = self.idf_penalty(pos_doc_repr)
            idf_pen_neg = torch.tensor(0.0, device=query_repr.device)
            if neg_doc_repr is not None:
                batch_size, num_negs, vocab_size = neg_doc_repr.shape
                idf_pen_neg = self.idf_penalty(
                    neg_doc_repr.view(batch_size * num_negs, vocab_size)
                )
            idf_pen = idf_pen_pos + idf_pen_neg
            loss_components.append(idf_pen)
            loss_dict['idf_penalty'] = idf_pen.item()

        # Sum all loss components (avoid inplace operations)
        total_loss = sum(loss_components)
        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict
