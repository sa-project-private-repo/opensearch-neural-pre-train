"""
Loss functions for neural sparse retrieval training.

This module implements proper loss functions for training neural sparse models
with inference-free query encoding (IDF-based).

Key differences from naive BCE loss:
- Handles dot product similarities correctly (always positive)
- Implements contrastive learning for proper ranking
- Supports in-batch negatives for efficient training
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def in_batch_negatives_loss(
    doc_sparse: Tensor,
    query_sparse: Tensor,
    temperature: float = 0.05,
    reduction: str = "mean",
) -> Tensor:
    """
    In-batch negatives contrastive loss for neural sparse retrieval.

    Treats all other documents in the batch as negatives. This is the
    RECOMMENDED loss function for inference-free neural sparse models.

    Args:
        doc_sparse: Document sparse vectors (batch_size, vocab_size)
        query_sparse: Query sparse vectors (batch_size, vocab_size)
        temperature: Temperature scaling for similarities (default: 0.05)
        reduction: Loss reduction method ('mean' or 'sum')

    Returns:
        Contrastive loss value

    Example:
        >>> doc_sparse = torch.randn(64, 32000)  # 64 docs
        >>> query_sparse = torch.randn(64, 32000)  # 64 queries
        >>> loss = in_batch_negatives_loss(doc_sparse, query_sparse)
        >>> # Each query treated as positive for its corresponding doc
        >>> # All other 63 docs treated as negatives
    """
    batch_size = doc_sparse.size(0)

    # Compute all-pairs similarity matrix: (batch_size, batch_size)
    # similarities[i, j] = dot(query[i], doc[j])
    similarities = torch.matmul(query_sparse, doc_sparse.T) / temperature

    # Labels: diagonal elements are positive pairs
    # query[i] should match doc[i]
    labels = torch.arange(batch_size, device=doc_sparse.device)

    # Cross-entropy loss with in-batch negatives
    # For each query, compute softmax over all docs in batch
    loss = F.cross_entropy(similarities, labels, reduction=reduction)

    return loss


def margin_ranking_loss(
    doc_sparse: Tensor,
    query_sparse: Tensor,
    relevance: Tensor,
    margin: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Pairwise margin ranking loss for neural sparse retrieval.

    Ensures positive pairs score higher than negative pairs by a margin.
    Works with explicit positive/negative labels.

    Args:
        doc_sparse: Document sparse vectors (batch_size, vocab_size)
        query_sparse: Query sparse vectors (batch_size, vocab_size)
        relevance: Relevance labels (batch_size,) - 1.0 for positive, 0.0 for negative
        margin: Margin to separate positive and negative pairs (default: 1.0)
        reduction: Loss reduction method ('mean' or 'sum')

    Returns:
        Margin ranking loss value

    Example:
        >>> # 4 query-doc pairs: 2 positive, 2 negative
        >>> doc_sparse = torch.randn(4, 32000)
        >>> query_sparse = torch.randn(4, 32000)
        >>> relevance = torch.tensor([1.0, 1.0, 0.0, 0.0])
        >>> loss = margin_ranking_loss(doc_sparse, query_sparse, relevance)
    """
    # Compute similarities
    similarity = torch.sum(doc_sparse * query_sparse, dim=-1)

    # Separate positive and negative pairs
    pos_mask = relevance == 1.0
    neg_mask = relevance == 0.0

    pos_sim = similarity[pos_mask]
    neg_sim = similarity[neg_mask]

    if pos_sim.size(0) == 0 or neg_sim.size(0) == 0:
        # No positive or negative pairs in batch
        return torch.tensor(0.0, device=doc_sparse.device)

    # Pairwise ranking loss: pos_sim > neg_sim + margin
    # Expand to all pairs: (num_pos, num_neg)
    pos_sim_expanded = pos_sim.unsqueeze(1)  # (num_pos, 1)
    neg_sim_expanded = neg_sim.unsqueeze(0)  # (1, num_neg)

    # Loss = max(0, margin - pos_sim + neg_sim)
    pairwise_loss = torch.clamp(
        margin - pos_sim_expanded + neg_sim_expanded, min=0
    )

    if reduction == "mean":
        return pairwise_loss.mean()
    elif reduction == "sum":
        return pairwise_loss.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def contrastive_loss_with_hard_negatives(
    doc_sparse: Tensor,
    query_sparse: Tensor,
    relevance: Tensor,
    temperature: float = 0.05,
    hard_negative_weight: float = 2.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Contrastive loss with weighted hard negatives.

    Combines in-batch negatives with explicit hard negatives (from BM25 etc).
    Hard negatives receive higher weight in the loss.

    Args:
        doc_sparse: Document sparse vectors (batch_size, vocab_size)
        query_sparse: Query sparse vectors (batch_size, vocab_size)
        relevance: Relevance labels (batch_size,) - 1.0 for positive, 0.0 for negative
        temperature: Temperature scaling (default: 0.05)
        hard_negative_weight: Weight multiplier for hard negatives (default: 2.0)
        reduction: Loss reduction method ('mean' or 'sum')

    Returns:
        Weighted contrastive loss

    Example:
        >>> # Batch of 8: 4 positives + 4 hard negatives
        >>> doc_sparse = torch.randn(8, 32000)
        >>> query_sparse = torch.randn(8, 32000)
        >>> relevance = torch.tensor([1., 1., 1., 1., 0., 0., 0., 0.])
        >>> loss = contrastive_loss_with_hard_negatives(
        ...     doc_sparse, query_sparse, relevance
        ... )
    """
    # Compute similarities
    similarity = torch.sum(doc_sparse * query_sparse, dim=-1) / temperature

    # Separate positive and negative
    pos_mask = relevance == 1.0
    neg_mask = relevance == 0.0

    pos_sim = similarity[pos_mask]
    neg_sim = similarity[neg_mask]

    if pos_sim.size(0) == 0:
        return torch.tensor(0.0, device=doc_sparse.device)

    # For each positive, compute log-softmax over positives + negatives
    total_loss = 0.0
    num_pos = pos_sim.size(0)

    for i in range(num_pos):
        # Numerator: exp(pos_sim[i])
        pos_exp = torch.exp(pos_sim[i])

        # Denominator: exp(pos_sim[i]) + weighted_sum(exp(neg_sim))
        # Hard negatives get higher weight
        neg_exp = torch.exp(neg_sim)
        weighted_neg_sum = torch.sum(neg_exp * hard_negative_weight)

        # Log probability
        log_prob = torch.log(pos_exp / (pos_exp + weighted_neg_sum + 1e-10))
        total_loss -= log_prob

    if reduction == "mean":
        return total_loss / num_pos
    elif reduction == "sum":
        return total_loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def neural_sparse_loss_with_regularization(
    doc_sparse: Tensor,
    query_sparse: Tensor,
    relevance: Optional[Tensor] = None,
    idf_dict: Optional[Dict[int, float]] = None,
    lambda_l0: float = 1e-3,
    lambda_idf: float = 1e-2,
    temperature: float = 0.05,
    use_in_batch_negatives: bool = True,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Complete neural sparse loss with ranking + regularization.

    Combines contrastive/ranking loss with sparsity regularization.

    Args:
        doc_sparse: Document sparse vectors (batch_size, vocab_size)
        query_sparse: Query sparse vectors (batch_size, vocab_size)
        relevance: Relevance labels (optional, for explicit negatives)
        idf_dict: IDF dictionary {token_id: idf_score} (optional)
        lambda_l0: L0 regularization weight (default: 1e-3)
        lambda_idf: IDF-aware penalty weight (default: 1e-2)
        temperature: Temperature for contrastive loss (default: 0.05)
        use_in_batch_negatives: Use in-batch negatives (default: True)

    Returns:
        total_loss: Combined loss value
        loss_dict: Dictionary with individual loss components

    Example:
        >>> doc_sparse = torch.randn(32, 32000)
        >>> query_sparse = torch.randn(32, 32000)
        >>> total_loss, losses = neural_sparse_loss_with_regularization(
        ...     doc_sparse, query_sparse
        ... )
        >>> print(losses.keys())  # ['ranking', 'l0', 'idf_penalty', 'total']
    """
    # 1. Ranking/Contrastive Loss
    if use_in_batch_negatives:
        ranking_loss = in_batch_negatives_loss(
            doc_sparse, query_sparse, temperature=temperature
        )
    elif relevance is not None:
        ranking_loss = margin_ranking_loss(
            doc_sparse, query_sparse, relevance
        )
    else:
        raise ValueError(
            "Must provide relevance labels if not using in-batch negatives"
        )

    # 2. L0 Regularization (FLOPS penalty for sparsity)
    # Encourages sparse document representations
    l0_loss = torch.mean(torch.sum(torch.abs(doc_sparse), dim=-1))

    # 3. IDF-aware Penalty (optional)
    # Penalizes activation of low-IDF (common) tokens
    if idf_dict is not None:
        vocab_size = doc_sparse.size(-1)
        idf_tensor = torch.tensor(
            [idf_dict.get(i, 1.0) for i in range(vocab_size)],
            device=doc_sparse.device,
        )
        inverse_idf = 1.0 / (idf_tensor + 1e-6)
        idf_penalty = torch.mean(torch.sum(doc_sparse * inverse_idf, dim=-1))
    else:
        idf_penalty = torch.tensor(0.0, device=doc_sparse.device)

    # Total Loss
    total_loss = (
        ranking_loss + lambda_l0 * l0_loss + lambda_idf * idf_penalty
    )

    loss_dict = {
        "ranking": ranking_loss,
        "l0": l0_loss,
        "idf_penalty": idf_penalty,
        "total": total_loss,
    }

    return total_loss, loss_dict


def compute_sparsity_metrics(sparse_vector: Tensor) -> Dict[str, float]:
    """
    Compute sparsity statistics for sparse vectors.

    Args:
        sparse_vector: Sparse vectors (batch_size, vocab_size)

    Returns:
        Dictionary with sparsity metrics

    Example:
        >>> doc_sparse = torch.randn(32, 32000)
        >>> metrics = compute_sparsity_metrics(doc_sparse)
        >>> print(f"Sparsity: {metrics['sparsity']:.2%}")
    """
    # Non-zero elements
    non_zero = (torch.abs(sparse_vector) > 1e-6).float()
    non_zero_count = torch.sum(non_zero, dim=-1)

    # Sparsity = percentage of zero elements
    vocab_size = sparse_vector.size(-1)
    sparsity = 1.0 - (non_zero_count / vocab_size)

    # L0 norm (count of non-zero elements)
    l0_norm = non_zero_count

    # L1 norm (sum of absolute values)
    l1_norm = torch.sum(torch.abs(sparse_vector), dim=-1)

    return {
        "sparsity": sparsity.mean().item(),
        "sparsity_std": sparsity.std().item(),
        "l0_norm_mean": l0_norm.mean().item(),
        "l0_norm_std": l0_norm.std().item(),
        "l1_norm_mean": l1_norm.mean().item(),
        "l1_norm_std": l1_norm.std().item(),
        "non_zero_count_mean": non_zero_count.mean().item(),
        "non_zero_count_max": non_zero_count.max().item(),
        "non_zero_count_min": non_zero_count.min().item(),
    }
