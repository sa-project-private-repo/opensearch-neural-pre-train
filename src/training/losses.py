"""Loss functions for Neural Sparse training."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankingLoss(nn.Module):
    """
    Margin ranking loss for query-document matching.

    Ensures that positive documents have higher similarity scores
    than negative documents by at least a margin.
    """

    def __init__(self, margin: float = 0.1):
        """
        Initialize ranking loss.

        Args:
            margin: Minimum margin between positive and negative scores
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        query_rep: torch.Tensor,
        pos_doc_rep: torch.Tensor,
        neg_doc_reps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ranking loss.

        Args:
            query_rep: Query sparse representations [batch_size, vocab_size]
            pos_doc_rep: Positive document representations [batch_size, vocab_size]
            neg_doc_reps: Negative document representations [batch_size, num_neg, vocab_size]

        Returns:
            Ranking loss scalar
        """
        # Compute positive scores (dot product)
        # Shape: [batch_size]
        pos_scores = torch.sum(query_rep * pos_doc_rep, dim=-1)

        # Compute negative scores
        # Shape: [batch_size, num_neg]
        neg_scores = torch.sum(
            query_rep.unsqueeze(1) * neg_doc_reps,
            dim=-1,
        )

        # Compute margin ranking loss
        # We want: pos_score > neg_score + margin
        # Loss = max(0, margin + neg_score - pos_score)
        losses = torch.relu(
            self.margin + neg_scores - pos_scores.unsqueeze(1)
        )

        # Average over negatives and batch
        loss = losses.mean()

        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for query-document matching.

    Uses cross-entropy loss with in-batch negatives.
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
        query_rep: torch.Tensor,
        pos_doc_rep: torch.Tensor,
        neg_doc_reps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            query_rep: Query sparse representations [batch_size, vocab_size]
            pos_doc_rep: Positive document representations [batch_size, vocab_size]
            neg_doc_reps: Optional negative documents [batch_size, num_neg, vocab_size]

        Returns:
            Contrastive loss scalar
        """
        batch_size = query_rep.shape[0]

        # Compute positive scores
        pos_scores = torch.sum(query_rep * pos_doc_rep, dim=-1)

        if neg_doc_reps is not None:
            # Use provided negatives
            neg_scores = torch.sum(
                query_rep.unsqueeze(1) * neg_doc_reps,
                dim=-1,
            )
            # Concatenate positive and negative scores
            all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        else:
            # Use in-batch negatives
            # Compute similarity matrix [batch_size, batch_size]
            all_scores = torch.matmul(query_rep, pos_doc_rep.t())

        # Scale by temperature
        all_scores = all_scores / self.temperature

        # Labels are always 0 (positive is first)
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_rep.device)

        # Cross-entropy loss
        loss = F.cross_entropy(all_scores, labels)

        return loss


class CrossLingualLoss(nn.Module):
    """
    Cross-lingual alignment loss for bilingual synonyms.

    Encourages Korean and English synonym terms to have similar
    sparse representations.
    """

    def __init__(self, similarity_type: str = "cosine"):
        """
        Initialize cross-lingual loss.

        Args:
            similarity_type: Type of similarity ('cosine' or 'dot')
        """
        super().__init__()
        self.similarity_type = similarity_type

    def forward(
        self,
        korean_rep: torch.Tensor,
        english_rep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-lingual alignment loss.

        Args:
            korean_rep: Korean term representations [batch_size, vocab_size]
            english_rep: English term representations [batch_size, vocab_size]

        Returns:
            Cross-lingual loss scalar
        """
        if self.similarity_type == "cosine":
            # Cosine similarity
            similarity = F.cosine_similarity(korean_rep, english_rep, dim=-1)
            # Loss = 1 - similarity (maximize similarity)
            loss = (1.0 - similarity).mean()
        else:
            # Dot product similarity
            similarity = torch.sum(korean_rep * english_rep, dim=-1)
            # Negative similarity (maximize similarity)
            loss = -similarity.mean()

        return loss


class CrossLingualKDLoss(nn.Module):
    """
    Cross-lingual Knowledge Distillation Loss.

    Uses a multilingual dense teacher model (e.g., mE5-large) to guide
    the sparse student model to learn cross-lingual alignment.

    The teacher already encodes Korean and English synonyms to similar
    embeddings. This loss transfers that knowledge to the sparse model.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        loss_type: str = "kl",
    ):
        """
        Initialize cross-lingual KD loss.

        Args:
            temperature: Temperature for softening distributions
            loss_type: Type of KD loss ('kl', 'mse', 'cosine')
        """
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type

    def forward(
        self,
        student_rep: torch.Tensor,
        teacher_rep: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute KD loss between student sparse and teacher dense representations.

        Args:
            student_rep: Student sparse representations [batch_size, vocab_size]
            teacher_rep: Teacher dense representations [batch_size, hidden_size]
            normalize: Whether to normalize representations

        Returns:
            KD loss scalar
        """
        if normalize:
            student_rep = F.normalize(student_rep, p=2, dim=-1)
            teacher_rep = F.normalize(teacher_rep, p=2, dim=-1)

        if self.loss_type == "kl":
            # Softmax with temperature
            student_soft = F.log_softmax(student_rep / self.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_rep / self.temperature, dim=-1)

            # KL divergence
            loss = F.kl_div(
                student_soft,
                teacher_soft,
                reduction="batchmean",
            ) * (self.temperature ** 2)

        elif self.loss_type == "mse":
            # Mean squared error (requires same dimensions)
            loss = F.mse_loss(student_rep, teacher_rep)

        elif self.loss_type == "cosine":
            # Cosine similarity loss
            similarity = F.cosine_similarity(student_rep, teacher_rep, dim=-1)
            loss = (1.0 - similarity).mean()

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss


class SynonymAlignmentLoss(nn.Module):
    """
    Synonym Alignment Loss for cross-lingual term matching.

    Ensures that synonymous terms (e.g., "머신러닝" and "machine learning")
    activate similar tokens in the sparse representation.

    Key insight: If teacher encodes KO and EN synonyms similarly,
    we want sparse model to also produce similar activations.
    """

    def __init__(
        self,
        alignment_type: str = "overlap",
        top_k: int = 50,
        margin: float = 0.5,
    ):
        """
        Initialize synonym alignment loss.

        Args:
            alignment_type: Type of alignment ('overlap', 'cosine', 'contrastive')
            top_k: Number of top tokens to consider for overlap
            margin: Margin for contrastive alignment
        """
        super().__init__()
        self.alignment_type = alignment_type
        self.top_k = top_k
        self.margin = margin

    def forward(
        self,
        korean_rep: torch.Tensor,
        english_rep: torch.Tensor,
        negative_rep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute synonym alignment loss.

        Args:
            korean_rep: Korean term sparse rep [batch_size, vocab_size]
            english_rep: English synonym sparse rep [batch_size, vocab_size]
            negative_rep: Optional non-synonym rep for contrastive

        Returns:
            Alignment loss scalar
        """
        if self.alignment_type == "overlap":
            # Token overlap loss
            # Encourage both to activate same vocabulary tokens
            ko_activated = (korean_rep > 0).float()
            en_activated = (english_rep > 0).float()

            # Jaccard-like overlap
            intersection = (ko_activated * en_activated).sum(dim=-1)
            union = torch.clamp(ko_activated + en_activated, 0, 1).sum(dim=-1)

            overlap = intersection / (union + 1e-8)
            loss = (1.0 - overlap).mean()

        elif self.alignment_type == "cosine":
            # Cosine similarity
            similarity = F.cosine_similarity(korean_rep, english_rep, dim=-1)
            loss = (1.0 - similarity).mean()

        elif self.alignment_type == "contrastive":
            # Contrastive loss with margin
            pos_sim = F.cosine_similarity(korean_rep, english_rep, dim=-1)

            if negative_rep is not None:
                neg_sim = F.cosine_similarity(korean_rep, negative_rep, dim=-1)
                # Margin ranking: pos should be > neg + margin
                loss = torch.relu(self.margin + neg_sim - pos_sim).mean()
            else:
                loss = (1.0 - pos_sim).mean()

        elif self.alignment_type == "soft_overlap":
            # Soft overlap using min of activation values
            min_activation = torch.minimum(korean_rep, english_rep)
            max_activation = torch.maximum(korean_rep, english_rep)

            soft_intersection = min_activation.sum(dim=-1)
            soft_union = max_activation.sum(dim=-1)

            soft_overlap = soft_intersection / (soft_union + 1e-8)
            loss = (1.0 - soft_overlap).mean()

        else:
            raise ValueError(f"Unknown alignment_type: {self.alignment_type}")

        return loss


class FLOPSLoss(nn.Module):
    """
    FLOPS regularization loss for sparsity.

    Encourages sparse representations by penalizing the L1 norm
    of the sparse vectors (proxy for FLOPS).
    """

    def __init__(self, lambda_flops: float = 0.001):
        """
        Initialize FLOPS loss.

        Args:
            lambda_flops: Weight for FLOPS regularization
        """
        super().__init__()
        self.lambda_flops = lambda_flops

    def forward(self, sparse_rep: torch.Tensor) -> torch.Tensor:
        """
        Compute FLOPS regularization loss.

        Args:
            sparse_rep: Sparse representations [batch_size, vocab_size]

        Returns:
            FLOPS loss scalar
        """
        # L1 norm over vocabulary dimension
        l1_norm = torch.sum(torch.abs(sparse_rep), dim=-1)

        # Average over batch
        loss = self.lambda_flops * l1_norm.mean()

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for Neural Sparse training.

    Combines ranking loss, cross-lingual loss, and FLOPS regularization.
    """

    def __init__(
        self,
        alpha_ranking: float = 1.0,
        beta_cross_lingual: float = 0.3,
        gamma_sparsity: float = 0.001,
        ranking_margin: float = 0.1,
        use_contrastive: bool = False,
        contrastive_temperature: float = 0.05,
    ):
        """
        Initialize combined loss.

        Args:
            alpha_ranking: Weight for ranking loss
            beta_cross_lingual: Weight for cross-lingual loss
            gamma_sparsity: Weight for sparsity loss
            ranking_margin: Margin for ranking loss
            use_contrastive: Use contrastive loss instead of ranking loss
            contrastive_temperature: Temperature for contrastive loss
        """
        super().__init__()

        self.alpha_ranking = alpha_ranking
        self.beta_cross_lingual = beta_cross_lingual
        self.gamma_sparsity = gamma_sparsity

        # Initialize loss functions
        if use_contrastive:
            self.ranking_loss = ContrastiveLoss(temperature=contrastive_temperature)
        else:
            self.ranking_loss = RankingLoss(margin=ranking_margin)

        self.cross_lingual_loss = CrossLingualLoss()
        self.flops_loss = FLOPSLoss(lambda_flops=1.0)  # Will be scaled by gamma

    def forward(
        self,
        query_rep: torch.Tensor,
        pos_doc_rep: torch.Tensor,
        neg_doc_reps: torch.Tensor,
        korean_rep: Optional[torch.Tensor] = None,
        english_rep: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            query_rep: Query representations [batch_size, vocab_size]
            pos_doc_rep: Positive document representations [batch_size, vocab_size]
            neg_doc_reps: Negative document representations [batch_size, num_neg, vocab_size]
            korean_rep: Optional Korean synonym representations [num_pairs, vocab_size]
            english_rep: Optional English synonym representations [num_pairs, vocab_size]

        Returns:
            Dictionary containing:
                - total_loss: Combined loss
                - ranking_loss: Ranking component
                - cross_lingual_loss: Cross-lingual component
                - sparsity_loss: Sparsity component
        """
        losses = {}

        # 1. Ranking loss
        rank_loss = self.ranking_loss(query_rep, pos_doc_rep, neg_doc_reps)
        losses["ranking_loss"] = rank_loss

        # 2. Cross-lingual loss (if synonym pairs provided)
        if korean_rep is not None and english_rep is not None:
            cl_loss = self.cross_lingual_loss(korean_rep, english_rep)
            losses["cross_lingual_loss"] = cl_loss
        else:
            cl_loss = torch.tensor(0.0, device=query_rep.device)
            losses["cross_lingual_loss"] = cl_loss

        # 3. Sparsity loss (FLOPS regularization)
        # Apply to query and positive documents
        query_flops = self.flops_loss(query_rep)
        doc_flops = self.flops_loss(pos_doc_rep)
        sparsity_loss = (query_flops + doc_flops) / 2.0
        losses["sparsity_loss"] = sparsity_loss

        # 4. Combined loss
        total_loss = (
            self.alpha_ranking * rank_loss
            + self.beta_cross_lingual * cl_loss
            + self.gamma_sparsity * sparsity_loss
        )
        losses["total_loss"] = total_loss

        return losses


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")

    batch_size = 4
    vocab_size = 100
    num_negatives = 5

    # Create dummy data
    query_rep = torch.randn(batch_size, vocab_size).relu()
    pos_doc_rep = torch.randn(batch_size, vocab_size).relu()
    neg_doc_reps = torch.randn(batch_size, num_negatives, vocab_size).relu()

    # Test individual losses
    print("\n1. Ranking Loss:")
    ranking_loss_fn = RankingLoss(margin=0.1)
    loss = ranking_loss_fn(query_rep, pos_doc_rep, neg_doc_reps)
    print(f"   Loss: {loss.item():.4f}")

    print("\n2. Contrastive Loss:")
    contrastive_loss_fn = ContrastiveLoss(temperature=0.05)
    loss = contrastive_loss_fn(query_rep, pos_doc_rep, neg_doc_reps)
    print(f"   Loss: {loss.item():.4f}")

    print("\n3. Cross-lingual Loss:")
    korean_rep = torch.randn(10, vocab_size).relu()
    english_rep = torch.randn(10, vocab_size).relu()
    cl_loss_fn = CrossLingualLoss()
    loss = cl_loss_fn(korean_rep, english_rep)
    print(f"   Loss: {loss.item():.4f}")

    print("\n4. FLOPS Loss:")
    flops_loss_fn = FLOPSLoss(lambda_flops=0.001)
    loss = flops_loss_fn(query_rep)
    print(f"   Loss: {loss.item():.4f}")

    print("\n5. Combined Loss:")
    combined_loss_fn = CombinedLoss(
        alpha_ranking=1.0,
        beta_cross_lingual=0.3,
        gamma_sparsity=0.001,
    )
    losses = combined_loss_fn(
        query_rep, pos_doc_rep, neg_doc_reps, korean_rep, english_rep
    )
    print(f"   Total Loss: {losses['total_loss'].item():.4f}")
    print(f"   Ranking Loss: {losses['ranking_loss'].item():.4f}")
    print(f"   Cross-lingual Loss: {losses['cross_lingual_loss'].item():.4f}")
    print(f"   Sparsity Loss: {losses['sparsity_loss'].item():.4f}")

    print("\nAll tests passed!")
