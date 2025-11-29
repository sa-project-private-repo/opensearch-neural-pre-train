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

    Since student (sparse, vocab_size) and teacher (dense, hidden_size) have
    different dimensions, we use relation-based KD:
    - Teacher's KO-EN similarity should match Student's KO-EN similarity
    - This transfers cross-lingual alignment without dimension matching
    """

    def __init__(
        self,
        temperature: float = 1.0,
        loss_type: str = "relation",
    ):
        """
        Initialize cross-lingual KD loss.

        Args:
            temperature: Temperature for softening distributions
            loss_type: Type of KD loss:
                - 'relation': Match KO-EN similarity (recommended, dimension-free)
                - 'mse_relation': MSE on similarity matrices
        """
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type

    def forward(
        self,
        student_ko: torch.Tensor,
        student_en: torch.Tensor,
        teacher_ko: torch.Tensor,
        teacher_en: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute relation-based KD loss.

        The key insight: Teacher maps KO and EN synonyms to similar vectors.
        We want Student to do the same, without requiring dimension matching.

        Args:
            student_ko: Student Korean sparse rep [batch_size, vocab_size]
            student_en: Student English sparse rep [batch_size, vocab_size]
            teacher_ko: Teacher Korean dense rep [batch_size, hidden_size]
            teacher_en: Teacher English dense rep [batch_size, hidden_size]

        Returns:
            KD loss scalar
        """
        # Normalize all representations
        student_ko = F.normalize(student_ko, p=2, dim=-1)
        student_en = F.normalize(student_en, p=2, dim=-1)
        teacher_ko = F.normalize(teacher_ko, p=2, dim=-1)
        teacher_en = F.normalize(teacher_en, p=2, dim=-1)

        if self.loss_type == "relation":
            # Compute KO-EN similarity for both student and teacher
            # Teacher similarity: how similar are KO-EN in teacher space
            teacher_sim = F.cosine_similarity(teacher_ko, teacher_en, dim=-1)

            # Student similarity: how similar are KO-EN in student space
            student_sim = F.cosine_similarity(student_ko, student_en, dim=-1)

            # Loss: Student's KO-EN similarity should match Teacher's
            # Teacher already has high KO-EN similarity for synonyms
            loss = F.mse_loss(student_sim, teacher_sim)

        elif self.loss_type == "mse_relation":
            # Full similarity matrix matching
            # Teacher: [batch, batch] similarity matrix
            teacher_sim_matrix = torch.mm(teacher_ko, teacher_en.t())

            # Student: [batch, batch] similarity matrix
            student_sim_matrix = torch.mm(student_ko, student_en.t())

            # Scale by temperature
            teacher_sim_matrix = teacher_sim_matrix / self.temperature
            student_sim_matrix = student_sim_matrix / self.temperature

            # MSE between similarity matrices
            loss = F.mse_loss(student_sim_matrix, teacher_sim_matrix)

        elif self.loss_type == "kl_relation":
            # KL divergence on softmaxed similarity distributions
            # Each row becomes a probability distribution over targets
            teacher_sim_matrix = torch.mm(teacher_ko, teacher_en.t())
            student_sim_matrix = torch.mm(student_ko, student_en.t())

            teacher_probs = F.softmax(teacher_sim_matrix / self.temperature, dim=-1)
            student_log_probs = F.log_softmax(
                student_sim_matrix / self.temperature, dim=-1
            )

            loss = F.kl_div(
                student_log_probs, teacher_probs, reduction="batchmean"
            ) * (self.temperature**2)

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


class TokenExpansionLoss(nn.Module):
    """
    Token Expansion Loss for cross-lingual token activation.

    This loss directly encourages Korean input to activate English tokens
    by using the English synonym's activated tokens as targets.

    Key insight: When "머신러닝" is input, we want the model to also
    activate tokens that would be activated by "machine learning".
    """

    def __init__(
        self,
        expansion_type: str = "soft_target",
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        """
        Initialize token expansion loss.

        Args:
            expansion_type: Type of expansion loss:
                - 'soft_target': KL divergence to match EN activation distribution
                - 'hard_target': BCE to activate specific EN tokens
                - 'additive': Add EN activations to KO (union of tokens)
            temperature: Temperature for softening distributions
            top_k: Only expand to top-k EN tokens (None = all)
        """
        super().__init__()
        self.expansion_type = expansion_type
        self.temperature = temperature
        self.top_k = top_k

    def forward(
        self,
        korean_rep: torch.Tensor,
        english_rep: torch.Tensor,
        english_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute token expansion loss.

        The goal: Make korean_rep activate the same tokens as english_rep.

        Args:
            korean_rep: Korean term sparse rep [batch_size, vocab_size]
            english_rep: English synonym sparse rep [batch_size, vocab_size]
            english_token_ids: Optional specific EN token IDs to activate

        Returns:
            Token expansion loss scalar
        """
        if self.expansion_type == "soft_target":
            # Treat english_rep as soft target distribution
            # Korean rep should match this distribution

            # Normalize to get probability distributions
            en_target = F.softmax(english_rep / self.temperature, dim=-1)
            ko_log_pred = F.log_softmax(korean_rep / self.temperature, dim=-1)

            # KL divergence: KO should predict EN's activation pattern
            loss = F.kl_div(
                ko_log_pred, en_target, reduction="batchmean"
            ) * (self.temperature ** 2)

        elif self.expansion_type == "hard_target":
            # Binary cross-entropy: activate EN's activated tokens
            en_activated = (english_rep > 0).float()

            # Optional: only target top-k tokens
            if self.top_k is not None:
                _, top_indices = torch.topk(english_rep, self.top_k, dim=-1)
                mask = torch.zeros_like(english_rep)
                mask.scatter_(1, top_indices, 1.0)
                en_activated = en_activated * mask

            # Sigmoid on korean_rep for BCE
            ko_sigmoid = torch.sigmoid(korean_rep)

            # BCE loss: KO should activate where EN activates
            loss = F.binary_cross_entropy(
                ko_sigmoid, en_activated, reduction="mean"
            )

        elif self.expansion_type == "additive":
            # Directly add EN activations to KO
            # Loss = negative of EN token activations in KO output
            # This encourages KO to have high values where EN has high values

            # Get EN's top activations
            if self.top_k is not None:
                en_values, en_indices = torch.topk(english_rep, self.top_k, dim=-1)
                # Gather KO values at EN's top positions
                ko_at_en_positions = torch.gather(korean_rep, 1, en_indices)
                # Loss: KO should have high values at EN's top positions
                loss = -ko_at_en_positions.mean()
            else:
                # Weight KO by EN activations
                weighted = korean_rep * english_rep
                loss = -weighted.sum(dim=-1).mean()

        elif self.expansion_type == "mse_expansion":
            # MSE between KO and EN representations
            # This directly aligns the activation patterns
            loss = F.mse_loss(korean_rep, english_rep)

        else:
            raise ValueError(f"Unknown expansion_type: {self.expansion_type}")

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


class FrequencyPenaltyLoss(nn.Module):
    """
    Frequency-based penalty loss for noise token suppression.

    This loss penalizes tokens that are activated across ALL inputs in a batch,
    which are likely noise tokens (e.g., 'function', 'operator', 'organization').

    Key insight:
    - Noise tokens: high mean activation, low variance (fire for everything)
    - Meaningful tokens: selective activation, high variance (fire for relevant inputs)

    The loss encourages discriminative token activation by penalizing
    tokens with high mean but low variance.
    """

    def __init__(
        self,
        penalty_type: str = "mean_var_ratio",
        lambda_freq: float = 0.1,
        eps: float = 1e-8,
        top_k_penalty: Optional[int] = None,
    ):
        """
        Initialize frequency penalty loss.

        Args:
            penalty_type: Type of penalty calculation:
                - 'mean_var_ratio': Penalize mean/variance ratio (recommended)
                - 'mean_only': Penalize mean activation only
                - 'low_variance': Penalize low variance tokens
            lambda_freq: Weight for frequency penalty
            eps: Small value to avoid division by zero
            top_k_penalty: Only penalize top-k highest mean tokens (None = all)
        """
        super().__init__()
        self.penalty_type = penalty_type
        self.lambda_freq = lambda_freq
        self.eps = eps
        self.top_k_penalty = top_k_penalty

    def forward(self, sparse_repr: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency penalty loss.

        Args:
            sparse_repr: Sparse representations [batch_size, vocab_size]

        Returns:
            Frequency penalty loss scalar
        """
        # Compute per-token statistics across batch
        mean_activation = sparse_repr.mean(dim=0)  # [vocab_size]
        var_activation = sparse_repr.var(dim=0)    # [vocab_size]

        if self.penalty_type == "mean_var_ratio":
            # Penalize high mean / low variance (noise pattern)
            # High score = likely noise (high everywhere, no discrimination)
            noise_score = mean_activation / (var_activation + self.eps)

            if self.top_k_penalty is not None:
                # Only penalize top-k highest noise scores
                top_k_scores, _ = torch.topk(noise_score, self.top_k_penalty)
                loss = self.lambda_freq * top_k_scores.mean()
            else:
                loss = self.lambda_freq * noise_score.mean()

        elif self.penalty_type == "mean_only":
            # Simple: penalize high average activation
            if self.top_k_penalty is not None:
                top_k_mean, _ = torch.topk(mean_activation, self.top_k_penalty)
                loss = self.lambda_freq * top_k_mean.mean()
            else:
                loss = self.lambda_freq * mean_activation.mean()

        elif self.penalty_type == "low_variance":
            # Encourage high variance (discriminative tokens)
            # Negative because we want to MAXIMIZE variance
            loss = -self.lambda_freq * var_activation.mean()

        else:
            raise ValueError(f"Unknown penalty_type: {self.penalty_type}")

        return loss

    def get_noise_tokens(
        self,
        sparse_repr: torch.Tensor,
        tokenizer,
        top_k: int = 20,
    ) -> list:
        """
        Identify likely noise tokens based on activation patterns.

        Args:
            sparse_repr: Sparse representations [batch_size, vocab_size]
            tokenizer: Tokenizer for decoding token IDs
            top_k: Number of top noise tokens to return

        Returns:
            List of (token, noise_score) tuples
        """
        with torch.no_grad():
            mean_activation = sparse_repr.mean(dim=0)
            var_activation = sparse_repr.var(dim=0)
            noise_score = mean_activation / (var_activation + self.eps)

            top_scores, top_indices = torch.topk(noise_score, top_k)
            tokens = tokenizer.convert_ids_to_tokens(top_indices.cpu().tolist())

            return list(zip(tokens, top_scores.cpu().tolist()))


class ExplicitNoiseTokenLoss(nn.Module):
    """
    Explicit noise token penalty loss.

    This loss directly penalizes activation of known noise tokens.
    Unlike FrequencyPenaltyLoss which uses statistical patterns,
    this approach explicitly specifies tokens to suppress.

    Advantages:
    - More precise control over which tokens to penalize
    - No risk of suppressing useful common tokens
    - Interpretable and predictable behavior
    """

    # Default noise tokens commonly appearing in Top-K
    DEFAULT_NOISE_TOKENS = [
        # Programming/technical terms (not domain-relevant)
        "function", "operator", "operation", "operations",
        "programming", "integration", "organization",
        "implementation", "configuration", "application",
        # Generic terms
        "system", "systems", "process", "processing",
        "method", "methods", "type", "types",
        # Subword noise
        "##ing", "##tion", "##ation", "##ment",
        # Common but uninformative
        "the", "and", "for", "with", "from",
    ]

    def __init__(
        self,
        tokenizer,
        noise_tokens: Optional[list[str]] = None,
        lambda_noise: float = 0.1,
        penalty_type: str = "sum",
    ):
        """
        Initialize explicit noise token loss.

        Args:
            tokenizer: Tokenizer for converting tokens to IDs
            noise_tokens: List of noise tokens to penalize.
                         Uses DEFAULT_NOISE_TOKENS if None.
            lambda_noise: Weight for noise penalty
            penalty_type: Type of penalty calculation:
                - 'sum': Sum of activations for noise tokens
                - 'max': Maximum activation among noise tokens
                - 'softmax': Softmax-weighted penalty (focuses on high activations)
        """
        super().__init__()
        self.lambda_noise = lambda_noise
        self.penalty_type = penalty_type

        # Get noise token IDs
        tokens_to_use = noise_tokens or self.DEFAULT_NOISE_TOKENS
        self.noise_token_ids = []
        self.noise_tokens_found = []

        for token in tokens_to_use:
            # Try exact match first
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:
                self.noise_token_ids.append(token_id)
                self.noise_tokens_found.append(token)

        # Convert to tensor for efficient indexing
        self.register_buffer(
            "noise_indices",
            torch.tensor(self.noise_token_ids, dtype=torch.long)
        )

        print(f"ExplicitNoiseTokenLoss initialized with {len(self.noise_token_ids)} "
              f"noise tokens: {self.noise_tokens_found[:10]}...")

    def forward(self, sparse_repr: torch.Tensor) -> torch.Tensor:
        """
        Compute explicit noise token penalty.

        Args:
            sparse_repr: Sparse representations [batch_size, vocab_size]

        Returns:
            Noise penalty loss scalar
        """
        if len(self.noise_token_ids) == 0:
            return torch.tensor(0.0, device=sparse_repr.device)

        # Extract activations for noise tokens: [batch_size, num_noise_tokens]
        noise_activations = sparse_repr[:, self.noise_indices]

        if self.penalty_type == "sum":
            # Sum of all noise token activations
            loss = noise_activations.sum(dim=-1).mean()

        elif self.penalty_type == "max":
            # Maximum noise token activation per sample
            loss = noise_activations.max(dim=-1).values.mean()

        elif self.penalty_type == "softmax":
            # Softmax-weighted penalty (focuses on high activations)
            weights = F.softmax(noise_activations, dim=-1)
            loss = (weights * noise_activations).sum(dim=-1).mean()

        else:
            raise ValueError(f"Unknown penalty_type: {self.penalty_type}")

        return self.lambda_noise * loss

    def get_noise_activations(
        self,
        sparse_repr: torch.Tensor,
    ) -> dict[str, float]:
        """
        Get activation statistics for noise tokens.

        Args:
            sparse_repr: Sparse representations [batch_size, vocab_size]

        Returns:
            Dictionary mapping noise tokens to their mean activation
        """
        with torch.no_grad():
            noise_activations = sparse_repr[:, self.noise_indices]
            mean_activations = noise_activations.mean(dim=0)

            return {
                token: mean_activations[i].item()
                for i, token in enumerate(self.noise_tokens_found)
            }


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
