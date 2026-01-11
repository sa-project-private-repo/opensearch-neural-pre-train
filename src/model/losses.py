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


class IDFAwareFLOPSLoss(nn.Module):
    """
    IDF-aware FLOPS regularization loss.

    Applies higher penalty to low-IDF tokens (stopwords),
    lower penalty to high-IDF tokens (informative terms).

    Formula: L = Σ(w_j * |mean_act_j|) + β * Σ(w_j * mean_act_j)²
    where w_j = exp(-α * normalized_idf_j)

    This addresses the issue where uniform FLOPS penalty suppresses
    informative high-IDF tokens while under-penalizing common stopwords.
    """

    def __init__(
        self,
        vocab_size: int,
        idf_weights: Optional[torch.Tensor] = None,
        alpha: float = 2.0,
        beta: float = 0.3,
        use_exponential: bool = True,
    ):
        """
        Initialize IDF-aware FLOPS loss.

        Args:
            vocab_size: Vocabulary size
            idf_weights: Pre-computed IDF weights [vocab_size]
            alpha: Exponential scaling factor (higher = sharper penalty difference)
            beta: L2 penalty weight (L1 weight is implicitly 1.0)
            use_exponential: Use exp(-alpha*idf) vs linear (1-idf)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.beta = beta
        self.use_exponential = use_exponential

        # Initialize penalty weights
        if idf_weights is not None:
            self._set_idf_weights(idf_weights)
        else:
            # Default: uniform weights (falls back to standard FLOPS)
            self.register_buffer(
                "penalty_weights",
                torch.ones(vocab_size),
            )

    def _set_idf_weights(self, idf_weights: torch.Tensor):
        """Compute and set penalty weights from IDF values."""
        # Normalize IDF to [0, 1]
        idf_min = idf_weights.min()
        idf_max = idf_weights.max()
        idf_normalized = (idf_weights - idf_min) / (idf_max - idf_min + 1e-8)

        # Compute penalty weights
        # High IDF (informative) -> low penalty
        # Low IDF (stopwords) -> high penalty
        if self.use_exponential:
            # Exponential: w = exp(-alpha * idf_normalized)
            penalty_weights = torch.exp(-self.alpha * idf_normalized)
        else:
            # Linear: w = 1 - idf_normalized
            penalty_weights = 1.0 - idf_normalized

        self.register_buffer("penalty_weights", penalty_weights)

    def forward(self, sparse_repr: torch.Tensor) -> torch.Tensor:
        """
        Compute IDF-aware FLOPS loss.

        Args:
            sparse_repr: Sparse representations [batch_size, vocab_size]

        Returns:
            Scalar loss value
        """
        # Mean activation per vocabulary token across batch
        mean_activation = sparse_repr.mean(dim=0)  # [vocab_size]

        # Weighted L1 penalty (promotes exact sparsity)
        weighted_l1 = (self.penalty_weights * mean_activation.abs()).sum()

        # Weighted L2 penalty (smooth gradients)
        weighted_l2 = ((self.penalty_weights * mean_activation) ** 2).sum()

        # Hybrid loss
        loss = weighted_l1 + self.beta * weighted_l2

        return loss

    def update_idf_weights(self, idf_weights: torch.Tensor):
        """Update IDF weights (e.g., after corpus change)."""
        self._set_idf_weights(idf_weights)

    @staticmethod
    def compute_idf_from_corpus(
        corpus: list,
        tokenizer,
        smoothing: str = "bm25",
    ) -> torch.Tensor:
        """
        Compute IDF weights from a corpus.

        Args:
            corpus: List of documents
            tokenizer: HuggingFace tokenizer
            smoothing: IDF smoothing method ("bm25" or "standard")

        Returns:
            IDF weights tensor [vocab_size]
        """
        from collections import Counter

        vocab_size = tokenizer.vocab_size
        df = Counter()  # Document frequency
        N = len(corpus)

        # Count document frequencies
        for doc in corpus:
            tokens = tokenizer.encode(doc, add_special_tokens=False)
            unique_tokens = set(tokens)
            for token_id in unique_tokens:
                df[token_id] += 1

        # Compute IDF
        idf = torch.zeros(vocab_size)
        for token_id in range(vocab_size):
            doc_freq = df.get(token_id, 0)

            if smoothing == "bm25":
                # BM25-style: log(1 + (N - df + 0.5) / (df + 0.5))
                idf[token_id] = torch.log(
                    torch.tensor(1.0 + (N - doc_freq + 0.5) / (doc_freq + 0.5))
                )
            else:
                # Standard: log(N / (df + 1))
                idf[token_id] = torch.log(
                    torch.tensor(N / (doc_freq + 1.0))
                )

        return idf


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge distillation loss from dense teacher to sparse student.

    Uses KL divergence with temperature scaling:
    L_KD = T² * KL(softmax(z_teacher/T) || softmax(z_student/T))
         + α_mse * MSE(z_norm_teacher, z_norm_student)

    The T² factor compensates for gradient magnitude reduction.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha_kl: float = 0.7,
        alpha_mse: float = 0.3,
    ):
        """
        Initialize knowledge distillation loss.

        Args:
            temperature: Softmax temperature (higher = softer distributions)
            alpha_kl: Weight for KL divergence loss
            alpha_mse: Weight for MSE auxiliary loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha_kl = alpha_kl
        self.alpha_mse = alpha_mse

    def forward(
        self,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.

        Args:
            student_scores: Student similarity scores [batch, batch] or [batch, num_docs]
            teacher_scores: Teacher similarity scores [batch, batch] or [batch, num_docs]

        Returns:
            Scalar loss value
        """
        # Temperature-scaled softmax
        student_log_probs = F.log_softmax(
            student_scores / self.temperature, dim=-1
        )
        teacher_probs = F.softmax(
            teacher_scores / self.temperature, dim=-1
        )

        # KL divergence with T² scaling
        kl_loss = (self.temperature ** 2) * F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        )

        # MSE on z-score normalized scores (scale-invariant)
        student_norm = self._z_normalize(student_scores)
        teacher_norm = self._z_normalize(teacher_scores)
        mse_loss = F.mse_loss(student_norm, teacher_norm)

        # Combined loss
        loss = self.alpha_kl * kl_loss + self.alpha_mse * mse_loss

        return loss

    def _z_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Z-score normalization for scale invariance."""
        mean = x.mean()
        std = x.std() + 1e-8
        return (x - mean) / std

    def update_temperature(self, temperature: float):
        """Update distillation temperature."""
        self.temperature = temperature


class DenseTeacherScorer(nn.Module):
    """
    Dense encoder teacher for knowledge distillation.

    Uses sentence-transformers or similar dense models to compute
    teacher similarity scores for sparse student training.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize dense teacher scorer.

        Args:
            model_name: HuggingFace model name for dense encoder
            device: Device to run model on
        """
        super().__init__()
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy load to avoid import issues
        self._encoder = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load the model."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)
                self._encoder = self._encoder.to(self.device)
                self._encoder.eval()
            except ImportError:
                # Fallback to transformers
                from transformers import AutoModel, AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._encoder = AutoModel.from_pretrained(self.model_name)
                self._encoder = self._encoder.to(self.device)
                self._encoder.eval()

    @torch.no_grad()
    def encode(self, texts: list) -> torch.Tensor:
        """
        Encode texts to dense embeddings.

        Args:
            texts: List of texts to encode

        Returns:
            Dense embeddings [batch_size, hidden_dim]
        """
        self._load_model()

        try:
            # sentence-transformers path
            embeddings = self._encoder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
        except AttributeError:
            # transformers fallback
            inputs = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            outputs = self._encoder(**inputs)
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            embeddings = (
                outputs.last_hidden_state * attention_mask.unsqueeze(-1)
            ).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        return embeddings.to(self.device)

    @torch.no_grad()
    def compute_scores(
        self,
        queries: list,
        documents: list,
    ) -> torch.Tensor:
        """
        Compute similarity matrix between queries and documents.

        Args:
            queries: List of query texts
            documents: List of document texts

        Returns:
            Similarity matrix [num_queries, num_documents]
        """
        query_embeddings = self.encode(queries)
        doc_embeddings = self.encode(documents)

        # Normalize for cosine similarity
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=-1)

        # Compute similarity matrix
        scores = torch.mm(query_embeddings, doc_embeddings.t())

        return scores


class SPLADELossV23(nn.Module):
    """
    Enhanced SPLADE loss v23 with expert-recommended improvements.

    Key improvements over v22:
    - IDF-aware FLOPS penalty (preserves informative tokens)
    - Knowledge distillation from dense teacher
    - Fixed triplet margin (0.3 instead of 1.5)
    - Balanced loss weights

    Loss components:
    - InfoNCE contrastive loss
    - Self-reconstruction loss
    - Positive activation loss
    - Triplet margin loss (optional, disabled by default)
    - IDF-aware FLOPS regularization
    - Minimum activation loss
    - Knowledge distillation loss (optional)
    """

    def __init__(
        self,
        # Loss weights (expert-recommended defaults)
        lambda_infonce: float = 2.5,
        lambda_self: float = 1.0,
        lambda_positive: float = 3.0,
        lambda_margin: float = 0.0,  # Disabled by default
        lambda_flops: float = 0.003,
        lambda_min_act: float = 1.0,
        lambda_kd: float = 1.0,
        # Hyperparameters
        temperature: float = 0.07,
        margin: float = 0.3,  # Fixed from 1.5
        top_k: int = 5,
        min_activation: float = 0.5,
        kd_temperature: float = 2.0,
        # IDF configuration
        vocab_size: int = 50000,
        idf_weights: Optional[torch.Tensor] = None,
        idf_alpha: float = 2.0,
        # Teacher model for KD
        teacher_model: Optional[DenseTeacherScorer] = None,
    ):
        """
        Initialize SPLADE v23 loss.

        Args:
            lambda_*: Loss component weights
            temperature: InfoNCE temperature
            margin: Triplet margin (reduced from 1.5 to 0.3)
            top_k: Top-k for minimum activation
            min_activation: Minimum activation threshold
            kd_temperature: Knowledge distillation temperature
            vocab_size: Vocabulary size for IDF-aware FLOPS
            idf_weights: Pre-computed IDF weights
            idf_alpha: IDF exponential scaling factor
            teacher_model: Dense teacher for knowledge distillation
        """
        super().__init__()

        # Loss weights
        self.lambda_infonce = lambda_infonce
        self.lambda_self = lambda_self
        self.lambda_positive = lambda_positive
        self.lambda_margin = lambda_margin
        self.lambda_flops = lambda_flops
        self.lambda_min_act = lambda_min_act
        self.lambda_kd = lambda_kd

        # Individual loss modules
        self.infonce_loss = InfoNCELoss(temperature=temperature)
        self.self_loss = SelfReconstructionLoss()
        self.positive_loss = PositiveActivationLoss()
        self.margin_loss = TripletMarginLoss(margin=margin)
        self.flops_loss = IDFAwareFLOPSLoss(
            vocab_size=vocab_size,
            idf_weights=idf_weights,
            alpha=idf_alpha,
        )
        self.min_act_loss = MinimumActivationLoss(
            top_k=top_k,
            min_activation=min_activation,
        )
        self.kd_loss = KnowledgeDistillationLoss(temperature=kd_temperature)

        # Teacher model (optional)
        self.teacher_model = teacher_model

    def forward(
        self,
        anchor_repr: torch.Tensor,
        positive_repr: torch.Tensor,
        negative_repr: torch.Tensor,
        anchor_input_ids: torch.Tensor,
        anchor_attention_mask: torch.Tensor,
        positive_input_ids: torch.Tensor,
        positive_attention_mask: torch.Tensor,
        # Optional: for knowledge distillation
        anchor_texts: Optional[list] = None,
        positive_texts: Optional[list] = None,
        teacher_scores: Optional[torch.Tensor] = None,
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
            anchor_texts: Raw anchor texts for teacher scoring
            positive_texts: Raw positive texts for teacher scoring
            teacher_scores: Pre-computed teacher scores (if available)

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
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

        # Knowledge distillation (if teacher available)
        loss_kd = torch.tensor(0.0, device=anchor_repr.device)
        if self.lambda_kd > 0 and (teacher_scores is not None or self.teacher_model is not None):
            # Compute student scores
            anchor_norm = F.normalize(anchor_repr, p=2, dim=-1)
            positive_norm = F.normalize(positive_repr, p=2, dim=-1)
            student_scores = torch.mm(anchor_norm, positive_norm.t())

            # Get teacher scores
            if teacher_scores is None and self.teacher_model is not None:
                if anchor_texts is not None and positive_texts is not None:
                    teacher_scores = self.teacher_model.compute_scores(
                        anchor_texts, positive_texts
                    )

            if teacher_scores is not None:
                loss_kd = self.kd_loss(student_scores, teacher_scores)

        # Combine losses
        total_loss = (
            self.lambda_infonce * loss_infonce
            + self.lambda_self * loss_self
            + self.lambda_positive * loss_positive
            + self.lambda_margin * loss_margin
            + self.lambda_flops * loss_flops
            + self.lambda_min_act * loss_min_act
            + self.lambda_kd * loss_kd
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
            "kd": loss_kd.item() if isinstance(loss_kd, torch.Tensor) else loss_kd,
        }

        return total_loss, loss_dict

    def update_temperature(self, temperature: float):
        """Update InfoNCE temperature."""
        self.infonce_loss.temperature = temperature

    def update_kd_temperature(self, temperature: float):
        """Update knowledge distillation temperature."""
        self.kd_loss.update_temperature(temperature)

    def update_weights(
        self,
        lambda_infonce: Optional[float] = None,
        lambda_flops: Optional[float] = None,
        lambda_min_act: Optional[float] = None,
        lambda_kd: Optional[float] = None,
    ):
        """Update loss weights (for curriculum learning)."""
        if lambda_infonce is not None:
            self.lambda_infonce = lambda_infonce
        if lambda_flops is not None:
            self.lambda_flops = lambda_flops
        if lambda_min_act is not None:
            self.lambda_min_act = lambda_min_act
        if lambda_kd is not None:
            self.lambda_kd = lambda_kd

    def update_idf_weights(self, idf_weights: torch.Tensor):
        """Update IDF weights for FLOPS loss."""
        self.flops_loss.update_idf_weights(idf_weights)
