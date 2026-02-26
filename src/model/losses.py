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

    V26 Enhancement: Special tokens (<s>, </s>) are excluded from IDF
    normalization range and receive a fixed high penalty to prevent
    their activation from dominating sparse representations.
    """

    def __init__(
        self,
        vocab_size: int,
        idf_weights: Optional[torch.Tensor] = None,
        alpha: float = 2.0,
        beta: float = 0.3,
        use_exponential: bool = True,
        special_token_ids: Optional[set] = None,
        special_penalty: float = 1.0,
    ):
        """
        Initialize IDF-aware FLOPS loss.

        Args:
            vocab_size: Vocabulary size
            idf_weights: Pre-computed IDF weights [vocab_size]
            alpha: Exponential scaling factor (higher = sharper penalty difference)
            beta: L2 penalty weight (L1 weight is implicitly 1.0)
            use_exponential: Use exp(-alpha*idf) vs linear (1-idf)
            special_token_ids: Set of special token IDs to exclude from normalization
            special_penalty: Fixed penalty for special tokens (default 1.0, V26 uses 100.0)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.beta = beta
        self.use_exponential = use_exponential
        self.special_token_ids = special_token_ids or set()
        self.special_penalty = special_penalty

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
        """
        Compute and set penalty weights from IDF values.

        V26 Enhancement: Excludes special tokens from IDF min/max calculation
        to prevent them from compressing the normalization range.
        """
        # Create mask for real tokens (excluding special tokens)
        if self.special_token_ids:
            real_mask = torch.ones(len(idf_weights), dtype=torch.bool)
            for tid in self.special_token_ids:
                if 0 <= tid < len(idf_weights):
                    real_mask[tid] = False
            real_idf = idf_weights[real_mask]
            idf_min = real_idf.min()
            idf_max = real_idf.max()
        else:
            idf_min = idf_weights.min()
            idf_max = idf_weights.max()

        # Normalize IDF to [0, 1] using real token range
        idf_normalized = (idf_weights - idf_min) / (idf_max - idf_min + 1e-8)
        idf_normalized = idf_normalized.clamp(0, 1)

        # Compute penalty weights
        # High IDF (informative) -> low penalty
        # Low IDF (stopwords) -> high penalty
        if self.use_exponential:
            # Exponential: w = exp(-alpha * idf_normalized)
            penalty_weights = torch.exp(-self.alpha * idf_normalized)
        else:
            # Linear: w = 1 - idf_normalized
            penalty_weights = 1.0 - idf_normalized

        # Apply fixed penalty to special tokens (V26)
        if self.special_token_ids and self.special_penalty > 1.0:
            for tid in self.special_token_ids:
                if 0 <= tid < len(penalty_weights):
                    penalty_weights[tid] = self.special_penalty

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
        # Formula: Σ(w_j * mean_act_j²) - weight applied to squared activation
        weighted_l2 = (self.penalty_weights * (mean_activation ** 2)).sum()

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
                self._encoder = SentenceTransformer(
                    self.model_name, device=self.device
                )
                # Cap seq length to avoid 8192-token overhead
                # (training texts are max ~192 tokens)
                self._encoder.max_seq_length = 256
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
                batch_size=64,
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


class SPLADELossV25(nn.Module):
    """
    SPLADE loss v25 with mandatory IDF-aware FLOPS and stopword masking.

    Key differences from v23/v24:
    - IDF weights are REQUIRED (not optional)
    - Stopword mask support for hard masking of grammatical tokens
    - Semantic token ratio logging for training monitoring

    This ensures proper frequency-based discrimination for large
    vocabularies (250K XLM-RoBERTa) where Korean particles and
    common words would otherwise dominate sparse representations.
    """

    def __init__(
        self,
        idf_weights: torch.Tensor,
        # Loss weights
        lambda_infonce: float = 3.0,
        lambda_self: float = 0.5,
        lambda_positive: float = 2.0,
        lambda_margin: float = 0.0,
        lambda_flops: float = 0.002,
        lambda_min_act: float = 1.0,
        lambda_kd: float = 2.0,
        # Hyperparameters
        temperature: float = 0.07,
        margin: float = 0.3,
        top_k: int = 5,
        min_activation: float = 0.5,
        kd_temperature: float = 3.0,
        # IDF configuration
        idf_alpha: float = 2.5,
        # Stopword masking
        stopword_mask: Optional[torch.Tensor] = None,
        stopword_penalty: float = 5.0,
        # Teacher model
        teacher_model: Optional[DenseTeacherScorer] = None,
    ):
        """
        Initialize SPLADE v25 loss with mandatory IDF weights.

        Args:
            idf_weights: Pre-computed IDF weights [vocab_size] - REQUIRED
            lambda_*: Loss component weights
            temperature: InfoNCE temperature
            margin: Triplet margin
            top_k: Top-k for minimum activation
            min_activation: Minimum activation threshold
            kd_temperature: Knowledge distillation temperature
            idf_alpha: IDF exponential scaling factor
            stopword_mask: Binary mask for stopwords (1=keep, 0=mask)
            stopword_penalty: Extra penalty multiplier for stopwords in FLOPS
            teacher_model: Dense teacher for knowledge distillation

        Raises:
            ValueError: If idf_weights is None
        """
        super().__init__()

        if idf_weights is None:
            raise ValueError(
                "SPLADELossV25 requires idf_weights. "
                "Use load_or_compute_idf() to compute from corpus."
            )

        vocab_size = idf_weights.shape[0]

        # Loss weights
        self.lambda_infonce = lambda_infonce
        self.lambda_self = lambda_self
        self.lambda_positive = lambda_positive
        self.lambda_margin = lambda_margin
        self.lambda_flops = lambda_flops
        self.lambda_min_act = lambda_min_act
        self.lambda_kd = lambda_kd

        # Store stopword mask
        if stopword_mask is not None:
            self.register_buffer("stopword_mask", stopword_mask)
        else:
            self.stopword_mask = None

        # Individual loss modules
        self.infonce_loss = InfoNCELoss(temperature=temperature)
        self.self_loss = SelfReconstructionLoss()
        self.positive_loss = PositiveActivationLoss()
        self.margin_loss = TripletMarginLoss(margin=margin)

        # IDF-aware FLOPS loss (create first with original IDF)
        self.flops_loss = IDFAwareFLOPSLoss(
            vocab_size=vocab_size,
            idf_weights=idf_weights,
            alpha=idf_alpha,
        )

        # Apply stopword penalty AFTER penalty_weights are computed
        # This multiplies penalty for stopword positions (higher penalty)
        if stopword_mask is not None:
            self._apply_stopword_penalty_to_flops(stopword_mask, stopword_penalty)

        self.min_act_loss = MinimumActivationLoss(
            top_k=top_k,
            min_activation=min_activation,
        )
        self.kd_loss = KnowledgeDistillationLoss(temperature=kd_temperature)

        # Teacher model
        self.teacher_model = teacher_model

        # Tracking for logging
        self._semantic_ratio_sum = 0.0
        self._semantic_ratio_count = 0

    def _apply_stopword_penalty_to_flops(
        self,
        stopword_mask: torch.Tensor,
        stopword_penalty: float,
    ) -> None:
        """
        Apply extra penalty to stopword positions in FLOPS loss.

        Multiplies penalty_weights for stopword positions AFTER IDF
        normalization, avoiding issues with pre-normalization modification.

        Args:
            stopword_mask: Binary mask (1=keep, 0=stopword)
            stopword_penalty: Penalty multiplier (>1 increases penalty)
        """
        # Get current penalty weights from FLOPS loss
        penalty_weights = self.flops_loss.penalty_weights.clone()

        # Stopword positions (mask=0) get multiplied penalty
        stopword_indices = (stopword_mask == 0)
        penalty_weights[stopword_indices] = penalty_weights[stopword_indices] * stopword_penalty

        # Update the buffer
        self.flops_loss.register_buffer("penalty_weights", penalty_weights)

    def forward(
        self,
        anchor_repr: torch.Tensor,
        positive_repr: torch.Tensor,
        negative_repr: torch.Tensor,
        anchor_input_ids: torch.Tensor,
        anchor_attention_mask: torch.Tensor,
        positive_input_ids: torch.Tensor,
        positive_attention_mask: torch.Tensor,
        anchor_texts: Optional[list] = None,
        positive_texts: Optional[list] = None,
        teacher_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss with IDF-aware regularization.

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
            teacher_scores: Pre-computed teacher scores

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # V32: Apply stopword mask ONLY to InfoNCE and FLOPS (not self/positive)
        # R2 revision: self-reconstruction needs unmasked repr to learn
        # which tokens to activate; masking everything caused V31 collapse
        masked_anchor = anchor_repr
        masked_positive = positive_repr
        masked_negative = negative_repr
        if self.stopword_mask is not None:
            masked_anchor = anchor_repr * self.stopword_mask
            masked_positive = positive_repr * self.stopword_mask
            masked_negative = negative_repr * self.stopword_mask

        # V32: InfoNCE uses masked repr directly (no IDF scoring weights)
        # R3 revert: idf_scoring_weights double-penalized with IDF-FLOPS
        loss_infonce = self.infonce_loss(
            masked_anchor, masked_positive, masked_negative
        )

        # Self-reconstruction and positive activation use RAW repr
        # so the model can still learn to activate input tokens
        loss_self = self.self_loss(
            anchor_repr, anchor_input_ids, anchor_attention_mask
        )

        loss_positive = self.positive_loss(
            anchor_repr, positive_input_ids, positive_attention_mask
        )

        loss_margin = self.margin_loss(
            masked_anchor, masked_positive, masked_negative
        )

        # IDF-aware FLOPS on masked repr
        loss_flops = self.flops_loss(masked_anchor)

        loss_min_act = self.min_act_loss(anchor_repr)

        # Knowledge distillation
        loss_kd = torch.tensor(0.0, device=anchor_repr.device)
        if self.lambda_kd > 0 and (teacher_scores is not None or self.teacher_model):
            anchor_norm = F.normalize(masked_anchor, p=2, dim=-1)
            positive_norm = F.normalize(masked_positive, p=2, dim=-1)
            student_scores = torch.mm(anchor_norm, positive_norm.t())

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

        # Compute semantic ratio for monitoring
        semantic_ratio = self._compute_semantic_ratio(masked_anchor)
        self._semantic_ratio_sum += semantic_ratio
        self._semantic_ratio_count += 1

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
            "semantic_ratio": semantic_ratio,
        }

        return total_loss, loss_dict

    def _compute_semantic_ratio(self, repr: torch.Tensor) -> float:
        """
        Compute ratio of semantic to stopword activations.

        Higher ratio = better (semantic tokens dominating).

        Args:
            repr: Sparse representations [batch_size, vocab_size]

        Returns:
            Ratio of mean semantic activation to mean stopword activation
        """
        if self.stopword_mask is None:
            return 1.0

        mean_activation = repr.mean(dim=0)  # [vocab_size]

        semantic_mask = (self.stopword_mask == 1)
        stopword_mask = (self.stopword_mask == 0)

        semantic_mean = mean_activation[semantic_mask].mean().item()
        stopword_mean = mean_activation[stopword_mask].mean().item() + 1e-8

        return semantic_mean / stopword_mean

    def get_average_semantic_ratio(self) -> float:
        """Get average semantic ratio since last reset."""
        if self._semantic_ratio_count == 0:
            return 0.0
        return self._semantic_ratio_sum / self._semantic_ratio_count

    def reset_metrics(self) -> None:
        """Reset accumulated metrics."""
        self._semantic_ratio_sum = 0.0
        self._semantic_ratio_count = 0

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


class SPLADELossV26(nn.Module):
    """
    SPLADE loss v26 with enhanced IDF-aware FLOPS and special token handling.

    Key improvements over v25:
    - Special tokens excluded from IDF normalization range
    - Fixed high penalty (100.0) for special tokens (<s>, </s>)
    - Increased stopword penalty (15.0 vs 5.0)
    - Higher FLOPS weight (0.010 vs 0.002)
    - Sharper IDF penalty curve (alpha 4.0 vs 2.5)

    This addresses V25's root cause: special tokens with max IDF compressed
    the normalization range, making stopword penalties ineffective.
    """

    def __init__(
        self,
        idf_weights: torch.Tensor,
        special_token_ids: set,
        # Loss weights (V26 defaults - higher FLOPS weight)
        lambda_infonce: float = 3.0,
        lambda_self: float = 0.5,
        lambda_positive: float = 2.0,
        lambda_margin: float = 0.0,
        lambda_flops: float = 0.010,  # 5x increase from V25
        lambda_min_act: float = 1.0,
        lambda_kd: float = 2.0,
        # Hyperparameters
        temperature: float = 0.07,
        margin: float = 0.3,
        top_k: int = 5,
        min_activation: float = 0.5,
        kd_temperature: float = 3.0,
        # IDF configuration (V26 - sharper curve)
        idf_alpha: float = 4.0,  # Increased from 2.5
        # Special token handling (V26 new)
        special_penalty: float = 100.0,
        # Stopword masking (V26 - stronger penalty)
        stopword_mask: Optional[torch.Tensor] = None,
        stopword_penalty: float = 15.0,  # 3x increase from V25
        # Teacher model
        teacher_model: Optional[DenseTeacherScorer] = None,
    ):
        """
        Initialize SPLADE v26 loss with enhanced special token handling.

        Args:
            idf_weights: Pre-computed IDF weights [vocab_size] - REQUIRED
            special_token_ids: Set of special token IDs to exclude from normalization
            lambda_*: Loss component weights
            temperature: InfoNCE temperature
            margin: Triplet margin
            top_k: Top-k for minimum activation
            min_activation: Minimum activation threshold
            kd_temperature: Knowledge distillation temperature
            idf_alpha: IDF exponential scaling factor (sharper in V26)
            special_penalty: Fixed penalty for special tokens (high in V26)
            stopword_mask: Binary mask for stopwords (1=keep, 0=mask)
            stopword_penalty: Extra penalty multiplier for stopwords
            teacher_model: Dense teacher for knowledge distillation

        Raises:
            ValueError: If idf_weights or special_token_ids is None
        """
        super().__init__()

        if idf_weights is None:
            raise ValueError(
                "SPLADELossV26 requires idf_weights. "
                "Use load_or_compute_idf() to compute from corpus."
            )

        if special_token_ids is None or len(special_token_ids) == 0:
            raise ValueError(
                "SPLADELossV26 requires special_token_ids for proper IDF handling."
            )

        vocab_size = idf_weights.shape[0]

        # Loss weights
        self.lambda_infonce = lambda_infonce
        self.lambda_self = lambda_self
        self.lambda_positive = lambda_positive
        self.lambda_margin = lambda_margin
        self.lambda_flops = lambda_flops
        self.lambda_min_act = lambda_min_act
        self.lambda_kd = lambda_kd

        # Store stopword mask
        if stopword_mask is not None:
            self.register_buffer("stopword_mask", stopword_mask)
        else:
            self.stopword_mask = None

        # Store special token IDs for reference
        self.special_token_ids = special_token_ids

        # Store IDF weights for scoring (Issue #23: paper Eq.5)
        # Normalize IDF to [0, 1] range, then sqrt for symmetric weighting
        idf_min = idf_weights.min()
        idf_max = idf_weights.max()
        idf_normalized = (idf_weights - idf_min) / (idf_max - idf_min + 1e-8)
        idf_scoring_weights = torch.sqrt(idf_normalized.clamp(min=0.01))
        self.register_buffer("idf_scoring_weights", idf_scoring_weights)

        # Individual loss modules
        self.infonce_loss = InfoNCELoss(temperature=temperature)
        self.self_loss = SelfReconstructionLoss()
        self.positive_loss = PositiveActivationLoss()
        self.margin_loss = TripletMarginLoss(margin=margin)

        # V26 IDF-aware FLOPS with special token handling
        self.flops_loss = IDFAwareFLOPSLoss(
            vocab_size=vocab_size,
            idf_weights=idf_weights,
            alpha=idf_alpha,
            special_token_ids=special_token_ids,
            special_penalty=special_penalty,
        )

        # Apply stopword penalty AFTER penalty_weights are computed
        if stopword_mask is not None:
            self._apply_stopword_penalty_to_flops(stopword_mask, stopword_penalty)

        self.min_act_loss = MinimumActivationLoss(
            top_k=top_k,
            min_activation=min_activation,
        )
        self.kd_loss = KnowledgeDistillationLoss(temperature=kd_temperature)

        # Teacher model
        self.teacher_model = teacher_model

        # Tracking for logging
        self._semantic_ratio_sum = 0.0
        self._semantic_ratio_count = 0

    def _apply_stopword_penalty_to_flops(
        self,
        stopword_mask: torch.Tensor,
        stopword_penalty: float,
    ) -> None:
        """
        Apply extra penalty to stopword positions in FLOPS loss.

        Multiplies penalty_weights for stopword positions, excluding
        special tokens which already have fixed high penalty.

        Args:
            stopword_mask: Binary mask (1=keep, 0=stopword)
            stopword_penalty: Penalty multiplier (>1 increases penalty)
        """
        penalty_weights = self.flops_loss.penalty_weights.clone()

        # Stopword positions (mask=0) get multiplied penalty
        # But exclude special tokens (they already have special_penalty)
        stopword_indices = (stopword_mask == 0)
        for idx in range(len(stopword_indices)):
            if stopword_indices[idx] and idx not in self.special_token_ids:
                penalty_weights[idx] = penalty_weights[idx] * stopword_penalty

        self.flops_loss.register_buffer("penalty_weights", penalty_weights)

    def forward(
        self,
        anchor_repr: torch.Tensor,
        positive_repr: torch.Tensor,
        negative_repr: torch.Tensor,
        anchor_input_ids: torch.Tensor,
        anchor_attention_mask: torch.Tensor,
        positive_input_ids: torch.Tensor,
        positive_attention_mask: torch.Tensor,
        anchor_texts: Optional[list] = None,
        positive_texts: Optional[list] = None,
        teacher_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss with V26 IDF-aware regularization.

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
            teacher_scores: Pre-computed teacher scores

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # V32: Apply stopword mask ONLY to InfoNCE and FLOPS (not self/positive)
        # R2 revision: self-reconstruction needs unmasked repr to learn
        # which tokens to activate; masking everything caused V31 collapse
        masked_anchor = anchor_repr
        masked_positive = positive_repr
        masked_negative = negative_repr
        if self.stopword_mask is not None:
            masked_anchor = anchor_repr * self.stopword_mask
            masked_positive = positive_repr * self.stopword_mask
            masked_negative = negative_repr * self.stopword_mask

        # V32: InfoNCE uses masked repr directly (no IDF scoring weights)
        # R3 revert: idf_scoring_weights double-penalized with IDF-FLOPS
        loss_infonce = self.infonce_loss(
            masked_anchor, masked_positive, masked_negative
        )

        # Self-reconstruction and positive activation use RAW repr
        # so the model can still learn to activate input tokens
        loss_self = self.self_loss(
            anchor_repr, anchor_input_ids, anchor_attention_mask
        )

        loss_positive = self.positive_loss(
            anchor_repr, positive_input_ids, positive_attention_mask
        )

        loss_margin = self.margin_loss(
            masked_anchor, masked_positive, masked_negative
        )

        # IDF-aware FLOPS on masked repr
        loss_flops = self.flops_loss(masked_anchor)

        loss_min_act = self.min_act_loss(anchor_repr)

        # Knowledge distillation
        loss_kd = torch.tensor(0.0, device=anchor_repr.device)
        if self.lambda_kd > 0 and (teacher_scores is not None or self.teacher_model):
            anchor_norm = F.normalize(masked_anchor, p=2, dim=-1)
            positive_norm = F.normalize(masked_positive, p=2, dim=-1)
            student_scores = torch.mm(anchor_norm, positive_norm.t())

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

        # Compute semantic ratio for monitoring
        semantic_ratio = self._compute_semantic_ratio(masked_anchor)
        self._semantic_ratio_sum += semantic_ratio
        self._semantic_ratio_count += 1

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
            "semantic_ratio": semantic_ratio,
        }

        return total_loss, loss_dict

    def _compute_semantic_ratio(self, repr: torch.Tensor) -> float:
        """
        Compute ratio of semantic to stopword activations.

        Higher ratio = better (semantic tokens dominating).

        Args:
            repr: Sparse representations [batch_size, vocab_size]

        Returns:
            Ratio of mean semantic activation to mean stopword activation
        """
        if self.stopword_mask is None:
            return 1.0

        mean_activation = repr.mean(dim=0)

        semantic_mask = (self.stopword_mask == 1)
        stopword_mask = (self.stopword_mask == 0)

        semantic_mean = mean_activation[semantic_mask].mean().item()
        stopword_mean = mean_activation[stopword_mask].mean().item() + 1e-8

        return semantic_mean / stopword_mean

    def get_average_semantic_ratio(self) -> float:
        """Get average semantic ratio since last reset."""
        if self._semantic_ratio_count == 0:
            return 0.0
        return self._semantic_ratio_sum / self._semantic_ratio_count

    def reset_metrics(self) -> None:
        """Reset accumulated metrics."""
        self._semantic_ratio_sum = 0.0
        self._semantic_ratio_count = 0

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



class SPLADELossV28(SPLADELossV26):
    """
    SPLADE loss v28 with language filtering and context-aware expansion.

    Key improvements over V26:
    - V28a: Korean language filtering (suppress non-Korean token activation)
    - V28b: Context-aware loss components for context-gated models

    This addresses multilingual token leakage where non-Korean tokens
    inappropriately dominate sparse representations for Korean queries.
    """

    def __init__(
        self,
        idf_weights: torch.Tensor,
        special_token_ids: set,
        # V28: Language filtering
        korean_token_ids: Optional[set] = None,
        lambda_language: float = 0.1,
        non_korean_penalty: float = 1.0,
        korean_penalty: float = 0.0,
        enable_language_filtering: bool = True,
        # Language penalty warmup
        language_warmup_steps: int = 5000,
        language_penalty_max: float = 0.1,
        # Collapse detection
        collapse_flops_threshold: float = 0.01,
        collapse_check_window: int = 3,
        # Loss weights (inherit V26 defaults)
        lambda_infonce: float = 3.0,
        lambda_self: float = 0.5,
        lambda_positive: float = 2.0,
        lambda_margin: float = 0.0,
        lambda_flops: float = 0.010,
        flops_warmup_steps: int = 0,
        lambda_min_act: float = 5.0,
        lambda_kd: float = 2.0,
        # Hyperparameters
        temperature: float = 0.07,
        margin: float = 0.3,
        top_k: int = 10,
        min_activation: float = 1.0,
        kd_temperature: float = 3.0,
        # IDF configuration
        idf_alpha: float = 4.0,
        special_penalty: float = 100.0,
        # Stopword masking
        stopword_mask: Optional[torch.Tensor] = None,
        stopword_penalty: float = 15.0,
        # Teacher model
        teacher_model: Optional[DenseTeacherScorer] = None,
    ):
        """
        Initialize SPLADE v28 loss with language filtering.

        Args:
            idf_weights: Pre-computed IDF weights [vocab_size]
            special_token_ids: Set of special token IDs
            korean_token_ids: Korean token IDs for filtering
            lambda_language: Weight for language filtering loss
            non_korean_penalty: Penalty for non-Korean tokens
            korean_penalty: Penalty for Korean tokens (0 = none)
            enable_language_filtering: Apply language filtering
            language_warmup_steps: Steps to warmup penalty
            language_penalty_max: Max lambda_language after warmup
            collapse_flops_threshold: FLOPS below this = collapse
            collapse_check_window: Consecutive checks for guard
        """
        super().__init__(
            idf_weights=idf_weights,
            special_token_ids=special_token_ids,
            lambda_infonce=lambda_infonce,
            lambda_self=lambda_self,
            lambda_positive=lambda_positive,
            lambda_margin=lambda_margin,
            lambda_flops=lambda_flops,
            lambda_min_act=lambda_min_act,
            lambda_kd=lambda_kd,
            temperature=temperature,
            margin=margin,
            top_k=top_k,
            min_activation=min_activation,
            kd_temperature=kd_temperature,
            idf_alpha=idf_alpha,
            special_penalty=special_penalty,
            stopword_mask=stopword_mask,
            stopword_penalty=stopword_penalty,
            teacher_model=teacher_model,
        )

        # V28: FLOPS warmup
        self.flops_warmup_steps = flops_warmup_steps
        self.target_lambda_flops = lambda_flops

        # V28: Language filtering
        self.enable_language_filtering = enable_language_filtering
        self.lambda_language = lambda_language
        self.non_korean_penalty = non_korean_penalty
        self.korean_penalty = korean_penalty
        self.korean_token_ids = korean_token_ids or set()

        # V28: Language penalty warmup
        self.language_warmup_steps = language_warmup_steps
        self.language_penalty_max = language_penalty_max
        self._global_step = 0

        # V28: Collapse detection
        self.collapse_flops_threshold = collapse_flops_threshold
        self.collapse_check_window = collapse_check_window
        self._low_flops_count = 0
        self._collapse_halvings = 0

        # Build non-Korean mask for penalty computation
        if enable_language_filtering and korean_token_ids:
            self._non_korean_mask = self._build_non_korean_mask(
                idf_weights.shape[0]
            )
            self.register_buffer(
                "non_korean_mask", self._non_korean_mask
            )
        else:
            self.non_korean_mask = None

        # V28 metrics
        self._korean_ratio_sum = 0.0
        self._korean_ratio_count = 0

    def _build_non_korean_mask(self, vocab_size: int) -> torch.Tensor:
        """
        Build penalty mask for non-Korean tokens.

        Returns:
            Mask where 1.0 = non-Korean (penalize), 0.0 = Korean (preserve)
        """
        mask = torch.ones(vocab_size)

        # Korean tokens get 0 (no penalty)
        for token_id in self.korean_token_ids:
            if 0 <= token_id < vocab_size:
                mask[token_id] = 0.0

        # Special tokens also get 0 (handled separately)
        for token_id in self.special_token_ids:
            if 0 <= token_id < vocab_size:
                mask[token_id] = 0.0

        return mask

    def _compute_language_penalty(
        self,
        sparse_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute penalty for non-Korean token activation.

        Args:
            sparse_repr: Sparse representations [batch_size, vocab_size]

        Returns:
            Scalar penalty loss
        """
        if self.non_korean_mask is None:
            return torch.tensor(0.0, device=sparse_repr.device)

        # Penalize activation of non-Korean tokens
        # non_korean_mask: 1.0 for non-Korean, 0.0 for Korean
        non_korean_activation = sparse_repr * self.non_korean_mask.to(sparse_repr.device)

        # Mean activation of non-Korean tokens (should be minimized)
        penalty = non_korean_activation.sum(dim=-1).mean()

        return penalty

    def _compute_korean_ratio(self, sparse_repr: torch.Tensor) -> float:
        """
        Compute ratio of Korean to non-Korean token activations.

        Higher ratio = better (Korean tokens dominating).
        """
        if self.non_korean_mask is None:
            return 1.0

        mean_activation = sparse_repr.mean(dim=0)  # [vocab_size]

        korean_mask = (self.non_korean_mask == 0)
        non_korean_mask = (self.non_korean_mask == 1)

        korean_mean = mean_activation[korean_mask].mean().item()
        non_korean_mean = mean_activation[non_korean_mask].mean().item() + 1e-8

        return korean_mean / non_korean_mean

    def _get_effective_lambda_language(self) -> float:
        """Get warmup-scaled language penalty weight."""
        if self._global_step >= self.language_warmup_steps:
            return self.language_penalty_max
        ratio = self._global_step / max(self.language_warmup_steps, 1)
        return self.language_penalty_max * ratio

    def _get_effective_lambda_flops(self) -> float:
        """Get warmup-scaled FLOPS regularization weight."""
        if self.flops_warmup_steps <= 0 or self._global_step >= self.flops_warmup_steps:
            return self.target_lambda_flops
        ratio = self._global_step / max(self.flops_warmup_steps, 1)
        return self.target_lambda_flops * ratio

    def _check_collapse(self, flops: float) -> None:
        """Check for training collapse and auto-reduce penalty."""
        if flops < self.collapse_flops_threshold:
            self._low_flops_count += 1
        else:
            self._low_flops_count = 0

        if self._low_flops_count >= self.collapse_check_window:
            self.language_penalty_max *= 0.5
            self._collapse_halvings += 1
            self._low_flops_count = 0
            import logging
            logging.getLogger(__name__).warning(
                f"Collapse detected! Halving language_penalty_max "
                f"to {self.language_penalty_max:.6f} "
                f"(halving #{self._collapse_halvings})"
            )

    def _check_sparsity(self, active_tokens: float) -> None:
        """Check if model is producing too many active tokens."""
        if active_tokens > 1000 and self._global_step > self.flops_warmup_steps:
            # Model is not sparse enough even after warmup
            self.target_lambda_flops *= 1.5
            import logging
            logging.getLogger(__name__).warning(
                f"Insufficient sparsity (active_tokens={active_tokens:.0f})! "
                f"Increasing target_lambda_flops to {self.target_lambda_flops:.4f}"
            )

    def set_global_step(self, step: int) -> None:
        """Update global step for warmup scheduling."""
        self._global_step = step

    def forward(
        self,
        anchor_repr: torch.Tensor,
        positive_repr: torch.Tensor,
        negative_repr: torch.Tensor,
        anchor_input_ids: torch.Tensor,
        anchor_attention_mask: torch.Tensor,
        positive_input_ids: torch.Tensor,
        positive_attention_mask: torch.Tensor,
        anchor_texts: Optional[list] = None,
        positive_texts: Optional[list] = None,
        teacher_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss with V28 language filtering.

        Includes warmup scheduling and collapse detection.
        """
        # Get V26 base losses
        total_loss, loss_dict = super().forward(
            anchor_repr=anchor_repr,
            positive_repr=positive_repr,
            negative_repr=negative_repr,
            anchor_input_ids=anchor_input_ids,
            anchor_attention_mask=anchor_attention_mask,
            positive_input_ids=positive_input_ids,
            positive_attention_mask=positive_attention_mask,
            anchor_texts=anchor_texts,
            positive_texts=positive_texts,
            teacher_scores=teacher_scores,
        )

        # V28: FLOPS warmup correction
        # The base V26 used self.lambda_flops for FLOPS. We correct it here.
        if self.flops_warmup_steps > 0:
            effective_flops_lambda = self._get_effective_lambda_flops()
            flops_val = loss_dict.get("flops", 0.0)
            # Remove old FLOPS contribution and add new one
            old_flops_contrib = self.lambda_flops * flops_val
            new_flops_contrib = effective_flops_lambda * flops_val
            total_loss = total_loss + (new_flops_contrib - old_flops_contrib)
            loss_dict["effective_lambda_flops"] = effective_flops_lambda

        # Sparsity monitoring
        with torch.no_grad():
            active_tokens = (anchor_repr > 0.01).float().sum(dim=-1).mean().item()
            loss_dict["active_tokens"] = active_tokens

        self._check_sparsity(active_tokens)

        # V28: Add language filtering with warmup
        if self.enable_language_filtering and self.non_korean_mask is not None:
            lang_penalty = self._compute_language_penalty(anchor_repr)
            effective_lambda = self._get_effective_lambda_language()
            total_loss = total_loss + effective_lambda * lang_penalty
            loss_dict["language_penalty"] = lang_penalty.item()
            loss_dict["effective_lambda_lang"] = effective_lambda

            # Track Korean ratio
            korean_ratio = self._compute_korean_ratio(anchor_repr)
            self._korean_ratio_sum += korean_ratio
            self._korean_ratio_count += 1
            loss_dict["korean_ratio"] = korean_ratio

            # Collapse detection using flops from base loss
            flops_val = loss_dict.get("flops", 1.0)
            self._check_collapse(flops_val)
            loss_dict["collapse_halvings"] = self._collapse_halvings
        else:
            loss_dict["language_penalty"] = 0.0
            loss_dict["korean_ratio"] = 1.0
            loss_dict["effective_lambda_lang"] = 0.0
            loss_dict["collapse_halvings"] = 0

        return total_loss, loss_dict

    def get_average_korean_ratio(self) -> float:
        """Get average Korean token ratio since last reset."""
        if self._korean_ratio_count == 0:
            return 0.0
        return self._korean_ratio_sum / self._korean_ratio_count

    def reset_metrics(self) -> None:
        """Reset accumulated metrics."""
        super().reset_metrics()
        self._korean_ratio_sum = 0.0
        self._korean_ratio_count = 0

    def update_language_weight(self, lambda_language: float) -> None:
        """Update language filtering weight."""
        self.lambda_language = lambda_language
        self.language_penalty_max = lambda_language


class SPLADELossV29(SPLADELossV28):
    """
    SPLADE loss v29 with SPLADE v2 style FLOPS regularization.

    Key improvements over V28:
    - Separate FLOPS regularization for query (lambda_q) and document (lambda_d)
    - Quadratic lambda warmup support via external scheduler
    - Margin-MSE distillation support from cross-encoder

    Based on SPLADE v2 paper (Formal et al., 2022):
    - L_FLOPS = sum_j (1/N * sum_i w_j^(d_i))^2
    - L = L_rank + λ_q * L_reg^q + λ_d * L_reg^d
    """

    def __init__(
        self,
        idf_weights: torch.Tensor,
        special_token_ids: set,
        # V29: Separate FLOPS weights
        lambda_flops_q: float = 1e-4,
        lambda_flops_d: float = 1e-3,
        # V29: Distillation
        use_margin_mse: bool = False,
        # Inherited from V28
        korean_token_ids: Optional[set] = None,
        lambda_language: float = 0.3,
        non_korean_penalty: float = 5.0,
        korean_penalty: float = 0.0,
        enable_language_filtering: bool = True,
        # Standard loss weights
        lambda_infonce: float = 3.0,
        lambda_self: float = 0.5,
        lambda_positive: float = 2.0,
        lambda_margin: float = 0.0,
        lambda_flops: float = 0.0,  # Disabled, use lambda_flops_q/d instead
        lambda_min_act: float = 1.0,
        lambda_kd: float = 2.0,
        # Hyperparameters
        temperature: float = 0.07,
        margin: float = 0.3,
        top_k: int = 5,
        min_activation: float = 0.5,
        kd_temperature: float = 3.0,
        # IDF configuration
        idf_alpha: float = 4.0,
        special_penalty: float = 100.0,
        # Stopword masking
        stopword_mask: Optional[torch.Tensor] = None,
        stopword_penalty: float = 15.0,
        # Teacher model
        teacher_model: Optional[DenseTeacherScorer] = None,
    ):
        """
        Initialize SPLADE v29 loss with separate FLOPS regularization.

        Args:
            idf_weights: Pre-computed IDF weights [vocab_size]
            special_token_ids: Set of special token IDs
            lambda_flops_q: FLOPS regularization weight for queries
            lambda_flops_d: FLOPS regularization weight for documents
            use_margin_mse: Use margin-MSE distillation from cross-encoder
            Other args: Inherited from SPLADELossV28
        """
        super().__init__(
            idf_weights=idf_weights,
            special_token_ids=special_token_ids,
            korean_token_ids=korean_token_ids,
            lambda_language=lambda_language,
            non_korean_penalty=non_korean_penalty,
            korean_penalty=korean_penalty,
            enable_language_filtering=enable_language_filtering,
            lambda_infonce=lambda_infonce,
            lambda_self=lambda_self,
            lambda_positive=lambda_positive,
            lambda_margin=lambda_margin,
            lambda_flops=lambda_flops,  # Will be 0, handled separately
            lambda_min_act=lambda_min_act,
            lambda_kd=lambda_kd,
            temperature=temperature,
            margin=margin,
            top_k=top_k,
            min_activation=min_activation,
            kd_temperature=kd_temperature,
            idf_alpha=idf_alpha,
            special_penalty=special_penalty,
            stopword_mask=stopword_mask,
            stopword_penalty=stopword_penalty,
            teacher_model=teacher_model,
        )

        # V29 specific
        self.lambda_flops_q = lambda_flops_q
        self.lambda_flops_d = lambda_flops_d
        self.use_margin_mse = use_margin_mse

    def _compute_flops_v29(self, sparse_repr: torch.Tensor) -> torch.Tensor:
        """
        Compute IDF-weighted FLOPS regularization (Issue #21).

        L_FLOPS = sum_j (w_j * mean_activation_j^2)

        Uses V26 IDFAwareFLOPSLoss penalty_weights which include:
        - IDF-based weights: exp(-alpha * normalized_idf)
        - Stopword penalty: 15x for known stopwords
        - Special token penalty: 100x for <s>, </s>

        This ensures Korean particles (low IDF) get 15x+ higher
        FLOPS penalty than semantic content tokens (high IDF).

        Args:
            sparse_repr: Sparse representations [batch_size, vocab_size]

        Returns:
            FLOPS regularization loss
        """
        # Average activation per token across batch
        mean_activation = sparse_repr.mean(dim=0)  # [vocab_size]

        # IDF-weighted FLOPS: stopwords penalized 15x more
        penalty_weights = self.flops_loss.penalty_weights  # [vocab_size]
        flops_loss = (penalty_weights * (mean_activation ** 2)).sum()

        return flops_loss

    def _compute_margin_mse(
        self,
        anchor_repr: torch.Tensor,
        positive_repr: torch.Tensor,
        negative_repr: torch.Tensor,
        teacher_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute margin-MSE distillation loss.

        Args:
            anchor_repr: Query representations [batch, vocab]
            positive_repr: Positive document representations [batch, vocab]
            negative_repr: Negative document representations [batch, vocab]
            teacher_scores: Teacher scores [batch, 2] (pos_score, neg_score)

        Returns:
            Margin-MSE loss
        """
        # Student scores (dot product)
        student_pos = (anchor_repr * positive_repr).sum(dim=-1)
        student_neg = (anchor_repr * negative_repr).sum(dim=-1)
        student_margin = student_pos - student_neg

        # Teacher margin
        teacher_margin = teacher_scores[:, 0] - teacher_scores[:, 1]

        # MSE on margins
        margin_mse = F.mse_loss(student_margin, teacher_margin)

        return margin_mse

    def forward(
        self,
        anchor_repr: torch.Tensor,
        positive_repr: torch.Tensor,
        negative_repr: torch.Tensor,
        anchor_input_ids: torch.Tensor,
        anchor_attention_mask: torch.Tensor,
        positive_input_ids: torch.Tensor,
        positive_attention_mask: torch.Tensor,
        anchor_texts: Optional[list] = None,
        positive_texts: Optional[list] = None,
        teacher_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss with V29 FLOPS regularization.

        Args:
            anchor_repr: Query sparse representations [batch_size, vocab_size]
            positive_repr: Positive doc sparse representations [batch_size, vocab_size]
            negative_repr: Negative doc sparse representations [batch_size, vocab_size]
            anchor_input_ids: Query token IDs
            anchor_attention_mask: Query attention mask
            positive_input_ids: Positive token IDs
            positive_attention_mask: Positive attention mask
            anchor_texts: Raw query texts for teacher scoring
            positive_texts: Raw positive texts for teacher scoring
            teacher_scores: Pre-computed teacher scores [batch, 2]

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Get V28 base losses (with lambda_flops=0)
        total_loss, loss_dict = super().forward(
            anchor_repr=anchor_repr,
            positive_repr=positive_repr,
            negative_repr=negative_repr,
            anchor_input_ids=anchor_input_ids,
            anchor_attention_mask=anchor_attention_mask,
            positive_input_ids=positive_input_ids,
            positive_attention_mask=positive_attention_mask,
            anchor_texts=anchor_texts,
            positive_texts=positive_texts,
            teacher_scores=teacher_scores if not self.use_margin_mse else None,
        )

        # V29: Separate FLOPS for query and document (masked, Issue #22)
        masked_q = anchor_repr
        masked_p = positive_repr
        masked_n = negative_repr
        if self.stopword_mask is not None:
            masked_q = anchor_repr * self.stopword_mask
            masked_p = positive_repr * self.stopword_mask
            masked_n = negative_repr * self.stopword_mask
        flops_q = self._compute_flops_v29(masked_q)
        flops_d = self._compute_flops_v29(masked_p) + self._compute_flops_v29(masked_n)

        flops_loss = self.lambda_flops_q * flops_q + self.lambda_flops_d * flops_d
        total_loss = total_loss + flops_loss

        loss_dict["flops_q"] = flops_q.item()
        loss_dict["flops_d"] = flops_d.item()
        loss_dict["flops_total"] = flops_loss.item()

        # V29: Margin-MSE distillation
        if self.use_margin_mse and teacher_scores is not None:
            margin_mse = self._compute_margin_mse(
                anchor_repr, positive_repr, negative_repr, teacher_scores
            )
            total_loss = total_loss + self.lambda_kd * margin_mse
            loss_dict["margin_mse"] = margin_mse.item()

        return total_loss, loss_dict

    def update_lambda_flops(self, lambda_q: float, lambda_d: float) -> None:
        """Update FLOPS regularization weights (for warmup scheduler)."""
        self.lambda_flops_q = lambda_q
        self.lambda_flops_d = lambda_d


class SPLADELossV30(SPLADELossV26):
    """
    SPLADE loss v30 with simplified architecture.

    V30: Simplified loss - back to V26 baseline architecture.
    Only 4 active components: InfoNCE + IDF-FLOPS + Language + KD.
    Removes: self-reconstruction, positive activation, min activation,
    separate q/d FLOPS.

    Key design:
    - Inherits from V26 (not V28/V29) for proven baseline
    - Disables 4 loss components via zero weights
    - Adds V28-style Korean language filtering
    - Reduced KD weight (1.0 vs 2.0)
    - Simplified for faster iteration and clearer signal attribution
    """

    def __init__(
        self,
        idf_weights: torch.Tensor,
        special_token_ids: set,
        # V30 language filtering (from V28)
        korean_token_ids: Optional[set] = None,
        lambda_language: float = 0.3,
        non_korean_penalty: float = 5.0,
        enable_language_filtering: bool = True,
        # Override V26 defaults - simplified
        lambda_infonce: float = 3.0,
        lambda_self: float = 0.0,  # DISABLED
        lambda_positive: float = 0.0,  # DISABLED
        lambda_margin: float = 0.0,  # DISABLED
        lambda_flops: float = 0.010,  # V26-style IDF-weighted
        lambda_min_act: float = 0.0,  # DISABLED
        lambda_kd: float = 1.0,  # Reduced from V26's 2.0
        # Standard hyperparameters
        temperature: float = 0.07,
        kd_temperature: float = 3.0,
        idf_alpha: float = 4.0,
        special_penalty: float = 100.0,
        stopword_mask: Optional[torch.Tensor] = None,
        stopword_penalty: float = 15.0,
        teacher_model: Optional[DenseTeacherScorer] = None,
    ):
        """
        Initialize SPLADE v30 loss with simplified components.

        Args:
            idf_weights: Pre-computed IDF weights [vocab_size] - REQUIRED
            special_token_ids: Set of special token IDs
            korean_token_ids: Set of Korean token IDs for language filtering
            lambda_language: Weight for language filtering loss
            non_korean_penalty: Penalty for non-Korean token activation
            enable_language_filtering: Whether to apply language filtering
            lambda_infonce: Weight for InfoNCE loss (active)
            lambda_self: Self-reconstruction weight (disabled = 0.0)
            lambda_positive: Positive activation weight (disabled = 0.0)
            lambda_margin: Triplet margin weight (disabled = 0.0)
            lambda_flops: FLOPS regularization weight (active, V26-style)
            lambda_min_act: Minimum activation weight (disabled = 0.0)
            lambda_kd: Knowledge distillation weight (active, reduced)
            temperature: InfoNCE temperature
            kd_temperature: Knowledge distillation temperature
            idf_alpha: IDF exponential scaling factor
            special_penalty: Fixed penalty for special tokens
            stopword_mask: Binary mask for stopwords (1=keep, 0=mask)
            stopword_penalty: Extra penalty multiplier for stopwords
            teacher_model: Dense teacher for knowledge distillation
        """
        # Initialize V26 with simplified weights
        super().__init__(
            idf_weights=idf_weights,
            special_token_ids=special_token_ids,
            lambda_infonce=lambda_infonce,
            lambda_self=lambda_self,
            lambda_positive=lambda_positive,
            lambda_margin=lambda_margin,
            lambda_flops=lambda_flops,
            lambda_min_act=lambda_min_act,
            lambda_kd=lambda_kd,
            temperature=temperature,
            kd_temperature=kd_temperature,
            idf_alpha=idf_alpha,
            special_penalty=special_penalty,
            stopword_mask=stopword_mask,
            stopword_penalty=stopword_penalty,
            teacher_model=teacher_model,
        )

        # V30: Language filtering (from V28)
        self.enable_language_filtering = enable_language_filtering
        self.lambda_language = lambda_language
        self.non_korean_penalty = non_korean_penalty
        self.korean_token_ids = korean_token_ids or set()

        # Build non-Korean mask for penalty computation
        if enable_language_filtering and korean_token_ids:
            self._non_korean_mask = self._build_non_korean_mask(
                idf_weights.shape[0]
            )
            self.register_buffer("non_korean_mask", self._non_korean_mask)
        else:
            self.non_korean_mask = None

        # V30 metrics
        self._korean_ratio_sum = 0.0
        self._korean_ratio_count = 0

    def _build_non_korean_mask(self, vocab_size: int) -> torch.Tensor:
        """
        Build penalty mask for non-Korean tokens.

        Returns:
            Mask where 1.0 = non-Korean (penalize), 0.0 = Korean (preserve)
        """
        mask = torch.ones(vocab_size)

        # Korean tokens get 0 (no penalty)
        for token_id in self.korean_token_ids:
            if 0 <= token_id < vocab_size:
                mask[token_id] = 0.0

        # Special tokens also get 0 (handled separately)
        for token_id in self.special_token_ids:
            if 0 <= token_id < vocab_size:
                mask[token_id] = 0.0

        return mask

    def _compute_language_penalty(
        self,
        sparse_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute penalty for non-Korean token activation.

        Args:
            sparse_repr: Sparse representations [batch_size, vocab_size]

        Returns:
            Scalar penalty loss
        """
        if self.non_korean_mask is None:
            return torch.tensor(0.0, device=sparse_repr.device)

        # Penalize activation of non-Korean tokens
        # non_korean_mask: 1.0 for non-Korean, 0.0 for Korean
        non_korean_activation = sparse_repr * self.non_korean_mask.to(
            sparse_repr.device
        )

        # Mean activation of non-Korean tokens (should be minimized)
        penalty = non_korean_activation.sum(dim=-1).mean()

        return penalty

    def _compute_korean_ratio(self, sparse_repr: torch.Tensor) -> float:
        """
        Compute ratio of Korean to non-Korean token activations.

        Higher ratio = better (Korean tokens dominating).

        Args:
            sparse_repr: Sparse representations [batch_size, vocab_size]

        Returns:
            Ratio of mean Korean activation to mean non-Korean activation
        """
        if self.non_korean_mask is None:
            return 1.0

        mean_activation = sparse_repr.mean(dim=0)  # [vocab_size]

        korean_mask = self.non_korean_mask == 0
        non_korean_mask = self.non_korean_mask == 1

        korean_mean = mean_activation[korean_mask].mean().item()
        non_korean_mean = mean_activation[non_korean_mask].mean().item() + 1e-8

        return korean_mean / non_korean_mean

    def forward(
        self,
        anchor_repr: torch.Tensor,
        positive_repr: torch.Tensor,
        negative_repr: torch.Tensor,
        anchor_input_ids: torch.Tensor,
        anchor_attention_mask: torch.Tensor,
        positive_input_ids: torch.Tensor,
        positive_attention_mask: torch.Tensor,
        anchor_texts: Optional[list] = None,
        positive_texts: Optional[list] = None,
        teacher_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss with V30 simplification.

        Args:
            anchor_repr: Anchor sparse representations [batch_size, vocab_size]
            positive_repr: Positive sparse representations
            negative_repr: Negative sparse representations
            anchor_input_ids: Anchor token IDs [batch_size, seq_len]
            anchor_attention_mask: Anchor attention mask [batch_size, seq_len]
            positive_input_ids: Positive token IDs [batch_size, seq_len]
            positive_attention_mask: Positive attention mask [batch_size, seq_len]
            anchor_texts: Raw anchor texts for teacher scoring
            positive_texts: Raw positive texts for teacher scoring
            teacher_scores: Pre-computed teacher scores

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Get V26 base losses (most will be 0 due to zero weights)
        total_loss, loss_dict = super().forward(
            anchor_repr=anchor_repr,
            positive_repr=positive_repr,
            negative_repr=negative_repr,
            anchor_input_ids=anchor_input_ids,
            anchor_attention_mask=anchor_attention_mask,
            positive_input_ids=positive_input_ids,
            positive_attention_mask=positive_attention_mask,
            anchor_texts=anchor_texts,
            positive_texts=positive_texts,
            teacher_scores=teacher_scores,
        )

        # V30: Add language filtering penalty
        if (
            self.enable_language_filtering
            and self.non_korean_mask is not None
        ):
            lang_penalty = self._compute_language_penalty(anchor_repr)
            total_loss = total_loss + self.lambda_language * lang_penalty
            loss_dict["language_penalty"] = lang_penalty.item()

            # Track Korean ratio
            korean_ratio = self._compute_korean_ratio(anchor_repr)
            self._korean_ratio_sum += korean_ratio
            self._korean_ratio_count += 1
            loss_dict["korean_ratio"] = korean_ratio
        else:
            loss_dict["language_penalty"] = 0.0
            loss_dict["korean_ratio"] = 1.0

        return total_loss, loss_dict

    def get_average_korean_ratio(self) -> float:
        """
        Get average Korean token ratio since last reset.

        Returns:
            Average ratio of Korean to non-Korean token activations
        """
        if self._korean_ratio_count == 0:
            return 0.0
        return self._korean_ratio_sum / self._korean_ratio_count

    def reset_metrics(self) -> None:
        """Reset accumulated metrics for both V26 and V30."""
        super().reset_metrics()
        self._korean_ratio_sum = 0.0
        self._korean_ratio_count = 0

    def update_language_weight(self, lambda_language: float) -> None:
        """
        Update language filtering weight.

        Args:
            lambda_language: New language filtering weight
        """
        self.lambda_language = lambda_language


# ===== Issue #17: Unified Sparse + Dense Losses =====


class DenseContrastiveLoss(nn.Module):
    """InfoNCE loss for dense embeddings with in-batch negatives."""

    def __init__(self, temperature: float = 0.05):
        """
        Initialize dense contrastive loss.

        Args:
            temperature: Temperature for softmax scaling (lower = sharper)
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_dense: torch.Tensor,
        positive_dense: torch.Tensor,
        negative_dense: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss for dense embeddings with in-batch negatives.

        Args:
            query_dense: Query dense embeddings [batch, dim] (L2-normalized)
            positive_dense: Positive dense embeddings [batch, dim] (L2-normalized)
            negative_dense: Optional explicit negatives [batch, dim]

        Returns:
            Scalar InfoNCE loss
        """
        batch_size = query_dense.shape[0]

        # Embeddings should already be L2-normalized from DenseHead,
        # but normalize again for safety
        query_dense = F.normalize(query_dense, p=2, dim=-1)
        positive_dense = F.normalize(positive_dense, p=2, dim=-1)

        # Cosine similarity matrix [batch, batch]
        sim_matrix = torch.mm(query_dense, positive_dense.t()) / self.temperature

        if negative_dense is not None:
            negative_dense = F.normalize(negative_dense, p=2, dim=-1)
            neg_sim = torch.mm(query_dense, negative_dense.t()) / self.temperature
            sim_matrix = torch.cat([sim_matrix, neg_sim], dim=1)

        # Labels: diagonal elements are positives
        labels = torch.arange(batch_size, device=query_dense.device)
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class DenseKDLoss(nn.Module):
    """Knowledge distillation loss: MSE between dense scores and teacher scores."""

    def __init__(self):
        """Initialize dense KD loss."""
        super().__init__()

    def forward(
        self,
        student_dense_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MSE loss between student dense similarity scores and teacher scores.

        Args:
            student_dense_scores: Student cosine similarity matrix [batch, batch]
            teacher_scores: Teacher similarity scores [batch, batch]

        Returns:
            Scalar MSE loss
        """
        return F.mse_loss(student_dense_scores, teacher_scores)


class UnifiedLoss(nn.Module):
    """Combined loss for unified sparse + dense training."""

    def __init__(
        self,
        sparse_loss: nn.Module,
        alpha_dense_contrastive: float = 0.3,
        beta_dense_kd: float = 0.1,
        dense_temperature: float = 0.05,
    ):
        """
        Initialize unified loss combining sparse and dense objectives.

        Args:
            sparse_loss: Existing sparse loss (e.g. SPLADELossV28)
            alpha_dense_contrastive: Weight for dense contrastive loss
            beta_dense_kd: Weight for dense knowledge distillation loss
            dense_temperature: Temperature for dense contrastive loss
        """
        super().__init__()
        self.sparse_loss = sparse_loss
        self.dense_contrastive = DenseContrastiveLoss(temperature=dense_temperature)
        self.dense_kd = DenseKDLoss()
        self.alpha = alpha_dense_contrastive
        self.beta = beta_dense_kd

    def forward(
        self,
        # Sparse inputs (passed directly to sparse_loss)
        anchor_repr: torch.Tensor,
        positive_repr: torch.Tensor,
        negative_repr: torch.Tensor,
        anchor_input_ids: torch.Tensor,
        anchor_attention_mask: torch.Tensor,
        positive_input_ids: torch.Tensor,
        positive_attention_mask: torch.Tensor,
        # Dense inputs
        query_dense: torch.Tensor,
        positive_dense: torch.Tensor,
        negative_dense: Optional[torch.Tensor] = None,
        # Optional teacher scores for both sparse and dense KD
        teacher_scores: Optional[torch.Tensor] = None,
        # Additional kwargs forwarded to sparse_loss
        **sparse_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute unified sparse + dense loss.

        Args:
            anchor_repr: Anchor sparse representations [batch, vocab_size]
            positive_repr: Positive sparse representations [batch, vocab_size]
            negative_repr: Negative sparse representations [batch, vocab_size]
            anchor_input_ids: Anchor token IDs [batch, seq_len]
            anchor_attention_mask: Anchor attention mask [batch, seq_len]
            positive_input_ids: Positive token IDs [batch, seq_len]
            positive_attention_mask: Positive attention mask [batch, seq_len]
            query_dense: Query dense embeddings [batch, dim]
            positive_dense: Positive dense embeddings [batch, dim]
            negative_dense: Optional negative dense embeddings [batch, dim]
            teacher_scores: Pre-computed teacher scores [batch, batch]
            **sparse_kwargs: Additional keyword arguments for sparse_loss

        Returns:
            Dict with keys: total_loss, sparse_loss, dense_contrastive_loss, dense_kd_loss
        """
        # Sparse loss
        sparse_total, sparse_dict = self.sparse_loss(
            anchor_repr=anchor_repr,
            positive_repr=positive_repr,
            negative_repr=negative_repr,
            anchor_input_ids=anchor_input_ids,
            anchor_attention_mask=anchor_attention_mask,
            positive_input_ids=positive_input_ids,
            positive_attention_mask=positive_attention_mask,
            teacher_scores=teacher_scores,
            **sparse_kwargs,
        )

        # Dense contrastive loss
        dense_contrastive = self.dense_contrastive(
            query_dense, positive_dense, negative_dense
        )

        # Dense KD loss (optional, only when teacher scores available)
        dense_kd = torch.tensor(0.0, device=query_dense.device)
        if self.beta > 0 and teacher_scores is not None:
            query_norm = F.normalize(query_dense, p=2, dim=-1)
            pos_norm = F.normalize(positive_dense, p=2, dim=-1)
            student_dense_scores = torch.mm(query_norm, pos_norm.t())
            dense_kd = self.dense_kd(student_dense_scores, teacher_scores)

        total_loss = (
            sparse_total
            + self.alpha * dense_contrastive
            + self.beta * dense_kd
        )

        return {
            "total_loss": total_loss,
            "sparse_loss": sparse_total,
            "dense_contrastive_loss": dense_contrastive,
            "dense_kd_loss": dense_kd,
        }
