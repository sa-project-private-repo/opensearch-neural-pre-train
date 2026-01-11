"""
XLM-RoBERTa based SPLADE model for improved multilingual sparse retrieval.

XLM-RoBERTa advantages over mBERT/KoBERT:
- 250K vocabulary (vs 50K) = more expressive sparse vectors
- Better multilingual alignment across 100+ languages
- Proven performance in SPLADE research
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    XLMRobertaModel,
    XLMRobertaForMaskedLM,
)
from typing import Dict, Optional, Tuple, Union


class SPLADEDocXLMR(nn.Module):
    """
    SPLADE-doc with XLM-RoBERTa backbone.

    Uses XLM-RoBERTa's MLM head for vocabulary expansion,
    enabling activation of any token in the 250K vocabulary.

    Key improvements over mBERT/KoBERT:
    - 5x larger vocabulary for richer sparse representations
    - Better cross-lingual token alignment
    - Superior multilingual understanding
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        dropout: float = 0.1,
        use_mlm_head: bool = True,
    ):
        """
        Initialize XLM-RoBERTa SPLADE model.

        Args:
            model_name: XLM-RoBERTa model name
                       ("xlm-roberta-base" or "xlm-roberta-large")
            dropout: Dropout rate for regularization
            use_mlm_head: If True, use MLM head for full vocab expansion.
                         If False, only activate input tokens.
        """
        super().__init__()

        self.model_name = model_name
        self.use_mlm_head = use_mlm_head

        if use_mlm_head:
            # Load with MLM head for vocabulary expansion
            self.model = XLMRobertaForMaskedLM.from_pretrained(model_name)
            self.config = self.model.config
            self.transformer = self.model.roberta
        else:
            # Load base model without MLM head
            self.transformer = XLMRobertaModel.from_pretrained(model_name)
            self.config = self.transformer.config
            # Custom projection for token importance
            self.token_importance = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.config.hidden_size, 1),
            )

        # ReLU for sparsity induction
        self.relu = nn.ReLU()

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size (250K for XLM-R)."""
        return self.config.vocab_size

    @property
    def hidden_size(self) -> int:
        """Return hidden dimension."""
        return self.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,  # Not used by RoBERTa
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing sparse representations.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Ignored (XLM-RoBERTa doesn't use token types)

        Returns:
            sparse_repr: Sparse document representation [batch_size, vocab_size]
            token_weights: Per-position importance [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape

        if self.use_mlm_head:
            # Use MLM head for full vocabulary coverage
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # logits: [batch, seq_len, vocab_size]
            logits = outputs.logits
        else:
            # Use custom projection (input tokens only)
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = outputs.last_hidden_state
            # Predict importance: [batch, seq_len, 1]
            importance = self.token_importance(hidden_states)
            importance = importance.squeeze(-1)  # [batch, seq_len]

            # Convert to per-token sparse repr
            logits = self._create_input_only_logits(
                input_ids, importance, attention_mask
            )

        # Apply ReLU and log(1+x) for sparsity
        # Formula: sparse_repr = log(1 + ReLU(logits))
        sparse_scores = torch.log1p(self.relu(logits))

        # Mask padding positions
        mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
        sparse_scores = sparse_scores * mask

        # Max pooling across sequence positions
        # sparse_repr: [batch, vocab_size]
        sparse_repr, _ = sparse_scores.max(dim=1)

        # Per-position token weights (for analysis/visualization)
        token_weights = sparse_scores.max(dim=-1).values  # [batch, seq_len]

        return sparse_repr, token_weights

    def _create_input_only_logits(
        self,
        input_ids: torch.Tensor,
        importance: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create vocab-sized logits from input token importance scores.

        Only input tokens have non-zero scores.

        Args:
            input_ids: Token IDs [batch, seq_len]
            importance: Importance scores [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        vocab_size = self.config.vocab_size

        # Create one-hot encoding: [batch, seq_len, vocab_size]
        one_hot = torch.zeros(
            batch_size, seq_len, vocab_size,
            device=input_ids.device,
            dtype=importance.dtype,
        ).scatter(2, input_ids.unsqueeze(-1), 1)

        # Scale by importance scores
        # logits[b, s, v] = importance[b, s] if v == input_ids[b, s] else 0
        logits = one_hot * importance.unsqueeze(-1)

        return logits

    def encode_documents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode documents to sparse representations.

        Args:
            input_ids: Document token IDs
            attention_mask: Attention mask
            token_type_ids: Ignored

        Returns:
            Sparse document representations [batch_size, vocab_size]
        """
        sparse_repr, _ = self.forward(input_ids, attention_mask)
        return sparse_repr

    def get_top_k_tokens(
        self,
        sparse_repr: torch.Tensor,
        tokenizer: AutoTokenizer,
        k: int = 50,
    ) -> Dict[str, float]:
        """
        Get top-k tokens with highest weights.

        Args:
            sparse_repr: Sparse representation [vocab_size]
            tokenizer: XLM-R tokenizer for decoding
            k: Number of top tokens

        Returns:
            Dictionary of {token: weight}
        """
        # Get top-k indices and values
        top_k_values, top_k_indices = torch.topk(
            sparse_repr, k=min(k, sparse_repr.shape[0])
        )

        # Convert to dictionary
        result = {}
        for idx, val in zip(top_k_indices.tolist(), top_k_values.tolist()):
            if val > 0:
                token = tokenizer.decode([idx])
                result[token] = val

        return result


class SPLADEDocXLMRWithIDF(SPLADEDocXLMR):
    """
    XLM-RoBERTa SPLADE with IDF-aware weighting.

    Applies IDF-based weights to preserve informative tokens
    and suppress common tokens (stopwords).
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        dropout: float = 0.1,
        use_mlm_head: bool = True,
        idf_weights: Optional[torch.Tensor] = None,
        idf_alpha: float = 2.0,
    ):
        """
        Initialize with IDF weighting.

        Args:
            model_name: XLM-RoBERTa model name
            dropout: Dropout rate
            use_mlm_head: Whether to use MLM head
            idf_weights: Pre-computed IDF weights [vocab_size]
            idf_alpha: IDF penalty strength (higher = more penalty on low-IDF)
        """
        super().__init__(model_name, dropout, use_mlm_head)

        self.idf_alpha = idf_alpha

        if idf_weights is not None:
            self.register_buffer('idf_weights', idf_weights)
        else:
            self.idf_weights = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        apply_idf: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional IDF weighting.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Ignored
            apply_idf: Whether to apply IDF weighting

        Returns:
            sparse_repr: (Optionally IDF-weighted) sparse representation
            token_weights: Per-position token weights
        """
        sparse_repr, token_weights = super().forward(
            input_ids, attention_mask
        )

        if apply_idf and self.idf_weights is not None:
            # Apply IDF weighting: high-IDF tokens get preserved
            sparse_repr = sparse_repr * self.idf_weights.unsqueeze(0)

        return sparse_repr, token_weights

    def set_idf_weights(self, idf_weights: torch.Tensor) -> None:
        """
        Set IDF weights from corpus statistics.

        Args:
            idf_weights: IDF weights [vocab_size]
        """
        self.register_buffer('idf_weights', idf_weights)

    def compute_idf_penalty_weights(
        self, idf_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute penalty weights from IDF values.

        Higher IDF = more informative = lower penalty.

        Args:
            idf_values: Raw IDF values [vocab_size]

        Returns:
            Penalty weights (exponential decay based on IDF)
        """
        # Normalize IDF to [0, 1]
        normalized_idf = (idf_values - idf_values.min()) / (
            idf_values.max() - idf_values.min() + 1e-8
        )

        # Penalty weight: w_j = exp(-alpha * normalized_idf_j)
        # High IDF → low penalty, Low IDF → high penalty
        penalty_weights = torch.exp(-self.idf_alpha * normalized_idf)

        return penalty_weights


def create_splade_xlmr(
    model_name: str = "xlm-roberta-base",
    use_idf: bool = True,
    idf_weights: Optional[torch.Tensor] = None,
    dropout: float = 0.1,
    use_mlm_head: bool = True,
) -> Union[SPLADEDocXLMR, SPLADEDocXLMRWithIDF]:
    """
    Factory function to create XLM-RoBERTa SPLADE model.

    Args:
        model_name: XLM-RoBERTa model variant
        use_idf: Whether to use IDF-aware weighting
        idf_weights: Pre-computed IDF weights
        dropout: Dropout rate
        use_mlm_head: Whether to use MLM head for expansion

    Returns:
        SPLADEDocXLMR or SPLADEDocXLMRWithIDF instance
    """
    if use_idf:
        return SPLADEDocXLMRWithIDF(
            model_name=model_name,
            dropout=dropout,
            use_mlm_head=use_mlm_head,
            idf_weights=idf_weights,
        )
    else:
        return SPLADEDocXLMR(
            model_name=model_name,
            dropout=dropout,
            use_mlm_head=use_mlm_head,
        )


def load_splade_xlmr(
    checkpoint_path: str,
    model_name: str = "xlm-roberta-base",
    use_idf: bool = True,
    device: str = "cuda",
) -> SPLADEDocXLMR:
    """
    Load trained XLM-RoBERTa SPLADE model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        model_name: Base model name
        use_idf: Whether model uses IDF
        device: Target device

    Returns:
        Loaded model
    """
    model = create_splade_xlmr(
        model_name=model_name,
        use_idf=use_idf,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model
