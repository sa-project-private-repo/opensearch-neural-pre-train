"""SPLADE-doc model implementation for inference-free sparse retrieval."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Tuple


class SPLADEDoc(nn.Module):
    """
    SPLADE-doc: Inference-free learned sparse retrieval model.

    Based on "Towards Competitive Search Relevance For Inference-Free
    Learned Sparse Retrievers" paper.

    Key features:
    - Only encodes documents (not queries) at inference time
    - Uses BERT-based encoder
    - Produces sparse token importance scores
    - Supports IDF-aware penalty
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        dropout: float = 0.1,
    ):
        """
        Initialize SPLADE-doc model.

        Args:
            model_name: Pretrained transformer model name
            dropout: Dropout rate
        """
        super().__init__()

        # Load pretrained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        self.config = self.transformer.config

        # Token importance predictor
        self.token_importance = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.config.hidden_size, 1),
        )

        # ReLU for sparsity (log(1 + ReLU(·)))
        self.relu = nn.ReLU()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]

        Returns:
            sparse_repr: Sparse document representation [batch_size, vocab_size]
            token_weights: Token importance weights [batch_size, seq_len]
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Token-level hidden states [batch_size, seq_len, hidden_size]
        hidden_states = outputs.last_hidden_state

        # Predict token importance [batch_size, seq_len, 1]
        importance_scores = self.token_importance(hidden_states)
        importance_scores = importance_scores.squeeze(-1)  # [batch_size, seq_len]

        # Apply ReLU and log(1+x) for sparsity
        token_weights = torch.log1p(self.relu(importance_scores))

        # Mask out padding tokens
        token_weights = token_weights * attention_mask.float()

        # Create sparse representation by max pooling over token positions
        # For each token ID in vocabulary, take max weight across all positions
        batch_size, seq_len = input_ids.shape
        vocab_size = self.config.vocab_size

        # Use vectorized operations (gradient-safe, no inplace operations)
        # Create one-hot encoding using scatter (non-inplace version)
        one_hot = torch.zeros(
            batch_size, seq_len, vocab_size,
            device=input_ids.device,
            dtype=token_weights.dtype
        ).scatter(2, input_ids.unsqueeze(-1), 1)

        # Apply attention mask
        mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]

        # Broadcast token weights to vocabulary dimension
        # masked_weights: [batch, seq_len, vocab_size]
        masked_weights = token_weights.unsqueeze(-1) * one_hot * mask

        # Max pooling: take max across sequence dimension
        # sparse_repr: [batch, vocab_size]
        sparse_repr, _ = masked_weights.max(dim=1)

        return sparse_repr, token_weights

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
            token_type_ids: Token type IDs

        Returns:
            Sparse document representations [batch_size, vocab_size]
        """
        sparse_repr, _ = self.forward(input_ids, attention_mask, token_type_ids)
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
            tokenizer: Tokenizer for decoding
            k: Number of top tokens

        Returns:
            Dictionary of {token: weight}
        """
        # Get top-k indices and values
        top_k_values, top_k_indices = torch.topk(sparse_repr, k=min(k, sparse_repr.shape[0]))

        # Convert to dictionary
        result = {}
        for idx, val in zip(top_k_indices.tolist(), top_k_values.tolist()):
            if val > 0:  # Only include non-zero weights
                token = tokenizer.decode([idx])
                result[token] = val

        return result


class SPLADEDocWithIDF(SPLADEDoc):
    """
    SPLADE-doc with IDF-aware penalty.

    Applies IDF-based weighting to preserve informative tokens.
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        dropout: float = 0.1,
        idf_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize SPLADE-doc with IDF.

        Args:
            model_name: Pretrained transformer model name
            dropout: Dropout rate
            idf_weights: Pre-computed IDF weights [vocab_size]
        """
        super().__init__(model_name, dropout)

        # IDF weights (will be loaded from corpus statistics)
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
            token_type_ids: Token type IDs
            apply_idf: Whether to apply IDF weighting

        Returns:
            sparse_repr: IDF-weighted sparse representation
            token_weights: Token importance weights
        """
        # Get base sparse representation
        sparse_repr, token_weights = super().forward(
            input_ids, attention_mask, token_type_ids
        )

        # Apply IDF weighting if available
        if apply_idf and self.idf_weights is not None:
            sparse_repr = sparse_repr * self.idf_weights.unsqueeze(0)

        return sparse_repr, token_weights

    def set_idf_weights(self, idf_weights: torch.Tensor):
        """
        Set IDF weights.

        Args:
            idf_weights: IDF weights [vocab_size]
        """
        self.register_buffer('idf_weights', idf_weights)


class SPLADEDocExpansion(nn.Module):
    """
    SPLADE-doc with vocabulary expansion capability.

    Unlike standard SPLADE which only activates input tokens,
    this model can activate ANY token in the vocabulary by using
    the MLM head to predict token relevance scores.

    Key insight: Use BERT's MLM head to project hidden states to
    full vocabulary space, allowing activation of tokens not in input.

    This enables cross-lingual expansion:
    - Input: "머신러닝"
    - Output: [머신, ##닝, machine, learning, ML, ...]
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        dropout: float = 0.1,
        expansion_mode: str = "mlm",
    ):
        """
        Initialize SPLADE-doc with expansion.

        Args:
            model_name: Pretrained transformer model name
            dropout: Dropout rate
            expansion_mode: How to compute vocab-wide scores
                - 'mlm': Use MLM head (recommended)
                - 'projection': Learn a projection to vocab
        """
        super().__init__()

        self.expansion_mode = expansion_mode

        # Load pretrained transformer with LM head
        if expansion_mode == "mlm":
            from transformers import AutoModelForMaskedLM
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.config = self.model.config
            self.transformer = self.model.bert if hasattr(self.model, 'bert') else self.model.roberta
        else:
            self.transformer = AutoModel.from_pretrained(model_name)
            self.config = self.transformer.config
            # Learnable projection to vocabulary
            self.vocab_projection = nn.Linear(
                self.config.hidden_size, self.config.vocab_size
            )

        # ReLU for sparsity
        self.relu = nn.ReLU()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with vocabulary expansion.

        Unlike standard SPLADE, this outputs scores for ALL vocab tokens,
        not just the input tokens.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]

        Returns:
            sparse_repr: Sparse representation [batch_size, vocab_size]
                         Can have non-zero values for ANY token
            token_weights: Per-position weights [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape

        if self.expansion_mode == "mlm":
            # Use MLM head to get logits over full vocabulary
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            # logits: [batch, seq_len, vocab_size]
            logits = outputs.logits
        else:
            # Use transformer + projection
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            hidden_states = outputs.last_hidden_state
            # Project to vocabulary: [batch, seq_len, vocab_size]
            logits = self.vocab_projection(hidden_states)

        # Apply ReLU and log(1+x) for sparsity
        # This gives non-negative scores for all vocab tokens
        token_scores = torch.log1p(self.relu(logits))

        # Mask padding positions
        mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
        token_scores = token_scores * mask

        # Max pooling across sequence positions
        # sparse_repr: [batch, vocab_size]
        sparse_repr, _ = token_scores.max(dim=1)

        # Also compute per-position importance (for analysis)
        token_weights = token_scores.max(dim=-1).values  # [batch, seq_len]

        return sparse_repr, token_weights

    def encode_documents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode documents to sparse representations."""
        sparse_repr, _ = self.forward(input_ids, attention_mask, token_type_ids)
        return sparse_repr


def create_splade_model(
    model_name: str = "bert-base-multilingual-cased",
    use_idf: bool = True,
    idf_weights: Optional[torch.Tensor] = None,
    dropout: float = 0.1,
    use_expansion: bool = False,
    expansion_mode: str = "mlm",
) -> SPLADEDoc:
    """
    Factory function to create SPLADE model.

    Args:
        model_name: Pretrained transformer model name
        use_idf: Whether to use IDF-aware penalty
        idf_weights: Pre-computed IDF weights
        dropout: Dropout rate
        use_expansion: Whether to use vocabulary expansion model
        expansion_mode: Expansion mode ('mlm' or 'projection')

    Returns:
        SPLADE model instance
    """
    if use_expansion:
        return SPLADEDocExpansion(
            model_name=model_name,
            dropout=dropout,
            expansion_mode=expansion_mode,
        )
    elif use_idf:
        return SPLADEDocWithIDF(
            model_name=model_name,
            dropout=dropout,
            idf_weights=idf_weights,
        )
    else:
        return SPLADEDoc(
            model_name=model_name,
            dropout=dropout,
        )
