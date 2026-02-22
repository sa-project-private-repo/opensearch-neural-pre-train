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
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing sparse representations.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs (pass through to avoid HF internal buffer issues)

        Returns:
            sparse_repr: Sparse document representation [batch_size, vocab_size]
            token_weights: Per-position importance [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape

        if self.use_mlm_head:
            # Use MLM head for full vocabulary coverage
            # Pass token_type_ids to avoid HuggingFace's internal buffer inplace ops
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
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


class ContextGate(nn.Module):
    """
    Context-based gating module for sparse expansion (V28).

    Computes document-level context vector and projects it to a
    vocab-sized gate that modulates token activations.

    Architecture:
        Hidden States -> Attention Pooling -> Context Vector
        Context Vector -> Gate MLP -> Vocab-sized Gate [0, 1]
    """

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 250002,
        gate_hidden: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize context gate.

        Args:
            hidden_size: Transformer hidden dimension
            vocab_size: Vocabulary size for gate output
            gate_hidden: Hidden dimension for gate MLP
            num_heads: Number of attention heads for context pooling
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Multi-head self-attention for context pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Learnable query for pooling
        self.context_query = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Gate MLP: hidden_size -> gate_hidden -> vocab_size
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, vocab_size),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute context gate.

        Args:
            hidden_states: Transformer outputs [batch, seq_len, hidden]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            gate: Context-dependent gate [batch, vocab_size]
        """
        batch_size = hidden_states.shape[0]

        # Expand context query for batch
        query = self.context_query.expand(batch_size, -1, -1)

        # Create attention mask for multi-head attention
        # True = masked (ignore), False = attend
        key_padding_mask = (attention_mask == 0)

        # Attention pooling: query attends to all hidden states
        context, _ = self.attention(
            query=query,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
        )

        # context: [batch, 1, hidden] -> [batch, hidden]
        context = context.squeeze(1)
        context = self.layer_norm(context)

        # Project to vocab-sized gate
        gate = self.gate_proj(context)  # [batch, vocab_size]

        return gate


class SPLADEDocContextGated(SPLADEDocXLMR):
    """
    Context-Gated SPLADE model for V28.

    Extends SPLADEDocXLMR with a context gate that modulates
    token activations based on document-level context.

    This enables context-dependent sparse representations where
    the same keyword activates different tokens based on context:
    - "출근했는데 점심 메뉴" -> 회사, 직장인, 비빔밥
    - "학교를 갔는데 점심 메뉴" -> 학생, 급식, 도시락

    Architecture:
        Input -> XLM-R -> MLM Logits
                   |
                   +-> Context Gate -> Gate [batch, vocab]
                           |
        Gated Logits = MLM Logits * Gate.unsqueeze(1)
                           |
        Sparse = max_pool(ReLU(log1p(Gated Logits)))
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        dropout: float = 0.1,
        use_mlm_head: bool = True,
        gate_hidden: int = 256,
        gate_heads: int = 4,
    ):
        """
        Initialize context-gated SPLADE model.

        Args:
            model_name: XLM-RoBERTa model name
            dropout: Dropout rate
            use_mlm_head: Use MLM head for expansion
            gate_hidden: Hidden dimension for context gate
            gate_heads: Number of attention heads in context gate
        """
        super().__init__(
            model_name=model_name,
            dropout=dropout,
            use_mlm_head=use_mlm_head,
        )

        # Context gate module
        self.context_gate = ContextGate(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            gate_hidden=gate_hidden,
            num_heads=gate_heads,
            dropout=dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with context-gated sparse representations.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs (pass through to avoid HF internal buffer issues)

        Returns:
            sparse_repr: Context-gated sparse representation [batch, vocab_size]
            token_weights: Per-position importance [batch, seq_len]
        """
        # Get MLM logits from base model
        # Pass token_type_ids to avoid HuggingFace's internal buffer inplace ops
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden]

        # Compute context gate
        gate = self.context_gate(hidden_states, attention_mask)  # [batch, vocab_size]

        # Apply gate to logits (broadcast over sequence dimension)
        # gated_logits[b, s, v] = logits[b, s, v] * gate[b, v]
        gated_logits = logits * gate.unsqueeze(1)

        # Standard sparsification
        sparse_scores = torch.log1p(self.relu(gated_logits))

        # Mask padding positions
        mask = attention_mask.unsqueeze(-1).float()
        sparse_scores = sparse_scores * mask

        # Max pooling across sequence
        sparse_repr, _ = sparse_scores.max(dim=1)  # [batch, vocab_size]

        # Per-position weights for analysis
        token_weights = sparse_scores.max(dim=-1).values  # [batch, seq_len]

        return sparse_repr, token_weights

    def get_context_gate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get context gate values (for analysis/debugging).

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            gate: Context gate values [batch, vocab_size]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]

        return self.context_gate(hidden_states, attention_mask)


class SPLADEDocV29(SPLADEDocXLMR):
    """
    SPLADE v29 model with configurable pooling strategy.

    Based on SPLADE v2 paper:
    - Max pooling (Eq. 6): w_j = max_i log(1 + ReLU(w_ij))
    - Sum pooling (original): w_j = sum_i log(1 + ReLU(w_ij))

    Max pooling improves MRR by ~2 points in original paper.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        dropout: float = 0.1,
        use_mlm_head: bool = True,
        pooling: str = "max",
    ):
        """
        Initialize V29 SPLADE model.

        Args:
            model_name: XLM-RoBERTa model name
            dropout: Dropout rate
            use_mlm_head: Use MLM head for vocabulary expansion
            pooling: Pooling strategy - "max" (SPLADE v2) or "sum" (original)
        """
        super().__init__(
            model_name=model_name,
            dropout=dropout,
            use_mlm_head=use_mlm_head,
        )

        if pooling not in ("max", "sum"):
            raise ValueError(f"pooling must be 'max' or 'sum', got {pooling}")

        self.pooling = pooling

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with configurable pooling.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs (pass through to avoid HF buffer issues)

        Returns:
            sparse_repr: Sparse document representation [batch_size, vocab_size]
            token_weights: Per-position importance [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape

        # Pass token_type_ids to base model
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if self.use_mlm_head:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits  # [batch, seq_len, vocab_size]
        else:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = outputs.last_hidden_state
            importance = self.token_importance(hidden_states).squeeze(-1)
            logits = self._create_input_only_logits(input_ids, importance, attention_mask)

        # Apply ReLU and log(1+x) for sparsity
        sparse_scores = torch.log1p(self.relu(logits))

        # Mask padding positions
        mask = attention_mask.unsqueeze(-1).float()
        sparse_scores = sparse_scores * mask

        # Pooling across sequence positions
        if self.pooling == "max":
            # SPLADE v2 style max pooling
            sparse_repr, _ = sparse_scores.max(dim=1)
        else:
            # Original sum pooling
            sparse_repr = sparse_scores.sum(dim=1)

        # Per-position token weights
        token_weights = sparse_scores.max(dim=-1).values

        return sparse_repr, token_weights


class SPLADEDocV29ContextGated(SPLADEDocContextGated):
    """
    Context-gated SPLADE v29 model with configurable pooling.

    Combines V28's context gate with V29's max pooling.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        dropout: float = 0.1,
        use_mlm_head: bool = True,
        gate_hidden: int = 256,
        gate_heads: int = 4,
        pooling: str = "max",
    ):
        """
        Initialize context-gated V29 model.

        Args:
            model_name: XLM-RoBERTa model name
            dropout: Dropout rate
            use_mlm_head: Use MLM head for expansion
            gate_hidden: Hidden dimension for context gate
            gate_heads: Number of attention heads in context gate
            pooling: Pooling strategy - "max" or "sum"
        """
        super().__init__(
            model_name=model_name,
            dropout=dropout,
            use_mlm_head=use_mlm_head,
            gate_hidden=gate_hidden,
            gate_heads=gate_heads,
        )

        if pooling not in ("max", "sum"):
            raise ValueError(f"pooling must be 'max' or 'sum', got {pooling}")

        self.pooling = pooling

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with context gating and configurable pooling.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs

        Returns:
            sparse_repr: Context-gated sparse representation [batch, vocab_size]
            token_weights: Per-position importance [batch, seq_len]
        """
        # Get MLM logits and hidden states
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]

        # Compute context gate
        gate = self.context_gate(hidden_states, attention_mask)

        # Apply gate to logits
        gated_logits = logits * gate.unsqueeze(1)

        # Sparsification
        sparse_scores = torch.log1p(self.relu(gated_logits))

        # Mask padding
        mask = attention_mask.unsqueeze(-1).float()
        sparse_scores = sparse_scores * mask

        # Pooling
        if self.pooling == "max":
            sparse_repr, _ = sparse_scores.max(dim=1)
        else:
            sparse_repr = sparse_scores.sum(dim=1)

        # Per-position weights
        token_weights = sparse_scores.max(dim=-1).values

        return sparse_repr, token_weights


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


def load_splade_context_gated(
    checkpoint_path: str,
    model_name: str = "xlm-roberta-base",
    gate_hidden: int = 256,
    gate_heads: int = 4,
    device: str = "cuda",
) -> SPLADEDocContextGated:
    """
    Load trained context-gated SPLADE model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        model_name: Base model name
        gate_hidden: Gate hidden dimension
        gate_heads: Gate attention heads
        device: Target device

    Returns:
        Loaded SPLADEDocContextGated model
    """
    model = SPLADEDocContextGated(
        model_name=model_name,
        gate_hidden=gate_hidden,
        gate_heads=gate_heads,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

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


def load_unified_encoder(
    checkpoint_path: str,
    model_name: str = "xlm-roberta-base",
    gate_hidden: int = 256,
    gate_heads: int = 4,
    dense_output_size: int = 768,
    device: str = "cuda",
):
    """
    Load trained UnifiedEncoder (sparse + dense) model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        model_name: Base XLM-RoBERTa model name
        gate_hidden: Hidden dimension for context gate
        gate_heads: Number of attention heads in context gate
        dense_output_size: Dense head output embedding dimension
        device: Target device

    Returns:
        Loaded UnifiedEncoder model in eval mode
    """
    # Import here to avoid circular imports
    from src.model.unified_encoder import UnifiedEncoder

    model = UnifiedEncoder(
        model_name=model_name,
        gate_hidden=gate_hidden,
        gate_heads=gate_heads,
        dense_output_size=dense_output_size,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

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


def load_splade_v29(
    checkpoint_path: str,
    model_name: str = "xlm-roberta-base",
    pooling: str = "max",
    use_context_gate: bool = False,
    gate_hidden: int = 256,
    gate_heads: int = 4,
    device: str = "cuda",
) -> Union[SPLADEDocV29, SPLADEDocV29ContextGated]:
    """
    Load trained V29 SPLADE model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        model_name: Base model name
        pooling: Pooling strategy ("max" or "sum")
        use_context_gate: Whether model uses context gate
        gate_hidden: Gate hidden dimension
        gate_heads: Gate attention heads
        device: Target device

    Returns:
        Loaded V29 model
    """
    if use_context_gate:
        model = SPLADEDocV29ContextGated(
            model_name=model_name,
            pooling=pooling,
            gate_hidden=gate_hidden,
            gate_heads=gate_heads,
        )
    else:
        model = SPLADEDocV29(
            model_name=model_name,
            pooling=pooling,
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

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
