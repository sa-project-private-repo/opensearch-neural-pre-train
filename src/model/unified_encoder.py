"""
Unified Sparse + Dense Encoder for Korean neural sparse model (Issue #17).

Extends SPLADEDocContextGated with a dense retrieval head alongside
the existing sparse head, enabling dual retrieval (sparse + dense).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.model.splade_xlmr import SPLADEDocContextGated


class DenseHead(nn.Module):
    """Dense retrieval head using CLS pooling."""

    def __init__(self, hidden_size: int = 768, output_size: int = 768):
        """
        Initialize dense retrieval head.

        Args:
            hidden_size: Input hidden dimension from transformer
            output_size: Output embedding dimension
        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dense embedding via CLS pooling.

        Args:
            hidden_states: Transformer outputs [batch, seq_len, hidden]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            dense_repr: L2-normalized dense embedding [batch, output_size]
        """
        # CLS token is the first token
        cls_output = hidden_states[:, 0, :]  # [batch, hidden]

        # Project and normalize
        projected = self.linear(cls_output)  # [batch, output_size]
        normalized = self.layer_norm(projected)  # [batch, output_size]

        # L2 normalize for cosine similarity
        dense_repr = F.normalize(normalized, p=2, dim=-1)  # [batch, output_size]

        return dense_repr


class UnifiedEncoder(SPLADEDocContextGated):
    """
    Unified sparse + dense encoder on shared XLM-RoBERTa backbone.

    Extends SPLADEDocContextGated with a dense head for dual retrieval:
    - Sparse head: context-gated SPLADE representation [batch, vocab_size]
    - Dense head: CLS-pooled L2-normalized embedding [batch, dense_output_size]

    Both heads share the same XLM-RoBERTa backbone, enabling efficient
    joint training and inference.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        dropout: float = 0.1,
        use_mlm_head: bool = True,
        gate_hidden: int = 256,
        gate_heads: int = 4,
        dense_output_size: int = 768,
    ):
        """
        Initialize unified sparse + dense encoder.

        Args:
            model_name: XLM-RoBERTa model name
            dropout: Dropout rate
            use_mlm_head: Use MLM head for sparse vocabulary expansion
            gate_hidden: Hidden dimension for context gate MLP
            gate_heads: Number of attention heads in context gate
            dense_output_size: Output dimension for dense embeddings
        """
        super().__init__(
            model_name=model_name,
            dropout=dropout,
            use_mlm_head=use_mlm_head,
            gate_hidden=gate_hidden,
            gate_heads=gate_heads,
        )

        self.dense_output_size = dense_output_size
        self.dense_head = DenseHead(
            hidden_size=self.hidden_size,
            output_size=dense_output_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass producing both sparse and dense representations.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs (pass through for HF compat)

        Returns:
            sparse_repr: Context-gated sparse representation [batch, vocab_size]
            token_weights: Per-position token importance [batch, seq_len]
            dense_repr: L2-normalized dense embedding [batch, dense_output_size]
        """
        # Get MLM logits and hidden states from shared backbone
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden]

        # === Sparse path (same as SPLADEDocContextGated) ===
        gate = self.context_gate(hidden_states, attention_mask)  # [batch, vocab_size]
        gated_logits = logits * gate.unsqueeze(1)  # [batch, seq_len, vocab_size]
        sparse_scores = torch.log1p(self.relu(gated_logits))

        # Mask padding positions
        mask = attention_mask.unsqueeze(-1).float()
        sparse_scores = sparse_scores * mask

        # Max pooling across sequence
        sparse_repr, _ = sparse_scores.max(dim=1)  # [batch, vocab_size]
        token_weights = sparse_scores.max(dim=-1).values  # [batch, seq_len]

        # === Dense path ===
        dense_repr = self.dense_head(hidden_states, attention_mask)  # [batch, output_size]

        return sparse_repr, token_weights, dense_repr

    def encode_sparse(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return only sparse representation.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs

        Returns:
            sparse_repr: Sparse representation [batch, vocab_size]
        """
        sparse_repr, _, _ = self.forward(input_ids, attention_mask, token_type_ids)
        return sparse_repr

    def encode_dense(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return only dense representation.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs

        Returns:
            dense_repr: L2-normalized dense embedding [batch, dense_output_size]
        """
        _, _, dense_repr = self.forward(input_ids, attention_mask, token_type_ids)
        return dense_repr
