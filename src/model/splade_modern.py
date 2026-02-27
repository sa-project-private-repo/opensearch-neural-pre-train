"""
V33 SPLADE model using ModernBERT (skt/A.X-Encoder-base).

Pure SPLADE-max architecture following the SPLADE v2 paper:
- MLM head logits -> log(1 + ReLU(w_ij)) -> max pooling over sequence
- No context gate, no language filtering, no custom projections
- 50K vocab (48.4% Korean) vs XLM-RoBERTa's 250K

Reference: "SPLADE v2: Sparse Lexical and Expansion Model for
Information Retrieval" (Formal et al., 2021)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Dict, Optional, Tuple


class SPLADEModernBERT(nn.Module):
    """
    SPLADE-max model with ModernBERT backbone.

    Uses the MLM head from skt/A.X-Encoder-base (ModernBERT, 50K vocab)
    to produce sparse vocabulary-level representations via max pooling.

    Architecture:
        input_ids -> ModernBERT encoder -> MLM head -> logits [B, S, V]
        -> log(1 + ReLU(logits)) -> max_pool(dim=seq) -> sparse_repr [B, V]
    """

    def __init__(
        self,
        model_name: str = "skt/A.X-Encoder-base",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.config = self.model.config
        self.relu = nn.ReLU()

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SPLADE-max forward pass.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            token_type_ids: ignored (kept for interface compatibility)

        Returns:
            sparse_repr: [batch, vocab_size] - sparse representation
            token_weights: [batch, seq_len] - per-position max weight
        """
        # MLM head: [batch, seq, vocab]
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        # SPLADE-max: log(1 + ReLU(w_ij))  (Eq. 6 in SPLADE v2)
        sparse_scores = torch.log1p(self.relu(logits))

        # Mask padding positions
        mask = attention_mask.unsqueeze(-1).float()
        sparse_scores = sparse_scores * mask

        # Max pooling over sequence dimension
        sparse_repr, _ = sparse_scores.max(dim=1)  # [batch, vocab]

        # Per-position importance (for monitoring)
        token_weights = sparse_scores.max(dim=-1).values  # [batch, seq]

        return sparse_repr, token_weights

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode to sparse representation (inference shortcut)."""
        sparse_repr, _ = self.forward(input_ids, attention_mask)
        return sparse_repr

    def get_top_k_tokens(
        self,
        sparse_repr: torch.Tensor,
        tokenizer: AutoTokenizer,
        k: int = 50,
    ) -> Dict[str, float]:
        """Get top-k tokens with highest activation weights."""
        top_vals, top_ids = torch.topk(
            sparse_repr, k=min(k, sparse_repr.shape[0])
        )
        result = {}
        for val, idx in zip(top_vals.tolist(), top_ids.tolist()):
            if val > 0:
                token = tokenizer.decode([idx]).strip()
                result[token] = val
        return result
