"""
Korean Neural Sparse Encoder for OpenSearch.

This model is based on skt/A.X-Encoder-base and fine-tuned for Korean term expansion
in neural sparse retrieval tasks, specifically for legal and medical domains.

Usage:
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("sewoong/korean-neural-sparse-encoder-v1")
    model = AutoModel.from_pretrained(
        "sewoong/korean-neural-sparse-encoder-v1",
        trust_remote_code=True
    )

    # Encode text
    inputs = tokenizer("검색어", return_tensors="pt", padding=True, truncation=True)
    sparse_repr, token_weights = model(**inputs)
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig


class SPLADEConfig(PretrainedConfig):
    """Configuration for SPLADE model."""

    model_type = "splade"

    def __init__(
        self,
        max_length: int = 64,
        pooling: str = "max",
        activation: str = "log1p_relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.pooling = pooling
        self.activation = activation


class SPLADEModel(PreTrainedModel):
    """
    SPLADE model for Korean sparse retrieval.

    Uses log(1 + ReLU(MLM_logits)) for sparse representation.
    Fine-tuned for Korean synonym expansion including legal and medical domains.
    """

    config_class = SPLADEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.model = AutoModelForMaskedLM.from_config(config)
        self.relu = nn.ReLU()
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            **kwargs: Additional arguments passed to the base model

        Returns:
            sparse_repr: Sparse representation [batch, vocab_size]
            token_weights: Per-token weights [batch, seq_len]
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = outputs.logits  # [batch, seq_len, vocab_size]

        # SPLADE activation: log(1 + ReLU(x))
        token_scores = torch.log1p(self.relu(logits))

        # Mask padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        token_scores = token_scores * mask

        # Max pooling over sequence
        sparse_repr, _ = token_scores.max(dim=1)  # [batch, vocab_size]

        # Token weights for analysis
        token_weights = token_scores.max(dim=-1).values  # [batch, seq_len]

        return sparse_repr, token_weights

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text to sparse representation.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Sparse representation [batch, vocab_size]
        """
        sparse_repr, _ = self.forward(input_ids, attention_mask)
        return sparse_repr

    def get_top_tokens(
        self,
        sparse_repr: torch.Tensor,
        tokenizer,
        top_k: int = 20,
    ) -> list:
        """
        Get top-k activated tokens from sparse representation.

        Args:
            sparse_repr: Sparse representation [vocab_size] or [1, vocab_size]
            tokenizer: Tokenizer instance
            top_k: Number of top tokens to return

        Returns:
            List of (token, weight) tuples
        """
        if sparse_repr.dim() == 2:
            sparse_repr = sparse_repr[0]

        # Get top-k indices and values
        top_values, top_indices = sparse_repr.topk(top_k)

        result = []
        for idx, val in zip(top_indices.tolist(), top_values.tolist()):
            if val > 0:
                token = tokenizer.decode([idx]).strip()
                result.append((token, val))

        return result
