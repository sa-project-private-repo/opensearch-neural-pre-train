"""Neural Sparse Encoder for cross-lingual retrieval."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel


class NeuralSparseEncoder(nn.Module):
    """
    Neural Sparse Encoder that produces sparse term weights for retrieval.

    Based on SPLADE (Sparse Lexical and Expansion Model) architecture.
    Produces vocabulary-sized sparse vectors where each dimension represents
    a term weight in the vocabulary.
    """

    def __init__(
        self,
        model_name: str = "klue/bert-base",
        max_length: int = 256,
        use_relu: bool = True,
    ):
        """
        Initialize Neural Sparse Encoder.

        Args:
            model_name: HuggingFace model name (klue/bert-base, xlm-roberta-base)
            max_length: Maximum sequence length
            use_relu: Use ReLU activation (for non-negative weights)
        """
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length

        # Load base encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Get model config
        self.hidden_size = self.encoder.config.hidden_size
        self.vocab_size = self.encoder.config.vocab_size

        # Projection layer: hidden_size -> vocab_size
        # This produces term weights for each token in vocabulary
        self.projection = nn.Linear(self.hidden_size, self.vocab_size)

        # Activation function
        self.activation = nn.ReLU() if use_relu else nn.Identity()

        print(f"Initialized NeuralSparseEncoder:")
        print(f"  Base model: {model_name}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Max length: {max_length}")
        print(f"  Activation: {'ReLU' if use_relu else 'None'}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dense: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to produce sparse representations.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_dense: Whether to return dense embeddings as well

        Returns:
            Dictionary containing:
                - sparse_rep: Sparse representation [batch_size, vocab_size]
                - dense_rep (optional): Dense embedding [batch_size, hidden_size]
        """
        # Encode with BERT
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Get token embeddings
        # Shape: [batch_size, seq_len, hidden_size]
        token_embeddings = encoder_outputs.last_hidden_state

        # Project to vocabulary size
        # Shape: [batch_size, seq_len, vocab_size]
        token_weights = self.projection(token_embeddings)

        # Apply activation (ReLU for non-negative weights)
        token_weights = self.activation(token_weights)

        # Max pooling over sequence dimension
        # This gives us the maximum weight for each vocabulary term
        # across all tokens in the sequence
        # Shape: [batch_size, vocab_size]
        sparse_rep, _ = torch.max(token_weights, dim=1)

        outputs = {"sparse_rep": sparse_rep}

        # Optionally return dense representation (CLS token)
        if return_dense:
            dense_rep = encoder_outputs.last_hidden_state[:, 0, :]
            outputs["dense_rep"] = dense_rep

        return outputs

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Encode texts to sparse representations.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            device: Device to use for encoding

        Returns:
            Sparse representations [num_texts, vocab_size]
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        all_sparse_reps = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                # Move to device
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                # Forward pass
                outputs = self.forward(input_ids, attention_mask)
                sparse_rep = outputs["sparse_rep"]

                all_sparse_reps.append(sparse_rep.cpu())

        return torch.cat(all_sparse_reps, dim=0)

    def get_top_k_terms(
        self,
        sparse_rep: torch.Tensor,
        k: int = 10,
    ) -> list[Tuple[str, float]]:
        """
        Get top-k activated terms from sparse representation.

        Args:
            sparse_rep: Sparse representation [vocab_size]
            k: Number of top terms to return

        Returns:
            List of (term, weight) tuples
        """
        # Get top-k indices and values
        top_k_values, top_k_indices = torch.topk(sparse_rep, k=k)

        # Convert to terms
        terms_and_weights = []
        for idx, weight in zip(top_k_indices.tolist(), top_k_values.tolist()):
            term = self.tokenizer.convert_ids_to_tokens([idx])[0]
            terms_and_weights.append((term, weight))

        return terms_and_weights

    def compute_similarity(
        self,
        query_rep: torch.Tensor,
        doc_rep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity between query and document representations.

        Uses dot product similarity for sparse representations.

        Args:
            query_rep: Query sparse representation [batch_size, vocab_size]
            doc_rep: Document sparse representation [batch_size, vocab_size]

        Returns:
            Similarity scores [batch_size]
        """
        # Dot product similarity
        similarity = torch.sum(query_rep * doc_rep, dim=-1)
        return similarity

    def get_sparsity_stats(self, sparse_rep: torch.Tensor) -> Dict[str, float]:
        """
        Compute sparsity statistics for sparse representation.

        Args:
            sparse_rep: Sparse representation [batch_size, vocab_size]

        Returns:
            Dictionary with sparsity metrics
        """
        # Number of non-zero elements
        threshold = 1e-6
        num_nonzero = (sparse_rep > threshold).sum(dim=-1).float()

        # L1 norm (FLOPS proxy)
        l1_norm = sparse_rep.sum(dim=-1)

        # Max value
        max_value = sparse_rep.max(dim=-1).values

        return {
            "avg_nonzero_terms": num_nonzero.mean().item(),
            "avg_l1_norm": l1_norm.mean().item(),
            "avg_max_value": max_value.mean().item(),
            "sparsity_ratio": (
                1.0 - num_nonzero.mean().item() / sparse_rep.shape[-1]
            ),
        }

    def save_pretrained(self, save_path: str) -> None:
        """
        Save model to directory.

        Args:
            save_path: Directory to save model
        """
        import os

        os.makedirs(save_path, exist_ok=True)

        # Save encoder
        self.encoder.save_pretrained(save_path)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)

        # Save projection layer
        torch.save(
            {
                "projection": self.projection.state_dict(),
                "model_name": self.model_name,
                "max_length": self.max_length,
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
            },
            os.path.join(save_path, "neural_sparse_head.pt"),
        )

        print(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(
        cls,
        load_path: str,
        use_hf_hub: bool = True,
    ) -> "NeuralSparseEncoder":
        """
        Load model from directory or HuggingFace Hub.

        Args:
            load_path: Directory path or HuggingFace model ID
            use_hf_hub: If True, try to load from HuggingFace Hub first

        Returns:
            Loaded NeuralSparseEncoder
        """
        import os
        from pathlib import Path

        # Check if it's a local directory
        is_local = Path(load_path).exists()

        if is_local:
            # Load from local directory
            head_path = os.path.join(load_path, "neural_sparse_head.pt")
            if not os.path.exists(head_path):
                raise FileNotFoundError(
                    f"neural_sparse_head.pt not found in {load_path}. "
                    f"Expected at: {head_path}"
                )

            checkpoint = torch.load(head_path, map_location="cpu")

            # Initialize model
            model = cls(
                model_name=checkpoint["model_name"],
                max_length=checkpoint["max_length"],
            )

            # Load projection layer
            model.projection.load_state_dict(checkpoint["projection"])

            print(f"Model loaded from {load_path}")
            return model

        else:
            # Try to load from HuggingFace Hub
            if use_hf_hub:
                print(f"Loading from HuggingFace Hub: {load_path}")
                try:
                    from huggingface_hub import hf_hub_download

                    # Download neural_sparse_head.pt from HF Hub
                    head_path = hf_hub_download(
                        repo_id=load_path,
                        filename="neural_sparse_head.pt",
                    )

                    checkpoint = torch.load(head_path, map_location="cpu")

                    # Initialize model
                    model = cls(
                        model_name=checkpoint["model_name"],
                        max_length=checkpoint["max_length"],
                    )

                    # Load projection layer
                    model.projection.load_state_dict(checkpoint["projection"])

                    print(f"Model loaded from HuggingFace Hub: {load_path}")
                    return model

                except Exception as e:
                    print(
                        f"Failed to load from HuggingFace Hub: {e}\n"
                        f"Attempting to initialize from base model..."
                    )

            # Fallback: Initialize from base model without pretrained head
            print(
                f"Initializing new model from base encoder: {load_path}\n"
                f"Note: Projection layer will be randomly initialized"
            )
            model = cls(model_name=load_path)

            return model


if __name__ == "__main__":
    # Test the encoder
    print("Testing NeuralSparseEncoder...")

    # Initialize
    encoder = NeuralSparseEncoder(model_name="klue/bert-base", max_length=128)

    # Test texts
    texts = [
        "인공지능 모델 학습",
        "machine learning algorithm",
        "검색 시스템 개발",
    ]

    # Encode
    print("\nEncoding texts...")
    sparse_reps = encoder.encode(texts, device=torch.device("cpu"))

    print(f"Sparse representations shape: {sparse_reps.shape}")

    # Get sparsity stats
    stats = encoder.get_sparsity_stats(sparse_reps)
    print(f"\nSparsity statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

    # Get top-k terms for first text
    print(f"\nTop-10 terms for '{texts[0]}':")
    top_terms = encoder.get_top_k_terms(sparse_reps[0], k=10)
    for term, weight in top_terms:
        print(f"  {term:20s}: {weight:.4f}")
