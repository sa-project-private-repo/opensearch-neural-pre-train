"""
BGE-M3 Teacher model for knowledge distillation.

BGE-M3 is a state-of-the-art multilingual embedding model that
serves as an excellent teacher for training sparse retrieval models.

Key features:
- Multi-lingual (100+ languages)
- Dense 1024-dim embeddings
- Strong semantic understanding
- Can be used for ranking soft labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from sentence_transformers import SentenceTransformer


class BGEM3Teacher(nn.Module):
    """
    BGE-M3 as teacher model for knowledge distillation.

    Provides:
    - Dense embeddings for similarity scoring
    - Soft labels for contrastive distillation
    - Ranking scores for ListNet/ListMLE distillation
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        max_length: int = 512,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize BGE-M3 teacher.

        Args:
            model_name: HuggingFace model name
            device: Target device
            max_length: Maximum sequence length
            normalize_embeddings: Whether to L2 normalize embeddings
        """
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings

        # Load BGE-M3 model
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_length

        # Freeze teacher parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Encode texts to dense embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Encoding batch size
            show_progress: Show progress bar

        Returns:
            Dense embeddings [num_texts, 1024]
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_tensor=True,
        )

        return embeddings.to(self.device)

    @torch.no_grad()
    def compute_similarity(
        self,
        query_embeds: torch.Tensor,
        doc_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity matrix.

        Args:
            query_embeds: Query embeddings [batch_size, dim]
            doc_embeds: Document embeddings [num_docs, dim]

        Returns:
            Similarity matrix [batch_size, num_docs]
        """
        # Normalize if not already normalized
        if self.normalize_embeddings:
            query_embeds = F.normalize(query_embeds, p=2, dim=-1)
            doc_embeds = F.normalize(doc_embeds, p=2, dim=-1)

        # Cosine similarity via matrix multiplication
        similarity = torch.mm(query_embeds, doc_embeds.t())

        return similarity

    @torch.no_grad()
    def get_ranking_scores(
        self,
        queries: List[str],
        documents: List[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Get teacher ranking scores for query-document pairs.

        Args:
            queries: List of query texts
            documents: List of document texts (same length as queries)
            batch_size: Encoding batch size

        Returns:
            Similarity scores [num_pairs]
        """
        assert len(queries) == len(documents), "Must have same number of queries and docs"

        # Encode in batches
        query_embeds = self.encode(queries, batch_size=batch_size)
        doc_embeds = self.encode(documents, batch_size=batch_size)

        # Compute pairwise similarity
        similarity = (query_embeds * doc_embeds).sum(dim=-1)

        return similarity

    @torch.no_grad()
    def get_soft_labels(
        self,
        query: str,
        positive_doc: str,
        negative_docs: List[str],
        temperature: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get soft labels for contrastive distillation.

        Args:
            query: Query text
            positive_doc: Positive document text
            negative_docs: List of negative document texts
            temperature: Softmax temperature (higher = softer)

        Returns:
            soft_labels: Softmax distribution over [positive, *negatives]
            raw_scores: Raw similarity scores
        """
        # Encode all texts
        query_embed = self.encode(query)
        all_docs = [positive_doc] + negative_docs
        doc_embeds = self.encode(all_docs)

        # Compute similarities
        similarities = self.compute_similarity(query_embed, doc_embeds)
        similarities = similarities.squeeze(0)  # [num_docs]

        # Apply temperature and softmax
        soft_labels = F.softmax(similarities / temperature, dim=-1)

        return soft_labels, similarities

    @torch.no_grad()
    def get_batch_soft_labels(
        self,
        queries: List[str],
        positive_docs: List[str],
        negative_docs: List[List[str]],
        temperature: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get soft labels for a batch of triplets.

        Args:
            queries: List of query texts [batch_size]
            positive_docs: List of positive docs [batch_size]
            negative_docs: List of lists of negatives [batch_size, num_neg]
            temperature: Softmax temperature

        Returns:
            soft_labels: [batch_size, 1 + num_neg]
            raw_scores: [batch_size, 1 + num_neg]
        """
        batch_size = len(queries)
        num_neg = len(negative_docs[0]) if negative_docs else 0

        # Encode queries
        query_embeds = self.encode(queries)

        # Encode all documents (positive + negatives for each query)
        all_soft_labels = []
        all_raw_scores = []

        for i in range(batch_size):
            # Get documents for this query
            docs = [positive_docs[i]] + (negative_docs[i] if negative_docs else [])
            doc_embeds = self.encode(docs)

            # Compute similarity
            similarity = self.compute_similarity(
                query_embeds[i:i+1], doc_embeds
            ).squeeze(0)

            # Soft labels
            soft_label = F.softmax(similarity / temperature, dim=-1)

            all_soft_labels.append(soft_label)
            all_raw_scores.append(similarity)

        # Stack results
        soft_labels = torch.stack(all_soft_labels)
        raw_scores = torch.stack(all_raw_scores)

        return soft_labels, raw_scores

    @torch.no_grad()
    def get_in_batch_soft_labels(
        self,
        queries: List[str],
        documents: List[str],
        temperature: float = 2.0,
    ) -> torch.Tensor:
        """
        Get soft labels using in-batch negatives.

        Each query uses all other documents as negatives.

        Args:
            queries: List of query texts [batch_size]
            documents: List of document texts [batch_size]
            temperature: Softmax temperature

        Returns:
            soft_labels: [batch_size, batch_size]
        """
        # Encode all texts
        query_embeds = self.encode(queries)
        doc_embeds = self.encode(documents)

        # Full similarity matrix
        similarity = self.compute_similarity(query_embeds, doc_embeds)

        # Apply temperature and softmax (row-wise)
        soft_labels = F.softmax(similarity / temperature, dim=-1)

        return soft_labels

    def to(self, device: str) -> "BGEM3Teacher":
        """Move model to device."""
        self.device = device
        self.model = self.model.to(device)
        return self


def create_bge_m3_teacher(
    model_name: str = "BAAI/bge-m3",
    device: str = "cuda",
    max_length: int = 512,
) -> BGEM3Teacher:
    """
    Factory function to create BGE-M3 teacher.

    Args:
        model_name: BGE-M3 model variant
        device: Target device
        max_length: Maximum sequence length

    Returns:
        BGEM3Teacher instance
    """
    return BGEM3Teacher(
        model_name=model_name,
        device=device,
        max_length=max_length,
    )


class KDLossWithBGEM3(nn.Module):
    """
    Knowledge distillation loss using BGE-M3 as teacher.

    Combines:
    - KL divergence for distribution matching
    - MSE for score alignment
    - Margin loss for ranking preservation
    """

    def __init__(
        self,
        teacher: BGEM3Teacher,
        kl_weight: float = 1.0,
        mse_weight: float = 0.5,
        temperature: float = 2.0,
    ):
        """
        Initialize KD loss.

        Args:
            teacher: BGE-M3 teacher model
            kl_weight: Weight for KL divergence loss
            mse_weight: Weight for MSE loss
            temperature: Temperature for soft labels
        """
        super().__init__()

        self.teacher = teacher
        self.kl_weight = kl_weight
        self.mse_weight = mse_weight
        self.temperature = temperature

    def forward(
        self,
        student_scores: torch.Tensor,
        queries: List[str],
        documents: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute KD loss.

        Args:
            student_scores: Student similarity scores [batch_size, batch_size]
            queries: Query texts
            documents: Document texts

        Returns:
            Dict with 'total', 'kl', and 'mse' losses
        """
        # Get teacher soft labels
        with torch.no_grad():
            teacher_soft_labels = self.teacher.get_in_batch_soft_labels(
                queries, documents, temperature=self.temperature
            )

        # Student soft predictions
        student_log_probs = F.log_softmax(
            student_scores / self.temperature, dim=-1
        )

        # KL divergence: KL(teacher || student)
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_soft_labels,
            reduction="batchmean",
        )

        # MSE on raw scores
        teacher_scores = self.teacher.compute_similarity(
            self.teacher.encode(queries),
            self.teacher.encode(documents),
        )
        mse_loss = F.mse_loss(student_scores, teacher_scores)

        # Total loss (scale KL by T^2 for gradient compensation)
        total_loss = (
            self.kl_weight * (self.temperature ** 2) * kl_loss
            + self.mse_weight * mse_loss
        )

        return {
            "total": total_loss,
            "kl": kl_loss,
            "mse": mse_loss,
        }
