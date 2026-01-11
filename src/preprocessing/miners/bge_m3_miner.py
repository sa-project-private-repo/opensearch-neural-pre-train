"""BGE-M3 based hard negative mining with FAISS."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MiningResult:
    """Result of hard negative mining."""

    query: str
    positive: str
    negatives: List[str]
    negative_scores: List[float]


class BGEM3HardNegativeMiner:
    """BGE-M3 based hard negative miner using FAISS for efficient search.

    Uses BGE-M3 embeddings to find semantically similar but non-relevant
    documents as hard negatives. FAISS enables efficient similarity search
    over large document collections.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        max_length: int = 192,
        device: Optional[str] = None,
        use_fp16: bool = True,
    ):
        """Initialize BGE-M3 miner.

        Args:
            model_name: HuggingFace model name for embeddings
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
            device: Device to use (cuda/cpu, auto-detected if None)
            use_fp16: Use FP16 for faster inference
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = use_fp16

        # Lazy loading
        self._model = None
        self._index = None
        self._document_pool: List[str] = []
        self._document_embeddings: Optional[np.ndarray] = None

        # Device selection
        if device is None:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"BGE-M3 miner initialized (device={self.device})")

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """Load BGE-M3 model."""
        try:
            from FlagEmbedding import BGEM3FlagModel

            logger.info(f"Loading BGE-M3 model: {self.model_name}")
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device,
            )
            logger.info("BGE-M3 model loaded successfully")
        except ImportError:
            logger.error("FlagEmbedding not installed. Run: pip install FlagEmbedding")
            raise

    def build_index(
        self,
        documents: List[str],
        index_type: str = "flat",
        nlist: int = 100,
    ) -> None:
        """Build FAISS index from documents.

        Args:
            documents: List of documents to index
            index_type: FAISS index type ('flat' or 'ivf')
            nlist: Number of clusters for IVF index
        """
        try:
            import faiss
        except ImportError:
            logger.error("FAISS not installed. Run: pip install faiss-cpu")
            raise

        logger.info(f"Building FAISS index for {len(documents)} documents")
        self._document_pool = documents

        # Encode all documents
        self._document_embeddings = self._encode_batch(documents)
        dimension = self._document_embeddings.shape[1]

        # Create index
        if index_type == "flat":
            self._index = faiss.IndexFlatIP(dimension)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            self._index = faiss.IndexIVFFlat(
                quantizer,
                dimension,
                min(nlist, len(documents)),
                faiss.METRIC_INNER_PRODUCT,
            )
            self._index.train(self._document_embeddings)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Normalize for cosine similarity
        faiss.normalize_L2(self._document_embeddings)
        self._index.add(self._document_embeddings)

        logger.info(f"FAISS index built: {self._index.ntotal} vectors")

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of texts to encode

        Returns:
            Numpy array of embeddings
        """
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self.model.encode(
                batch,
                batch_size=self.batch_size,
                max_length=self.max_length,
            )
            # BGE-M3 returns dict with 'dense_vecs'
            if isinstance(embeddings, dict):
                embeddings = embeddings["dense_vecs"]
            all_embeddings.append(embeddings)

            if (i + self.batch_size) % 10000 == 0:
                logger.info(f"Encoded {i + self.batch_size}/{len(texts)} texts")

        return np.vstack(all_embeddings).astype("float32")

    def mine_negatives(
        self,
        queries: List[str],
        positives: List[str],
        num_negatives: int = 5,
        min_score: float = 0.3,
        max_score: float = 0.85,
    ) -> List[MiningResult]:
        """Mine hard negatives for query-positive pairs.

        Args:
            queries: List of query texts
            positives: List of positive document texts
            num_negatives: Number of negatives per query
            min_score: Minimum similarity score (too easy negatives filtered)
            max_score: Maximum similarity score (false negatives filtered)

        Returns:
            List of MiningResult with mined negatives
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        import faiss

        logger.info(f"Mining negatives for {len(queries)} queries")

        # Encode queries
        query_embeddings = self._encode_batch(queries)
        faiss.normalize_L2(query_embeddings)

        # Create positive set for filtering
        positive_set = set(positives)

        # Search for similar documents
        # Search more than needed to filter by score range
        k = min(num_negatives * 10, self._index.ntotal)
        scores, indices = self._index.search(query_embeddings, k)

        results = []
        for i, (query, positive) in enumerate(zip(queries, positives)):
            negatives = []
            negative_scores = []

            for j, (score, idx) in enumerate(zip(scores[i], indices[i])):
                if len(negatives) >= num_negatives:
                    break

                candidate = self._document_pool[idx]

                # Filter conditions
                if candidate == positive:  # Not the positive itself
                    continue
                if candidate == query:  # Not the query itself
                    continue
                if candidate in positive_set:  # Not any positive
                    continue
                if score < min_score:  # Not too easy
                    continue
                if score > max_score:  # Not false negative
                    continue

                negatives.append(candidate)
                negative_scores.append(float(score))

            results.append(
                MiningResult(
                    query=query,
                    positive=positive,
                    negatives=negatives,
                    negative_scores=negative_scores,
                )
            )

        found_count = sum(1 for r in results if r.negatives)
        avg_negatives = (
            sum(len(r.negatives) for r in results) / len(results) if results else 0
        )
        logger.info(
            f"Mining complete: {found_count}/{len(queries)} queries found negatives "
            f"(avg {avg_negatives:.1f} per query)"
        )

        return results

    def mine_for_triplets(
        self,
        triplets: List["Triplet"],
        num_negatives: int = 1,
        min_score: float = 0.3,
        max_score: float = 0.85,
        skip_complete: bool = True,
    ) -> List["Triplet"]:
        """Mine hard negatives for triplets that need them.

        Args:
            triplets: List of Triplet objects
            num_negatives: Number of negatives to mine
            min_score: Minimum similarity score
            max_score: Maximum similarity score
            skip_complete: Skip triplets that already have negatives

        Returns:
            Updated list of triplets with mined negatives
        """
        from src.preprocessing.converters.base import Triplet

        # Filter triplets needing negatives
        if skip_complete:
            needs_mining = [t for t in triplets if t.negative is None]
            complete = [t for t in triplets if t.negative is not None]
        else:
            needs_mining = triplets
            complete = []

        if not needs_mining:
            logger.info("No triplets need mining")
            return triplets

        logger.info(f"Mining negatives for {len(needs_mining)} triplets")

        # Build index from all positives if not already built
        if self._index is None:
            all_docs = list(set(t.positive for t in triplets))
            self.build_index(all_docs)

        # Mine negatives
        queries = [t.query for t in needs_mining]
        positives = [t.positive for t in needs_mining]
        results = self.mine_negatives(
            queries, positives, num_negatives, min_score, max_score
        )

        # Update triplets
        updated = []
        for triplet, result in zip(needs_mining, results):
            if result.negatives:
                # Take first negative
                updated.append(
                    Triplet(
                        query=triplet.query,
                        positive=triplet.positive,
                        negative=result.negatives[0],
                        pair_type=triplet.pair_type,
                        difficulty="hard",  # Mined negatives are hard
                        source=triplet.source,
                        metadata={
                            **triplet.metadata,
                            "negative_score": result.negative_scores[0],
                        },
                    )
                )
            else:
                # Keep original (no negative found)
                updated.append(triplet)

        return complete + updated

    def clear_index(self) -> None:
        """Clear the FAISS index to free memory."""
        self._index = None
        self._document_pool = []
        self._document_embeddings = None
        logger.info("FAISS index cleared")
