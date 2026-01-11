"""MinHash-based deduplication for triplets."""

import hashlib
import logging
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)


class MinHashDeduplicator:
    """Remove duplicate triplets using MinHash LSH.

    Uses MinHash signatures for approximate duplicate detection,
    which is efficient for large datasets compared to exact matching.

    Deduplication is based on (query, positive) pairs to avoid
    training on semantically identical examples.
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.8,
        ngram_size: int = 3,
    ):
        """Initialize deduplicator.

        Args:
            num_perm: Number of permutations for MinHash
            threshold: Jaccard similarity threshold for duplicates
            ngram_size: Size of character n-grams
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size

        # For storing seen signatures
        self._seen_hashes: Set[str] = set()
        self._hash_to_signature: dict = {}

    def _get_ngrams(self, text: str) -> Set[str]:
        """Extract character n-grams from text.

        Args:
            text: Input text

        Returns:
            Set of n-gram strings
        """
        text = text.lower().strip()
        if len(text) < self.ngram_size:
            return {text}

        ngrams = set()
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.add(text[i : i + self.ngram_size])
        return ngrams

    def _compute_minhash(self, ngrams: Set[str]) -> Tuple[int, ...]:
        """Compute MinHash signature for a set of n-grams.

        Args:
            ngrams: Set of n-gram strings

        Returns:
            Tuple of hash values (signature)
        """
        if not ngrams:
            return tuple([0] * self.num_perm)

        signature = []
        for i in range(self.num_perm):
            min_hash = float("inf")
            for ngram in ngrams:
                # Create hash with different seeds
                h = hashlib.md5(f"{i}_{ngram}".encode()).hexdigest()
                hash_val = int(h, 16)
                if hash_val < min_hash:
                    min_hash = hash_val
            signature.append(min_hash)

        return tuple(signature)

    def _jaccard_similarity(
        self, sig1: Tuple[int, ...], sig2: Tuple[int, ...]
    ) -> float:
        """Estimate Jaccard similarity from MinHash signatures.

        Args:
            sig1: First signature
            sig2: Second signature

        Returns:
            Estimated Jaccard similarity
        """
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def _get_pair_hash(self, query: str, positive: str) -> str:
        """Get a hash for the (query, positive) pair.

        Args:
            query: Query text
            positive: Positive text

        Returns:
            Hash string
        """
        combined = f"{query.strip().lower()}|||{positive.strip().lower()}"
        return hashlib.md5(combined.encode()).hexdigest()

    def is_duplicate(self, query: str, positive: str) -> bool:
        """Check if a (query, positive) pair is a duplicate.

        Uses exact hash first, then MinHash for fuzzy matching.

        Args:
            query: Query text
            positive: Positive text

        Returns:
            True if duplicate
        """
        # Exact match check first
        pair_hash = self._get_pair_hash(query, positive)
        if pair_hash in self._seen_hashes:
            return True

        # MinHash similarity check
        combined_text = f"{query} {positive}"
        ngrams = self._get_ngrams(combined_text)
        signature = self._compute_minhash(ngrams)

        # Check against stored signatures
        for stored_hash, stored_sig in self._hash_to_signature.items():
            similarity = self._jaccard_similarity(signature, stored_sig)
            if similarity >= self.threshold:
                return True

        # Not a duplicate, store it
        self._seen_hashes.add(pair_hash)
        self._hash_to_signature[pair_hash] = signature

        return False

    def deduplicate(self, triplets: List["Triplet"]) -> List["Triplet"]:
        """Remove duplicate triplets from a list.

        Args:
            triplets: List of triplets

        Returns:
            Deduplicated list of triplets
        """
        logger.info(f"Deduplicating {len(triplets)} triplets")

        # Reset state for new deduplication run
        self._seen_hashes.clear()
        self._hash_to_signature.clear()

        unique_triplets = []
        duplicates = 0

        for i, triplet in enumerate(triplets):
            if not self.is_duplicate(triplet.query, triplet.positive):
                unique_triplets.append(triplet)
            else:
                duplicates += 1

            if (i + 1) % 100000 == 0:
                logger.info(
                    f"Processed {i + 1}/{len(triplets)} "
                    f"(unique: {len(unique_triplets)}, duplicates: {duplicates})"
                )

        logger.info(
            f"Deduplication complete: {len(triplets)} -> {len(unique_triplets)} "
            f"(removed {duplicates} duplicates, {duplicates / len(triplets) * 100:.1f}%)"
        )

        return unique_triplets

    def clear(self) -> None:
        """Clear stored hashes and signatures."""
        self._seen_hashes.clear()
        self._hash_to_signature.clear()
        logger.info("Deduplicator state cleared")


class ExactDeduplicator:
    """Simple exact-match deduplicator for smaller datasets.

    Faster than MinHash but only catches exact duplicates.
    Use when dataset is small or exact matching is sufficient.
    """

    def __init__(self):
        """Initialize exact deduplicator."""
        self._seen: Set[str] = set()

    def _get_key(self, query: str, positive: str) -> str:
        """Get unique key for a pair."""
        return f"{query.strip()}|||{positive.strip()}"

    def is_duplicate(self, query: str, positive: str) -> bool:
        """Check if pair is an exact duplicate."""
        key = self._get_key(query, positive)
        if key in self._seen:
            return True
        self._seen.add(key)
        return False

    def deduplicate(self, triplets: List["Triplet"]) -> List["Triplet"]:
        """Remove exact duplicate triplets."""
        self._seen.clear()
        unique = []

        for triplet in triplets:
            if not self.is_duplicate(triplet.query, triplet.positive):
                unique.append(triplet)

        logger.info(
            f"Exact deduplication: {len(triplets)} -> {len(unique)} "
            f"(removed {len(triplets) - len(unique)} duplicates)"
        )

        return unique

    def clear(self) -> None:
        """Clear seen pairs."""
        self._seen.clear()
