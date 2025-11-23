"""Discover synonyms using word embeddings and KNN clustering.

This script:
1. Extracts vocabulary from Wikipedia/Namuwiki
2. Generates embeddings for each word using multilingual BERT
3. Uses KNN to find semantically similar words
4. Identifies Korean-English synonym pairs
5. Extracts documents containing these synonym pairs
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter, defaultdict
from tqdm import tqdm

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors
import faiss


class SynonymDiscoverer:
    """Discover synonyms using embeddings and KNN."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize synonym discoverer.

        Args:
            model_name: Pretrained model for embeddings (E5 model)
            device: Device to use
        """
        self.device = device
        print(f"Using device: {device}")

        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # E5 models use special instruction prefixes
        self.use_e5_instructions = "e5" in model_name.lower()

        # Vocabulary and embeddings
        self.korean_vocab = set()
        self.english_vocab = set()
        self.vocab_embeddings = {}
        self.vocab_list = []
        self.embedding_matrix = None

    def extract_vocabulary(
        self,
        data_files: List[str],
        min_freq: int = 5,
        max_vocab: int = 10000,
    ):
        """
        Extract vocabulary from documents.

        Args:
            data_files: List of JSONL files
            min_freq: Minimum frequency for a word
            max_vocab: Maximum vocabulary size
        """
        print("Extracting vocabulary...")

        korean_counter = Counter()
        english_counter = Counter()

        # Korean pattern: 2+ Korean characters
        korean_pattern = re.compile(r'[가-힣]{2,}')
        # English pattern: 2+ English letters
        english_pattern = re.compile(r'\b[a-zA-Z]{2,}\b')

        for file_path in tqdm(data_files, desc="Reading files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            text = data.get('text', '')

                            # Extract Korean words
                            korean_words = korean_pattern.findall(text)
                            korean_counter.update(korean_words)

                            # Extract English words
                            english_words = english_pattern.findall(text)
                            english_words = [w.lower() for w in english_words]
                            english_counter.update(english_words)

                        except json.JSONDecodeError:
                            continue
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                continue

        # Filter by frequency and limit size
        self.korean_vocab = {
            word for word, freq in korean_counter.most_common(max_vocab)
            if freq >= min_freq
        }
        self.english_vocab = {
            word for word, freq in english_counter.most_common(max_vocab)
            if freq >= min_freq
        }

        print(f"✓ Korean vocabulary: {len(self.korean_vocab):,} words")
        print(f"✓ English vocabulary: {len(self.english_vocab):,} words")

    def average_pool(self, last_hidden_states, attention_mask):
        """
        Average pooling for E5 models.

        Args:
            last_hidden_states: Model outputs
            attention_mask: Attention mask

        Returns:
            Pooled embeddings
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def generate_embeddings(self, batch_size: int = 16):
        """
        Generate embeddings for vocabulary.

        Args:
            batch_size: Batch size for encoding (reduced for large model)
        """
        print("Generating embeddings...")

        # Combine vocabularies
        all_vocab = list(self.korean_vocab) + list(self.english_vocab)
        self.vocab_list = all_vocab

        # Add E5 instruction prefix if using E5 models
        if self.use_e5_instructions:
            # For E5 models, add "query: " prefix for better semantic matching
            all_vocab_with_prefix = ["query: " + word for word in all_vocab]
        else:
            all_vocab_with_prefix = all_vocab

        embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(all_vocab), batch_size), desc="Encoding"):
                batch = all_vocab_with_prefix[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors='pt'
                ).to(self.device)

                # Get embeddings
                outputs = self.model(**inputs)

                # Use average pooling for E5 models (recommended)
                if self.use_e5_instructions:
                    pooled_embeddings = self.average_pool(
                        outputs.last_hidden_state,
                        inputs['attention_mask']
                    )
                else:
                    # [CLS] token for other models
                    pooled_embeddings = outputs.last_hidden_state[:, 0, :]

                embeddings.append(pooled_embeddings.cpu().numpy())

        # Concatenate all embeddings
        self.embedding_matrix = np.vstack(embeddings)  # [vocab_size, 768]

        # Store in dict for easy lookup
        for word, embedding in zip(all_vocab, self.embedding_matrix):
            self.vocab_embeddings[word] = embedding

        print(f"✓ Generated {len(self.vocab_list):,} embeddings")
        print(f"✓ Embedding shape: {self.embedding_matrix.shape}")

    def find_knn_synonyms(
        self,
        k: int = 10,
        cross_lingual_only: bool = True,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find K nearest neighbors for each word.

        Args:
            k: Number of neighbors
            cross_lingual_only: Only return cross-lingual pairs (Korean-English)

        Returns:
            Dict mapping word -> [(neighbor, similarity), ...]
        """
        print(f"Finding {k} nearest neighbors...")

        # Build FAISS index for fast similarity search
        dimension = self.embedding_matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)

        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.embedding_matrix.astype('float32')
        faiss.normalize_L2(normalized_embeddings)

        index.add(normalized_embeddings)

        # Search
        distances, indices = index.search(normalized_embeddings, k + 1)  # k+1 to exclude self

        # Build synonym dict
        synonyms = {}

        for i, word in enumerate(tqdm(self.vocab_list, desc="Building synonym dict")):
            neighbors = []

            # Skip first result (self)
            for j in range(1, k + 1):
                neighbor_idx = indices[i][j]
                neighbor_word = self.vocab_list[neighbor_idx]
                similarity = float(distances[i][j])

                # Filter by cross-lingual constraint
                if cross_lingual_only:
                    # Korean word should have English neighbors, vice versa
                    word_is_korean = bool(re.match(r'[가-힣]', word))
                    neighbor_is_korean = bool(re.match(r'[가-힣]', neighbor_word))

                    # Skip if both are same language
                    if word_is_korean == neighbor_is_korean:
                        continue

                neighbors.append((neighbor_word, similarity))

            if neighbors:
                synonyms[word] = neighbors[:k]

        print(f"✓ Found synonyms for {len(synonyms):,} words")
        return synonyms

    def filter_high_confidence_pairs(
        self,
        synonyms: Dict[str, List[Tuple[str, float]]],
        min_similarity: float = 0.7,
        bidirectional: bool = True,
    ) -> List[Tuple[str, str, float]]:
        """
        Filter high-confidence synonym pairs.

        Args:
            synonyms: Dict from find_knn_synonyms
            min_similarity: Minimum cosine similarity
            bidirectional: Require bidirectional matching (A→B and B→A)

        Returns:
            List of (word1, word2, similarity) tuples
        """
        print("Filtering high-confidence pairs...")

        pairs = []
        seen = set()

        for word, neighbors in tqdm(synonyms.items(), desc="Filtering"):
            for neighbor, similarity in neighbors:
                if similarity < min_similarity:
                    continue

                # Create canonical pair (Korean, English) or alphabetical
                word_is_korean = bool(re.match(r'[가-힣]', word))
                neighbor_is_korean = bool(re.match(r'[가-힣]', neighbor))

                if word_is_korean and not neighbor_is_korean:
                    pair = (word, neighbor)
                elif not word_is_korean and neighbor_is_korean:
                    pair = (neighbor, word)
                else:
                    pair = tuple(sorted([word, neighbor]))

                # Skip if already seen
                if pair in seen:
                    continue

                # Check bidirectional matching
                if bidirectional:
                    reverse_found = False
                    if neighbor in synonyms:
                        for rev_neighbor, rev_sim in synonyms[neighbor]:
                            if rev_neighbor == word and rev_sim >= min_similarity:
                                reverse_found = True
                                break

                    if not reverse_found:
                        continue

                seen.add(pair)
                pairs.append((pair[0], pair[1], similarity))

        # Sort by similarity
        pairs.sort(key=lambda x: x[2], reverse=True)

        print(f"✓ Found {len(pairs):,} high-confidence pairs")
        return pairs

    def save_synonym_pairs(
        self,
        pairs: List[Tuple[str, str, float]],
        output_path: str,
    ):
        """Save synonym pairs to JSON."""
        output = []
        for korean, english, similarity in pairs:
            output.append({
                "korean": korean,
                "english": english,
                "similarity": similarity,
                "method": "knn_embedding"
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"✓ Saved {len(output):,} synonym pairs to {output_path}")


def main():
    """Main entry point."""

    # Initialize with E5-large model
    discoverer = SynonymDiscoverer(
        model_name="intfloat/multilingual-e5-large",
        device="cuda"
    )

    # Extract vocabulary from Namuwiki
    data_files = list(Path("dataset/namuwiki").glob("*.jsonl"))
    print(f"Found {len(data_files)} data files")

    discoverer.extract_vocabulary(
        data_files=data_files,
        min_freq=10,      # Minimum 10 occurrences
        max_vocab=5000,   # Top 5K Korean + 5K English words
    )

    # Generate embeddings
    discoverer.generate_embeddings(batch_size=64)

    # Find KNN synonyms
    synonyms = discoverer.find_knn_synonyms(
        k=20,                      # Find 20 nearest neighbors
        cross_lingual_only=True,   # Only Korean-English pairs
    )

    # Filter high-confidence pairs
    pairs = discoverer.filter_high_confidence_pairs(
        synonyms=synonyms,
        min_similarity=0.65,       # Minimum 0.65 cosine similarity
        bidirectional=True,        # Require A→B and B→A
    )

    # Save results
    discoverer.save_synonym_pairs(
        pairs=pairs,
        output_path="dataset/synonyms/knn_discovered_synonyms.json"
    )

    # Print top examples
    print("\n=== Top 20 Discovered Synonym Pairs ===")
    for i, (korean, english, sim) in enumerate(pairs[:20], 1):
        print(f"{i:2d}. {korean:15s} ↔ {english:20s} (sim: {sim:.3f})")


if __name__ == "__main__":
    main()
