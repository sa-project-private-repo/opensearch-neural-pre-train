"""Extract documents where KNN-discovered synonyms co-occur.

This script:
1. Loads KNN-discovered synonym pairs
2. Scans Wikipedia/Namuwiki for documents containing both terms
3. Creates training samples (query, pos, negs) where pos contains synonym pairs
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from tqdm import tqdm


class SynonymCooccurrenceExtractor:
    """Extract documents with synonym co-occurrence."""

    def __init__(
        self,
        synonyms_path: str,
        min_similarity: float = 0.65,
    ):
        """
        Initialize extractor.

        Args:
            synonyms_path: Path to KNN-discovered synonyms JSON
            min_similarity: Minimum similarity threshold
        """
        self.synonyms_path = synonyms_path
        self.min_similarity = min_similarity

        # Load synonyms
        self.synonym_pairs = []
        self.korean_to_english = defaultdict(list)
        self.english_to_korean = defaultdict(list)

        self._load_synonyms()

    def _load_synonyms(self):
        """Load synonym pairs from JSON."""
        print(f"Loading synonyms from {self.synonyms_path}...")

        with open(self.synonyms_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Filter by similarity
        for item in data:
            korean = item['korean']
            english = item['english']
            similarity = item['similarity']

            if similarity >= self.min_similarity:
                self.synonym_pairs.append((korean, english, similarity))
                self.korean_to_english[korean].append((english, similarity))
                self.english_to_korean[english].append((korean, similarity))

        print(f"✓ Loaded {len(self.synonym_pairs):,} synonym pairs")
        print(f"✓ {len(self.korean_to_english):,} Korean terms")
        print(f"✓ {len(self.english_to_korean):,} English terms")

    def find_cooccurrence_documents(
        self,
        data_files: List[str],
        max_docs_per_pair: int = 10,
    ) -> List[Dict]:
        """
        Find documents where synonym pairs co-occur.

        Args:
            data_files: List of JSONL data files
            max_docs_per_pair: Maximum documents to collect per synonym pair

        Returns:
            List of documents with metadata
        """
        print("Scanning documents for synonym co-occurrence...")

        # Index: (korean, english) -> list of documents
        pair_documents = defaultdict(list)

        for file_path in tqdm(data_files, desc="Scanning files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            doc = json.loads(line)
                            text = doc.get('text', '').lower()
                            title = doc.get('title', '')

                            # Check each synonym pair
                            for korean, english, similarity in self.synonym_pairs:
                                korean_lower = korean.lower()
                                english_lower = english.lower()

                                # Check if both terms appear in document
                                if korean_lower in text and english_lower in text:
                                    # Avoid duplicates for same pair
                                    if len(pair_documents[(korean, english)]) < max_docs_per_pair:
                                        pair_documents[(korean, english)].append({
                                            'title': title,
                                            'text': doc.get('text', ''),
                                            'korean_term': korean,
                                            'english_term': english,
                                            'similarity': similarity,
                                            'source_id': doc.get('id', ''),
                                            'source_url': doc.get('url', ''),
                                        })

                        except json.JSONDecodeError:
                            continue
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                continue

        # Flatten results
        all_docs = []
        for (korean, english), docs in pair_documents.items():
            all_docs.extend(docs)

        print(f"✓ Found {len(all_docs):,} documents with synonym co-occurrence")
        print(f"✓ Covering {len(pair_documents):,} synonym pairs")

        return all_docs

    def create_training_samples(
        self,
        cooccurrence_docs: List[Dict],
        all_documents: List[Dict],
        num_negatives: int = 7,
        max_samples: int = 15000,
    ) -> List[Dict]:
        """
        Create training samples from co-occurrence documents.

        Args:
            cooccurrence_docs: Documents with synonym co-occurrence
            all_documents: All documents (for sampling negatives)
            num_negatives: Number of negative samples per query
            max_samples: Maximum training samples to generate

        Returns:
            List of training samples
        """
        print("Creating training samples...")

        # Shuffle and limit
        random.shuffle(cooccurrence_docs)
        cooccurrence_docs = cooccurrence_docs[:max_samples]

        training_samples = []

        skipped_samples = 0

        for doc in tqdm(cooccurrence_docs, desc="Generating samples"):
            # Query: use Korean term (primary language)
            query = doc['korean_term']
            korean_lower = doc['korean_term'].lower()
            english_lower = doc['english_term'].lower()

            # Positive document: the co-occurrence document
            pos_text = doc['text']
            # Increased from 1500 to 3000 characters to preserve synonym pairs
            if len(pos_text) > 3000:
                pos_text = pos_text[:3000]

            # CRITICAL: Verify that both synonym terms are still present after truncation
            if korean_lower not in pos_text.lower() or english_lower not in pos_text.lower():
                skipped_samples += 1
                continue  # Skip this sample - synonyms were cut off

            # Negative documents: random sample
            negatives = []
            neg_candidates = random.sample(all_documents, min(num_negatives * 3, len(all_documents)))

            for neg_doc in neg_candidates:
                neg_text = neg_doc.get('text', '')
                if len(neg_text) > 3000:
                    neg_text = neg_text[:3000]

                # Avoid using documents that also contain the synonym pair
                if korean_lower in neg_text.lower() and english_lower in neg_text.lower():
                    continue  # Skip, this is also a positive

                negatives.append(neg_text)

                if len(negatives) >= num_negatives:
                    break

            # Only add if we have enough negatives
            if len(negatives) >= num_negatives:
                training_samples.append({
                    'query': query,
                    'pos': pos_text,
                    'negs': negatives,
                    'metadata': {
                        'korean_term': doc['korean_term'],
                        'english_term': doc['english_term'],
                        'similarity': doc['similarity'],
                        'method': 'knn_cooccurrence',
                    }
                })

        print(f"✓ Generated {len(training_samples):,} training samples")
        if skipped_samples > 0:
            print(f"⚠️  Skipped {skipped_samples:,} samples (synonyms were truncated)")
        return training_samples

    def save_training_samples(
        self,
        samples: List[Dict],
        output_path: str,
    ):
        """Save training samples to JSONL."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"✓ Saved {len(samples):,} samples to {output_path}")


def main():
    """Main entry point."""

    # Initialize extractor
    extractor = SynonymCooccurrenceExtractor(
        synonyms_path="dataset/synonyms/knn_discovered_synonyms.json",
        min_similarity=0.65,
    )

    # Find all data files
    data_files = list(Path("dataset/namuwiki").glob("*.jsonl"))
    print(f"Found {len(data_files)} data files")

    # Find co-occurrence documents
    cooccurrence_docs = extractor.find_cooccurrence_documents(
        data_files=data_files,
        max_docs_per_pair=10,  # Max 10 documents per synonym pair
    )

    # Load all documents for negative sampling
    print("Loading all documents for negative sampling...")
    all_documents = []
    for file_path in tqdm(data_files[:5], desc="Loading docs"):  # Use first 5 files
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    all_documents.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"✓ Loaded {len(all_documents):,} documents for negative sampling")

    # Create training samples
    training_samples = extractor.create_training_samples(
        cooccurrence_docs=cooccurrence_docs,
        all_documents=all_documents,
        num_negatives=7,
        max_samples=15000,
    )

    # Save
    extractor.save_training_samples(
        samples=training_samples,
        output_path="dataset/baseline_samples/train_knn_synonyms.jsonl"
    )

    # Print statistics
    print("\n=== Training Data Statistics ===")
    print(f"Total samples: {len(training_samples):,}")

    # Count unique synonym pairs
    unique_pairs = set()
    for sample in training_samples:
        korean = sample['metadata']['korean_term']
        english = sample['metadata']['english_term']
        unique_pairs.add((korean, english))

    print(f"Unique synonym pairs: {len(unique_pairs):,}")

    # Show examples
    print("\n=== Sample Training Examples ===")
    for i, sample in enumerate(training_samples[:3], 1):
        korean = sample['metadata']['korean_term']
        english = sample['metadata']['english_term']
        similarity = sample['metadata']['similarity']

        print(f"\nExample {i}:")
        print(f"  Synonym Pair: {korean} ↔ {english} (sim: {similarity:.3f})")
        print(f"  Query: {sample['query']}")
        print(f"  Positive: {sample['pos'][:100]}...")
        print(f"  Negatives: {len(sample['negs'])} samples")


if __name__ == "__main__":
    main()
