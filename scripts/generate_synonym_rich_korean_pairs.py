"""Generate Korean same-language training pairs with synonym-rich documents.

This script creates training data in the correct format for SPLADE:
- Korean query → Korean document (with synonyms)
- Hard negatives from topically similar documents

Follows the GitHub sample format:
{
    "query": "xxx xxx xxx",
    "pos": "xxxx xxxx xxxx",
    "negs": ["xxx", "xxx", "xxx"]
}
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from tqdm import tqdm


class SynonymRichPairGenerator:
    """Generate training pairs from synonym-rich Korean Wikipedia articles."""

    def __init__(
        self,
        synonyms_path: str,
        wiki_chunks_dir: str,
        output_path: str,
        num_pairs: int = 10000,
        num_negatives: int = 3,
    ):
        """
        Initialize generator.

        Args:
            synonyms_path: Path to combined synonyms JSON
            wiki_chunks_dir: Directory containing Wikipedia chunk files
            output_path: Output JSONL file path
            num_pairs: Number of training pairs to generate
            num_negatives: Number of hard negatives per query
        """
        self.synonyms_path = synonyms_path
        self.wiki_chunks_dir = Path(wiki_chunks_dir)
        self.output_path = output_path
        self.num_pairs = num_pairs
        self.num_negatives = num_negatives

        # Data structures
        self.synonyms = []
        self.articles = []
        self.term_to_articles = defaultdict(list)  # term -> article indices

    def load_synonyms(self):
        """Load synonym pairs."""
        print("Loading synonyms...")
        with open(self.synonyms_path, 'r', encoding='utf-8') as f:
            self.synonyms = json.load(f)

        # Filter valid synonyms (both fields non-empty)
        self.synonyms = [
            syn for syn in self.synonyms
            if syn.get('korean') and syn.get('english')
        ]

        print(f"✓ Loaded {len(self.synonyms):,} synonym pairs")

    def load_wikipedia_articles(self, max_chunks: int = 10):
        """Load Wikipedia articles from chunks."""
        print(f"Loading Wikipedia articles from {max_chunks} chunks...")

        # Get Korean chunk files
        chunk_files = sorted(self.wiki_chunks_dir.glob("kowiki_*.jsonl"))[:max_chunks]

        for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
            with open(chunk_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        article = json.loads(line)
                        if article.get('text') and len(article['text']) > 100:
                            self.articles.append(article)
                    except json.JSONDecodeError:
                        continue

        print(f"✓ Loaded {len(self.articles):,} articles")

    def build_term_index(self):
        """Build index from terms to articles containing them."""
        print("Building term-to-article index...")

        # Extract key terms from synonyms
        key_terms = set()
        for syn in self.synonyms:
            if syn.get('korean'):
                key_terms.add(syn['korean'])
            if syn.get('english'):
                key_terms.add(syn['english'])

        # Index articles by terms
        for idx, article in enumerate(tqdm(self.articles, desc="Indexing")):
            text = article.get('text', '').lower()
            title = article.get('title', '').lower()

            for term in key_terms:
                term_lower = term.lower()
                if term_lower in text or term_lower in title:
                    self.term_to_articles[term].append(idx)

        print(f"✓ Indexed {len(self.term_to_articles):,} terms")

    def find_synonym_rich_articles(self, min_synonyms: int = 2) -> List[Tuple[int, Set[str]]]:
        """
        Find articles containing multiple synonym terms.

        Args:
            min_synonyms: Minimum number of synonym terms in article

        Returns:
            List of (article_idx, set of synonym terms)
        """
        print("Finding synonym-rich articles...")

        synonym_rich = []

        for idx, article in enumerate(tqdm(self.articles, desc="Scanning")):
            text = article.get('text', '').lower()
            title = article.get('title', '').lower()

            # Find all synonym terms in this article
            found_terms = set()
            for syn in self.synonyms:
                korean = syn.get('korean', '')
                english = syn.get('english', '')

                if korean and (korean.lower() in text or korean.lower() in title):
                    found_terms.add(korean)
                if english and (english.lower() in text or english.lower() in title):
                    found_terms.add(english)

            if len(found_terms) >= min_synonyms:
                synonym_rich.append((idx, found_terms))

        print(f"✓ Found {len(synonym_rich):,} synonym-rich articles")
        return synonym_rich

    def generate_query_from_title(self, title: str) -> List[str]:
        """Generate query variations from title."""
        queries = [
            title,  # Direct title
            f"{title}이란 무엇인가",  # What is X?
            f"{title}에 대해",  # About X
        ]
        return queries

    def sample_hard_negatives(
        self,
        positive_idx: int,
        query_terms: Set[str],
        num_negatives: int
    ) -> List[str]:
        """
        Sample hard negatives from topically related articles.

        Args:
            positive_idx: Index of positive article
            query_terms: Terms in the query
            num_negatives: Number of negatives to sample

        Returns:
            List of negative documents
        """
        candidates = set()

        # Find articles containing some query terms (but not all)
        for term in query_terms:
            if term in self.term_to_articles:
                candidates.update(self.term_to_articles[term])

        # Remove positive article
        candidates.discard(positive_idx)

        # Sample negatives
        if len(candidates) >= num_negatives:
            neg_indices = random.sample(list(candidates), num_negatives)
        else:
            # Fallback to random sampling
            all_indices = list(range(len(self.articles)))
            all_indices.remove(positive_idx)
            neg_indices = random.sample(all_indices, min(num_negatives, len(all_indices)))

        # Get negative documents
        negatives = []
        for idx in neg_indices:
            article = self.articles[idx]
            doc_text = self._format_document(article)
            negatives.append(doc_text)

        return negatives

    def _format_document(self, article: Dict) -> str:
        """Format article as document text."""
        title = article.get('title', '')
        text = article.get('text', '')

        # Truncate text to reasonable length
        max_length = 500
        if len(text) > max_length:
            text = text[:max_length]

        return f"{title} {text}".strip()

    def generate_pairs(self):
        """Generate training pairs."""
        print(f"Generating {self.num_pairs:,} training pairs...")

        # Find synonym-rich articles
        synonym_rich = self.find_synonym_rich_articles(min_synonyms=2)

        if len(synonym_rich) == 0:
            raise ValueError("No synonym-rich articles found")

        pairs = []

        for _ in tqdm(range(self.num_pairs), desc="Generating pairs"):
            # Sample a synonym-rich article
            article_idx, synonym_terms = random.choice(synonym_rich)
            article = self.articles[article_idx]

            # Generate query
            title = article.get('title', '')
            if not title:
                continue

            query_variations = self.generate_query_from_title(title)
            query = random.choice(query_variations)

            # Positive document
            pos_doc = self._format_document(article)

            # Hard negatives
            negatives = self.sample_hard_negatives(
                article_idx,
                synonym_terms,
                self.num_negatives
            )

            # Create training pair
            pair = {
                "query": query,
                "pos": pos_doc,
                "negs": negatives,
                "metadata": {
                    "synonym_terms": list(synonym_terms),
                    "article_id": article.get('id', ''),
                }
            }

            pairs.append(pair)

        return pairs

    def save_pairs(self, pairs: List[Dict]):
        """Save training pairs to JSONL."""
        print(f"Saving {len(pairs):,} pairs to {self.output_path}...")

        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')

        print(f"✓ Saved to {self.output_path}")

    def run(self):
        """Run the generation pipeline."""
        self.load_synonyms()
        self.load_wikipedia_articles(max_chunks=10)
        self.build_term_index()
        pairs = self.generate_pairs()
        self.save_pairs(pairs)

        print("\n=== Generation Complete ===")
        print(f"Total pairs: {len(pairs):,}")
        print(f"Negatives per query: {self.num_negatives}")
        print(f"Output: {self.output_path}")


def main():
    """Main entry point."""
    generator = SynonymRichPairGenerator(
        synonyms_path="dataset/synonyms/combined_synonyms_v2.json",
        wiki_chunks_dir="dataset/korean_wikipedia",
        output_path="dataset/baseline_samples/train_synonym_rich_korean.jsonl",
        num_pairs=10000,
        num_negatives=3,
    )

    generator.run()


if __name__ == "__main__":
    main()
