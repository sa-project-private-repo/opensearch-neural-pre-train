"""모두의 말뭉치 parser using Korpora library."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm


class ModuCorpusParser:
    """Parser for 모두의 말뭉치 (Everyone's Corpus)."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize 모두의 말뭉치 parser.

        Args:
            cache_dir: Directory to cache downloaded corpus
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("dataset/modu/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_korpora_corpus(self, corpus_name: str) -> List[Dict]:
        """
        Load corpus using Korpora library.

        Args:
            corpus_name: Name of corpus (e.g., 'modu_news', 'modu_spoken', 'modu_web')

        Returns:
            List of text samples
        """
        try:
            from Korpora import Korpora

            print(f"Loading {corpus_name} from Korpora...")
            corpus = Korpora.load(corpus_name, root_dir=str(self.cache_dir))

            articles = []
            for item in corpus.train:
                text = ""
                if hasattr(item, "text"):
                    text = item.text
                elif hasattr(item, "sentence"):
                    text = item.sentence
                elif hasattr(item, "utterances"):
                    text = " ".join(item.utterances)

                if text:
                    articles.append(
                        {
                            "text": text,
                            "source": corpus_name,
                        }
                    )

            print(f"✓ Loaded {len(articles):,} samples from {corpus_name}")
            return articles

        except Exception as e:
            print(f"✗ Failed to load {corpus_name}: {e}")
            print(
                f"  Note: Some corpora require authentication. Please check Korpora documentation."
            )
            return []

    def filter_text(
        self,
        article: Dict,
        min_length: int = 50,
        max_length: int = 10000,
    ) -> bool:
        """
        Filter texts by quality criteria.

        Args:
            article: Article dict
            min_length: Minimum text length
            max_length: Maximum text length

        Returns:
            True if text passes filters
        """
        text = article.get("text", "")

        # Length check
        if len(text) < min_length or len(text) > max_length:
            return False

        # Skip very short or empty texts
        if not text.strip():
            return False

        return True

    def process_modu_corpus(
        self,
        output_path: str,
        corpus_types: Optional[List[str]] = None,
        max_articles: Optional[int] = None,
        min_length: int = 50,
        max_length: int = 100000,
        chunk_size: int = 50000,
    ) -> List[Dict]:
        """
        Process 모두의 말뭉치 and save cleaned texts.

        Args:
            output_path: Output file path (will be chunked)
            corpus_types: List of corpus types to load (None for default set)
            max_articles: Maximum number of articles to process
            min_length: Minimum text length
            max_length: Maximum text length
            chunk_size: Number of articles per chunk file

        Returns:
            List of processed articles
        """
        if corpus_types is None:
            # Default: publicly available corpora
            corpus_types = [
                "modu_news",  # 신문 말뭉치
                "modu_spoken",  # 구어 말뭉치
                "modu_web",  # 웹 말뭉치
                "modu_messenger",  # 메신저 말뭉치
            ]

        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        all_articles = []
        chunk_num = 0
        current_chunk = []

        print("=" * 80)
        print("Processing 모두의 말뭉치 (Everyone's Corpus)")
        print("=" * 80)

        for corpus_name in corpus_types:
            print(f"\n>>> Loading {corpus_name}...")
            corpus_articles = self.load_korpora_corpus(corpus_name)

            for raw_article in tqdm(
                corpus_articles, desc=f"Processing {corpus_name}"
            ):
                article = {
                    "id": f"modu_{len(all_articles)}",
                    "title": f"{corpus_name}_{len(all_articles)}",
                    "text": raw_article["text"],
                    "language": "ko",
                    "source": raw_article["source"],
                }

                if self.filter_text(article, min_length, max_length):
                    current_chunk.append(article)
                    all_articles.append(article)

                    # Save chunk when it reaches the limit
                    if len(current_chunk) >= chunk_size:
                        chunk_num += 1
                        chunk_file = output_dir / f"modu_chunk_{chunk_num:03d}.jsonl"
                        self.save_articles(current_chunk, chunk_file)
                        current_chunk = []

                if max_articles and len(all_articles) >= max_articles:
                    break

            if max_articles and len(all_articles) >= max_articles:
                break

        # Save remaining articles in last chunk
        if current_chunk:
            chunk_num += 1
            chunk_file = output_dir / f"modu_chunk_{chunk_num:03d}.jsonl"
            self.save_articles(current_chunk, chunk_file)

        print(f"\n✓ Processed {len(all_articles):,} 모두의 말뭉치 texts")
        print(f"✓ Saved in {chunk_num} chunk files")

        return all_articles

    def save_articles(
        self,
        articles: List[Dict],
        output_path: Path,
    ) -> None:
        """
        Save articles to JSONL file.

        Args:
            articles: List of article dicts
            output_path: Output file path
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + "\n")

        print(f"Saved {len(articles)} texts to {output_path}")


if __name__ == "__main__":
    # Example usage
    parser = ModuCorpusParser(cache_dir="dataset/modu/cache")

    # Process 모두의 말뭉치 (sample)
    articles = parser.process_modu_corpus(
        output_path="dataset/modu/modu_corpus.jsonl",
        max_articles=1000,
    )

    print(f"Processed {len(articles)} texts")
