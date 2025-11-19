"""NamuWiki parser for extracting Korean text from HuggingFace dataset."""

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from datasets import load_dataset
from tqdm import tqdm


class NamuWikiParser:
    """Parser for NamuWiki dataset from HuggingFace."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize NamuWiki parser.

        Args:
            cache_dir: Directory to cache downloaded dataset
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def load_dataset(self, split: str = "train") -> Iterator[Dict]:
        """
        Load NamuWiki dataset from HuggingFace.

        Args:
            split: Dataset split (train, validation, test)

        Yields:
            Article dicts with title and text
        """
        print("Loading NamuWiki dataset from HuggingFace...")
        print("This may take a while for the first download...")

        # Load preprocessed NamuWiki dataset
        dataset = load_dataset(
            "heegyu/namuwiki-extracted",
            split=split,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )

        print(f"Loaded {len(dataset):,} articles")

        for item in dataset:
            yield {
                "title": item.get("title", ""),
                "text": item.get("text", ""),
            }

    def filter_article(
        self,
        article: Dict,
        min_length: int = 100,
        max_length: int = 10000,
    ) -> bool:
        """
        Filter articles by quality criteria.

        Args:
            article: Article dict
            min_length: Minimum text length
            max_length: Maximum text length

        Returns:
            True if article passes filters
        """
        text = article.get("text", "")
        title = article.get("title", "")

        # Length check
        if len(text) < min_length or len(text) > max_length:
            return False

        # Skip special pages
        if title.startswith(("틀:", "분류:", "파일:", "나무위키:")):
            return False

        # Skip very short titles (likely noise)
        if len(title) < 2:
            return False

        return True

    def process_namuwiki(
        self,
        output_path: str,
        max_articles: Optional[int] = None,
        min_length: int = 100,
        max_length: int = 100000,
        chunk_size: int = 50000,
    ) -> List[Dict]:
        """
        Process NamuWiki dataset and save cleaned articles.

        Args:
            output_path: Output file path (will be chunked)
            max_articles: Maximum number of articles to process
            min_length: Minimum article length
            max_length: Maximum article length
            chunk_size: Number of articles per chunk file

        Returns:
            List of processed articles
        """
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        articles = []
        chunk_num = 0
        current_chunk = []

        print("Processing NamuWiki articles...")

        iterator = self.load_dataset()
        if max_articles:
            iterator = tqdm(iterator, total=max_articles, desc="Processing NamuWiki")
        else:
            iterator = tqdm(iterator, desc="Processing NamuWiki")

        for raw_article in iterator:
            article = {
                "id": f"namu_{len(articles)}",
                "url": f"https://namu.wiki/w/{raw_article['title']}",
                "title": raw_article["title"],
                "text": raw_article["text"],
                "language": "ko",
                "source": "namuwiki",
            }

            if self.filter_article(article, min_length, max_length):
                current_chunk.append(article)
                articles.append(article)

                # Save chunk when it reaches the limit
                if len(current_chunk) >= chunk_size:
                    chunk_num += 1
                    chunk_file = (
                        output_dir / f"namuwiki_chunk_{chunk_num:03d}.jsonl"
                    )
                    self.save_articles(current_chunk, chunk_file)
                    current_chunk = []

            if max_articles and len(articles) >= max_articles:
                break

        # Save remaining articles in last chunk
        if current_chunk:
            chunk_num += 1
            chunk_file = output_dir / f"namuwiki_chunk_{chunk_num:03d}.jsonl"
            self.save_articles(current_chunk, chunk_file)

        print(f"\n✓ Processed {len(articles):,} NamuWiki articles")
        print(f"✓ Saved in {chunk_num} chunk files")

        return articles

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

        print(f"Saved {len(articles)} articles to {output_path}")


if __name__ == "__main__":
    # Example usage
    parser = NamuWikiParser(cache_dir="dataset/namuwiki/cache")

    # Process NamuWiki (sample)
    articles = parser.process_namuwiki(
        output_path="dataset/namuwiki/namuwiki_articles.jsonl",
        max_articles=1000,
    )

    print(f"Processed {len(articles)} NamuWiki articles")
    if articles:
        print(f"Sample: {articles[0]['title']}")
