"""Wikipedia data parser for extracting clean text and inter-language links."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm


class WikipediaParser:
    """Parser for Wikipedia dumps using HuggingFace datasets."""

    def __init__(
        self,
        language: str = "ko",
        cache_dir: Optional[str] = None,
        date: str = "20220301",
    ):
        """
        Initialize Wikipedia parser.

        Args:
            language: Language code (ko, en, etc.)
            cache_dir: Directory to cache downloaded data
            date: Wikipedia dump date (format: YYYYMMDD, available: 20220301)
        """
        self.language = language
        self.cache_dir = cache_dir
        self.date = date

    def load_wikipedia(
        self,
        split: str = "train",
        streaming: bool = False,
    ) -> "Dataset":
        """
        Load Wikipedia dataset from HuggingFace.

        Args:
            split: Dataset split (train)
            streaming: Whether to use streaming mode

        Returns:
            Wikipedia dataset
        """
        print(f"Loading {self.language} Wikipedia ({self.date})...")

        dataset = load_dataset(
            "wikipedia",
            f"{self.date}.{self.language}",
            split=split,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            streaming=streaming,
        )

        return dataset

    def clean_text(self, text: str) -> str:
        """
        Clean Wikipedia article text.

        Args:
            text: Raw article text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove citations like [1], [2], etc.
        text = re.sub(r"\[\d+\]", "", text)

        # Remove special markup
        text = re.sub(r"\{\{[^}]+\}\}", "", text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Normalize quotation marks
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")

        return text.strip()

    def extract_article(self, article: Dict) -> Dict:
        """
        Extract and clean article data.

        Args:
            article: Raw article dict from dataset

        Returns:
            Cleaned article dict
        """
        return {
            "id": article.get("id", ""),
            "url": article.get("url", ""),
            "title": article.get("title", ""),
            "text": self.clean_text(article.get("text", "")),
            "language": self.language,
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

        # Length check
        if len(text) < min_length or len(text) > max_length:
            return False

        # Title check (avoid disambiguation pages)
        title = article.get("title", "")
        if any(
            keyword in title.lower()
            for keyword in ["(disambiguation)", "목록", "list of"]
        ):
            return False

        return True

    def save_articles(
        self,
        articles: List[Dict],
        output_path: str,
        format: str = "jsonl",
    ) -> None:
        """
        Save articles to file.

        Args:
            articles: List of article dicts
            output_path: Output file path
            format: Output format (jsonl, json)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for article in articles:
                    f.write(json.dumps(article, ensure_ascii=False) + "\n")
        elif format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(articles)} articles to {output_path}")

    def load_articles(self, input_path: str) -> List[Dict]:
        """
        Load articles from JSONL file.

        Args:
            input_path: Input file path

        Returns:
            List of article dicts
        """
        articles = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                articles.append(json.loads(line))
        return articles

    def process_wikipedia(
        self,
        output_path: str,
        max_articles: Optional[int] = None,
        min_length: int = 100,
        max_length: int = 10000,
        streaming: bool = True,
    ) -> List[Dict]:
        """
        Process Wikipedia dump and save cleaned articles.

        Args:
            output_path: Output file path
            max_articles: Maximum number of articles to process
            min_length: Minimum article length
            max_length: Maximum article length
            streaming: Whether to use streaming mode

        Returns:
            List of processed articles
        """
        dataset = self.load_wikipedia(streaming=streaming)

        articles = []
        processed = 0

        print(f"Processing Wikipedia articles...")

        iterator = iter(dataset)
        if max_articles:
            iterator = tqdm(iterator, total=max_articles, desc="Processing")
        else:
            iterator = tqdm(iterator, desc="Processing")

        for raw_article in iterator:
            article = self.extract_article(raw_article)

            if self.filter_article(article, min_length, max_length):
                articles.append(article)

            processed += 1
            if max_articles and processed >= max_articles:
                break

        # Save articles
        self.save_articles(articles, output_path)

        return articles


class InterlanguageLinker:
    """Extract inter-language links from Wikipedia articles."""

    def __init__(self):
        """Initialize inter-language linker."""
        self.title_mapping: Dict[str, Dict[str, str]] = {}

    def load_wikipedia_titles(
        self,
        language: str,
        max_articles: int = 10000,
    ) -> Dict[str, str]:
        """
        Load Wikipedia article titles for a language.

        Args:
            language: Language code
            max_articles: Maximum articles to load

        Returns:
            Dict mapping article ID to title
        """
        print(f"Loading {language} Wikipedia titles...")

        dataset = load_dataset(
            "wikipedia",
            f"20220301.{language}",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        titles = {}
        for i, article in enumerate(dataset):
            if i >= max_articles:
                break
            titles[article["id"]] = article["title"]

        return titles

    def extract_interlang_links(
        self,
        ko_articles_path: str,
        en_articles_path: str,
        output_path: str,
    ) -> List[Tuple[str, str]]:
        """
        Extract Korean-English title pairs from Wikipedia.

        Args:
            ko_articles_path: Korean articles JSONL path
            en_articles_path: English articles JSONL path
            output_path: Output path for synonym pairs

        Returns:
            List of (korean_title, english_title) tuples
        """
        # Load article titles
        parser_ko = WikipediaParser(language="ko")
        parser_en = WikipediaParser(language="en")

        ko_articles = parser_ko.load_articles(ko_articles_path)
        en_articles = parser_en.load_articles(en_articles_path)

        # Create title mappings
        ko_titles = {a["id"]: a["title"] for a in ko_articles}
        en_titles = {a["id"]: a["title"] for a in en_articles}

        # For now, we'll use URL-based matching
        # Wikipedia URLs contain language-agnostic article identifiers
        synonym_pairs = []

        # Extract URL identifiers and match
        ko_url_to_title = {}
        for article in ko_articles:
            url = article.get("url", "")
            if url:
                # Extract article identifier from URL
                identifier = url.split("/")[-1]
                ko_url_to_title[identifier] = article["title"]

        for article in en_articles:
            url = article.get("url", "")
            if url:
                identifier = url.split("/")[-1]
                if identifier in ko_url_to_title:
                    ko_title = ko_url_to_title[identifier]
                    en_title = article["title"]
                    synonym_pairs.append((ko_title, en_title))

        # Save synonym pairs
        output_data = [
            {
                "korean": ko_title,
                "english": en_title,
                "source": "wikipedia_interlang",
            }
            for ko_title, en_title in synonym_pairs
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"Extracted {len(synonym_pairs)} inter-language links")
        return synonym_pairs


if __name__ == "__main__":
    # Example usage
    parser = WikipediaParser(language="ko")

    # Process Korean Wikipedia (sample)
    articles = parser.process_wikipedia(
        output_path="dataset/wikipedia/ko_articles.jsonl",
        max_articles=1000,
        streaming=True,
    )

    print(f"Processed {len(articles)} Korean articles")
