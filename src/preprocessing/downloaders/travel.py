"""Travel domain data downloader for V27 training."""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from src.preprocessing.downloaders.base import BaseDownloader, RawSample

logger = logging.getLogger(__name__)


class TravelDownloader(BaseDownloader):
    """Downloader for travel/tourism domain data.

    Loads pre-collected travel data from data/v27.0/raw/ directory
    which contains data from Wikipedia, Namuwiki, Korpora, and templates.
    """

    dataset_name = "korean_travel_tourism"
    hf_path = ""  # Local data, not from HuggingFace
    expected_size = 110000  # ~110K samples expected

    def __init__(
        self,
        data_dir: str = "data/v27.0/raw",
        cache_dir: Optional[str] = None,
    ):
        """Initialize downloader.

        Args:
            data_dir: Directory containing raw travel data
            cache_dir: Not used, kept for interface compatibility
        """
        super().__init__(cache_dir)
        self.data_dir = Path(data_dir)
        self.data: List[Dict] = []
        self._loaded = False

    def download(self) -> None:
        """Load travel data from local files.

        Sources:
            - wikipedia/travel_articles.jsonl
            - namuwiki/travel_articles.jsonl
            - korpora/travel_content.jsonl
            - generated/travel_triplets.jsonl
        """
        if self._loaded:
            return

        self.data = []

        # Load Wikipedia articles
        wiki_file = self.data_dir / "wikipedia" / "travel_articles.jsonl"
        if wiki_file.exists():
            count = self._load_articles(wiki_file, "wikipedia")
            logger.info(f"Loaded {count:,} Wikipedia articles")

        # Load Namuwiki articles
        namu_file = self.data_dir / "namuwiki" / "travel_articles.jsonl"
        if namu_file.exists():
            count = self._load_articles(namu_file, "namuwiki")
            logger.info(f"Loaded {count:,} Namuwiki articles")

        # Load Korpora content
        korpora_file = self.data_dir / "korpora" / "travel_content.jsonl"
        if korpora_file.exists():
            count = self._load_articles(korpora_file, "korpora")
            logger.info(f"Loaded {count:,} Korpora samples")

        # Load template-generated triplets (already in triplet format)
        template_file = self.data_dir / "generated" / "travel_triplets.jsonl"
        if template_file.exists():
            count = self._load_triplets(template_file, "template")
            logger.info(f"Loaded {count:,} template triplets")

        self._loaded = True
        logger.info(f"Total travel samples: {len(self.data):,}")

    def _load_articles(self, path: Path, source: str) -> int:
        """Load article-format data (title, text, location, category)."""
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    article = json.loads(line.strip())
                    # Store as document for later triplet generation
                    self.data.append(
                        {
                            "type": "article",
                            "title": article.get("title", ""),
                            "text": article.get("text", ""),
                            "location": article.get("location"),
                            "category": article.get("category"),
                            "source": source,
                        }
                    )
                    count += 1
                except json.JSONDecodeError:
                    continue
        return count

    def _load_triplets(self, path: Path, source: str) -> int:
        """Load pre-formatted triplets."""
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    triplet = json.loads(line.strip())
                    self.data.append(
                        {
                            "type": "triplet",
                            "query": triplet.get("query", ""),
                            "positive": triplet.get("positive", ""),
                            "negative": triplet.get("negative"),
                            "pair_type": triplet.get("pair_type", "travel"),
                            "difficulty": triplet.get("difficulty", "medium"),
                            "source": source,
                        }
                    )
                    count += 1
                except json.JSONDecodeError:
                    continue
        return count

    def iterate(self) -> Iterator[RawSample]:
        """Iterate over raw samples.

        For articles: yields (query=title/location, text=content)
        For triplets: yields (text1=query, text2=positive)

        Yields:
            RawSample objects
        """
        if not self._loaded:
            self.download()

        for item in self.data:
            if item["type"] == "triplet":
                # Already formatted as query-positive pair
                yield RawSample(
                    text1=item["query"],
                    text2=item["positive"],
                    label="positive",
                    source=f"travel_{item['source']}",
                    metadata={
                        "pair_type": item.get("pair_type", "travel"),
                        "difficulty": item.get("difficulty", "medium"),
                        "negative": item.get("negative"),
                    },
                )
            else:
                # Article format - create query from title/location
                title = item.get("title", "")
                text = item.get("text", "")
                location = item.get("location", "")
                category = item.get("category", "")

                if not text or len(text) < 50:
                    continue

                # Generate query from available info
                if title:
                    query = title
                elif location:
                    query = f"{location} 관광"
                else:
                    continue

                yield RawSample(
                    text1=query,
                    text2=text[:1000],  # Truncate long texts
                    label="positive",
                    source=f"travel_{item['source']}",
                    metadata={
                        "title": title,
                        "location": location,
                        "category": category,
                        "pair_type": f"travel_{category}" if category else "travel",
                    },
                )

    def get_stats(self) -> Dict[str, int]:
        """Return dataset statistics."""
        if not self._loaded:
            self.download()

        stats = {
            "total": len(self.data),
            "articles": sum(1 for d in self.data if d["type"] == "article"),
            "triplets": sum(1 for d in self.data if d["type"] == "triplet"),
        }

        # Count by source
        sources = {}
        for item in self.data:
            src = item.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        stats["by_source"] = sources

        return stats


# For convenience, create aliases
TravelTourismDownloader = TravelDownloader
