"""Korean Wikipedia travel/tourism article collector for V27."""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class TravelArticle:
    """Travel-related Wikipedia article."""

    title: str
    text: str
    category: str
    location: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class WikipediaTravelCollector:
    """Collect travel/tourism articles from Korean Wikipedia.

    Uses HuggingFace datasets to load Korean Wikipedia and filters
    for travel-related content based on keywords and patterns.
    """

    # Korean administrative regions (17 regions)
    LOCATIONS = [
        "서울",
        "부산",
        "인천",
        "대구",
        "광주",
        "대전",
        "울산",
        "세종",
        "경기",
        "강원",
        "충북",
        "충남",
        "경북",
        "경남",
        "전북",
        "전남",
        "제주",
    ]

    # Tourism-related keywords
    TRAVEL_KEYWORDS = [
        # Tourist attractions
        "관광",
        "관광지",
        "명소",
        "여행",
        "여행지",
        # Landmarks
        "궁",
        "궁궐",
        "성",
        "성곽",
        "사찰",
        "절",
        "탑",
        "문화재",
        # Natural attractions
        "국립공원",
        "산",
        "계곡",
        "폭포",
        "해수욕장",
        "해변",
        "섬",
        "호수",
        # Cultural facilities
        "박물관",
        "미술관",
        "기념관",
        "전시관",
        "체험관",
        # Accommodation & dining
        "호텔",
        "리조트",
        "펜션",
        "민박",
        "맛집",
        "음식점",
        # Events & activities
        "축제",
        "행사",
        "체험",
        "투어",
        "코스",
    ]

    # Negative keywords to filter out irrelevant articles
    EXCLUDE_KEYWORDS = [
        "선거",
        "정치",
        "군사",
        "전쟁",
        "재판",
        "범죄",
        "사고",
        "사건",
    ]

    def __init__(
        self,
        output_dir: str = "data/v27.0/raw/wikipedia",
        cache_dir: Optional[str] = None,
        max_articles: int = 50000,
    ):
        """Initialize collector.

        Args:
            output_dir: Directory to save collected articles
            cache_dir: HuggingFace cache directory
            max_articles: Maximum number of articles to collect
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir
        self.max_articles = max_articles
        self.dataset = None

    def download(self) -> None:
        """Download Korean Wikipedia from HuggingFace.

        Uses wikimedia/wikipedia dataset with Korean (ko) subset.
        """
        logger.info("Downloading Korean Wikipedia from HuggingFace...")
        logger.info("This may take a few minutes on first run...")

        try:
            self.dataset = load_dataset(
                "wikimedia/wikipedia",
                "20231101.ko",
                split="train",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            logger.info(f"Downloaded {len(self.dataset):,} articles")
        except Exception as e:
            logger.error(f"Failed to load Wikipedia: {e}")
            raise RuntimeError(
                f"Could not load Korean Wikipedia: {e}. "
                "Please check your internet connection."
            )

    def _extract_location(self, text: str, title: str) -> Optional[str]:
        """Extract location from article text or title."""
        combined = f"{title} {text[:500]}"
        for location in self.LOCATIONS:
            if location in combined:
                return location
        return None

    def _is_travel_related(self, text: str, title: str) -> bool:
        """Check if article is travel-related."""
        combined = f"{title} {text[:1000]}".lower()

        # Check for exclude keywords
        for keyword in self.EXCLUDE_KEYWORDS:
            if keyword in combined:
                return False

        # Check for travel keywords
        for keyword in self.TRAVEL_KEYWORDS:
            if keyword in combined:
                return True

        # Check if title contains location + common suffix
        travel_suffixes = ["역", "공원", "산", "해변", "절", "궁", "성", "관", "원"]
        for suffix in travel_suffixes:
            if title.endswith(suffix):
                return True

        return False

    def _categorize_article(self, text: str, title: str) -> str:
        """Determine article category."""
        combined = f"{title} {text[:500]}"

        if any(k in combined for k in ["궁", "궁궐", "성", "성곽"]):
            return "palace_fortress"
        if any(k in combined for k in ["사찰", "절", "탑"]):
            return "temple"
        if any(k in combined for k in ["박물관", "미술관", "기념관"]):
            return "museum"
        if any(k in combined for k in ["국립공원", "산", "계곡", "폭포"]):
            return "nature"
        if any(k in combined for k in ["해수욕장", "해변", "섬"]):
            return "beach"
        if any(k in combined for k in ["호텔", "리조트", "펜션"]):
            return "accommodation"
        if any(k in combined for k in ["축제", "행사"]):
            return "festival"
        if any(k in combined for k in ["맛집", "음식점", "식당"]):
            return "restaurant"

        return "general_tourism"

    def _clean_text(self, text: str) -> str:
        """Clean Wikipedia article text."""
        # Remove wiki markup
        text = re.sub(r"\[\[.*?\|", "", text)
        text = re.sub(r"\[\[|\]\]", "", text)
        text = re.sub(r"\{\{.*?\}\}", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"==+.*?==+", "", text)
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def iterate(self) -> Iterator[TravelArticle]:
        """Iterate over travel-related articles.

        Yields:
            TravelArticle objects
        """
        if self.dataset is None:
            self.download()

        count = 0
        for article in self.dataset:
            if count >= self.max_articles:
                break

            title = article.get("title", "")
            text = article.get("text", "")

            if not self._is_travel_related(text, title):
                continue

            location = self._extract_location(text, title)
            if location is None:
                continue

            category = self._categorize_article(text, title)
            cleaned_text = self._clean_text(text)

            if len(cleaned_text) < 100:
                continue

            count += 1
            yield TravelArticle(
                title=title,
                text=cleaned_text,
                category=category,
                location=location,
                metadata={"source": "wikipedia", "article_id": article.get("id")},
            )

        logger.info(f"Collected {count:,} travel articles from Wikipedia")

    def collect(self) -> List[TravelArticle]:
        """Collect all travel articles.

        Returns:
            List of TravelArticle objects
        """
        return list(self.iterate())

    def save(self) -> Path:
        """Save collected articles to JSONL file.

        Returns:
            Path to output file
        """
        output_file = self.output_dir / "travel_articles.jsonl"
        articles = []

        for article in self.iterate():
            articles.append(
                {
                    "title": article.title,
                    "text": article.text,
                    "category": article.category,
                    "location": article.location,
                    "metadata": article.metadata,
                }
            )

        with open(output_file, "w", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(articles):,} articles to {output_file}")

        # Save statistics
        stats = self.get_stats(articles)
        stats_file = self.output_dir / "stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        return output_file

    def get_stats(self, articles: Optional[List[Dict]] = None) -> Dict:
        """Get collection statistics.

        Args:
            articles: Optional list of articles (collects if None)

        Returns:
            Statistics dictionary
        """
        if articles is None:
            articles = [
                {
                    "title": a.title,
                    "text": a.text,
                    "category": a.category,
                    "location": a.location,
                }
                for a in self.iterate()
            ]

        stats = {
            "total": len(articles),
            "by_location": {},
            "by_category": {},
        }

        for article in articles:
            loc = article.get("location", "unknown")
            cat = article.get("category", "unknown")
            stats["by_location"][loc] = stats["by_location"].get(loc, 0) + 1
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

        return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = WikipediaTravelCollector()
    output_path = collector.save()
    print(f"Saved to: {output_path}")
