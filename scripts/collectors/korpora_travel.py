"""Korpora Korean corpus collector for travel/tourism content."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class KorporaArticle:
    """Article from Korpora corpus."""

    text: str
    category: str
    location: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class KorporaTravelCollector:
    """Collect travel/tourism content from Korpora.

    Korpora provides various Korean corpora that can be installed via pip.
    This collector filters for travel-related content.
    """

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

    TRAVEL_KEYWORDS = [
        "관광",
        "여행",
        "명소",
        "숙박",
        "호텔",
        "펜션",
        "리조트",
        "맛집",
        "음식점",
        "식당",
        "카페",
        "해변",
        "해수욕장",
        "산",
        "계곡",
        "공원",
        "박물관",
        "미술관",
        "궁",
        "사찰",
        "축제",
        "관람",
        "투어",
        "코스",
    ]

    # Korpora datasets to use
    DATASETS = [
        "korean_parallel_koen_news",  # News corpus
        "namuwikitext",  # Namuwiki text
        "kowikitext",  # Korean Wikipedia text
    ]

    def __init__(
        self,
        output_dir: str = "data/v27.0/raw/korpora",
        max_samples: int = 20000,
    ):
        """Initialize collector.

        Args:
            output_dir: Directory to save collected data
            max_samples: Maximum samples to collect
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self._korpora_available = self._check_korpora()

    def _check_korpora(self) -> bool:
        """Check if Korpora is installed."""
        try:
            import Korpora

            return True
        except ImportError:
            logger.warning(
                "Korpora not installed. Install with: pip install Korpora"
            )
            return False

    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location from text."""
        for location in self.LOCATIONS:
            if location in text:
                return location
        return None

    def _is_travel_related(self, text: str) -> bool:
        """Check if text is travel-related."""
        text_lower = text.lower()
        for keyword in self.TRAVEL_KEYWORDS:
            if keyword in text_lower:
                return True
        return False

    def _categorize_text(self, text: str) -> str:
        """Categorize text content."""
        if any(k in text for k in ["호텔", "펜션", "리조트", "숙박"]):
            return "accommodation"
        if any(k in text for k in ["맛집", "음식점", "식당", "카페"]):
            return "restaurant"
        if any(k in text for k in ["해변", "해수욕장", "바다"]):
            return "beach"
        if any(k in text for k in ["산", "계곡", "공원"]):
            return "nature"
        if any(k in text for k in ["박물관", "미술관", "전시"]):
            return "museum"
        if any(k in text for k in ["궁", "사찰", "문화재"]):
            return "cultural_heritage"
        if any(k in text for k in ["축제", "행사"]):
            return "festival"
        return "general"

    def _load_kowikitext(self) -> Iterator[str]:
        """Load Korean Wikipedia text from Korpora."""
        try:
            from Korpora import Korpora

            corpus = Korpora.load("kowikitext")
            for text in corpus.train:
                yield text
        except Exception as e:
            logger.warning(f"Failed to load kowikitext: {e}")

    def _load_namuwikitext(self) -> Iterator[str]:
        """Load Namuwiki text from Korpora."""
        try:
            from Korpora import Korpora

            corpus = Korpora.load("namuwikitext")
            for text in corpus.train:
                yield text
        except Exception as e:
            logger.warning(f"Failed to load namuwikitext: {e}")

    def iterate(self) -> Iterator[KorporaArticle]:
        """Iterate over travel-related content.

        Yields:
            KorporaArticle objects
        """
        if not self._korpora_available:
            logger.warning("Korpora not available. Skipping.")
            return

        count = 0

        # Try loading available corpora
        for corpus_loader in [self._load_kowikitext, self._load_namuwikitext]:
            if count >= self.max_samples:
                break

            try:
                for text in corpus_loader():
                    if count >= self.max_samples:
                        break

                    if not isinstance(text, str) or len(text) < 50:
                        continue

                    if not self._is_travel_related(text):
                        continue

                    location = self._extract_location(text)
                    if location is None:
                        continue

                    category = self._categorize_text(text)

                    count += 1
                    yield KorporaArticle(
                        text=text[:2000],  # Truncate long texts
                        category=category,
                        location=location,
                        metadata={"source": "korpora"},
                    )

            except Exception as e:
                logger.warning(f"Error loading corpus: {e}")
                continue

        logger.info(f"Collected {count:,} travel samples from Korpora")

    def collect(self) -> List[KorporaArticle]:
        """Collect all travel content.

        Returns:
            List of KorporaArticle objects
        """
        return list(self.iterate())

    def save(self) -> Optional[Path]:
        """Save collected content to JSONL file.

        Returns:
            Path to output file or None if not available
        """
        if not self._korpora_available:
            logger.warning("Korpora not available. Skipping.")
            return None

        output_file = self.output_dir / "travel_content.jsonl"
        articles = []

        for article in self.iterate():
            articles.append(
                {
                    "text": article.text,
                    "category": article.category,
                    "location": article.location,
                    "metadata": article.metadata,
                }
            )

        if not articles:
            logger.warning("No travel content found in Korpora")
            return None

        with open(output_file, "w", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(articles):,} samples to {output_file}")
        return output_file

    def get_stats(self, articles: Optional[List[Dict]] = None) -> Dict:
        """Get collection statistics."""
        if articles is None:
            articles = [
                {
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
    collector = KorporaTravelCollector()
    output_path = collector.save()
    if output_path:
        print(f"Saved to: {output_path}")
