"""Namuwiki dump parser for travel/tourism articles."""

import bz2
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NamuwikiArticle:
    """Namuwiki article data."""

    title: str
    text: str
    category: str
    location: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class NamuwikiDumpParser:
    """Parse Namuwiki dump files for travel/tourism content.

    Namuwiki dump can be downloaded from: https://mu-star.net/wikidb
    The dump is in JSON format, optionally compressed with bz2.
    """

    # Same location and keyword lists as Wikipedia collector
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
        "관광지",
        "명소",
        "여행",
        "여행지",
        "궁",
        "궁궐",
        "성",
        "성곽",
        "사찰",
        "절",
        "탑",
        "문화재",
        "국립공원",
        "산",
        "계곡",
        "폭포",
        "해수욕장",
        "해변",
        "섬",
        "호수",
        "박물관",
        "미술관",
        "기념관",
        "전시관",
        "체험관",
        "호텔",
        "리조트",
        "펜션",
        "민박",
        "맛집",
        "음식점",
        "축제",
        "행사",
        "체험",
        "투어",
        "코스",
    ]

    # Namuwiki-specific category patterns
    TRAVEL_CATEGORIES = [
        "분류:대한민국의 관광지",
        "분류:서울특별시",
        "분류:부산광역시",
        "분류:제주특별자치도",
        "분류:관광",
        "분류:여행",
        "분류:맛집",
        "분류:음식점",
        "분류:한국의 산",
        "분류:국립공원",
    ]

    def __init__(
        self,
        dump_path: Optional[str] = None,
        output_dir: str = "data/v27.0/raw/namuwiki",
        max_articles: int = 30000,
    ):
        """Initialize parser.

        Args:
            dump_path: Path to Namuwiki dump file (JSON or bz2)
            output_dir: Directory to save parsed articles
            max_articles: Maximum number of articles to extract
        """
        self.dump_path = Path(dump_path) if dump_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_articles = max_articles

    def _open_dump(self, path: Path):
        """Open dump file, handling compression."""
        if path.suffix == ".bz2":
            return bz2.open(path, "rt", encoding="utf-8")
        return open(path, "r", encoding="utf-8")

    def _clean_namuwiki_text(self, text: str) -> str:
        """Clean Namuwiki markup."""
        # Remove Namuwiki-specific markup
        text = re.sub(r"\[\[파일:.*?\]\]", "", text)
        text = re.sub(r"\[\[분류:.*?\]\]", "", text)
        text = re.sub(r"\[\[.*?\|", "", text)
        text = re.sub(r"\[\[|\]\]", "", text)
        text = re.sub(r"\{\{\{.*?\}\}\}", "", text)
        text = re.sub(r"'''|''", "", text)
        text = re.sub(r"--.*?--", "", text)
        text = re.sub(r"~~.*?~~", "", text)
        text = re.sub(r"=+.*?=+", "", text)
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_location(self, text: str, title: str) -> Optional[str]:
        """Extract location from article."""
        combined = f"{title} {text[:500]}"
        for location in self.LOCATIONS:
            if location in combined:
                return location
        return None

    def _is_travel_related(self, text: str, title: str) -> bool:
        """Check if article is travel-related."""
        combined = f"{title} {text[:1000]}"

        for keyword in self.TRAVEL_KEYWORDS:
            if keyword in combined:
                return True

        # Check category patterns
        for category in self.TRAVEL_CATEGORIES:
            if category in text:
                return True

        return False

    def _categorize_article(self, text: str, title: str) -> str:
        """Determine article category."""
        combined = f"{title} {text[:500]}"

        if any(k in combined for k in ["궁", "궁궐"]):
            return "palace"
        if any(k in combined for k in ["사찰", "절"]):
            return "temple"
        if any(k in combined for k in ["박물관", "미술관"]):
            return "museum"
        if any(k in combined for k in ["국립공원", "산"]):
            return "nature"
        if any(k in combined for k in ["해수욕장", "해변"]):
            return "beach"
        if any(k in combined for k in ["맛집", "음식점"]):
            return "restaurant"
        if any(k in combined for k in ["호텔", "리조트"]):
            return "accommodation"
        if any(k in combined for k in ["축제"]):
            return "festival"

        return "general"

    def iterate(self) -> Iterator[NamuwikiArticle]:
        """Iterate over travel-related articles from dump.

        Yields:
            NamuwikiArticle objects
        """
        if self.dump_path is None or not self.dump_path.exists():
            logger.warning(
                f"Namuwiki dump not found at {self.dump_path}. "
                "Download from https://mu-star.net/wikidb"
            )
            return

        count = 0
        with self._open_dump(self.dump_path) as f:
            for line in f:
                if count >= self.max_articles:
                    break

                try:
                    article = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                title = article.get("title", "")
                text = article.get("text", "")

                if not self._is_travel_related(text, title):
                    continue

                location = self._extract_location(text, title)
                if location is None:
                    continue

                cleaned_text = self._clean_namuwiki_text(text)
                if len(cleaned_text) < 100:
                    continue

                category = self._categorize_article(text, title)

                count += 1
                yield NamuwikiArticle(
                    title=title,
                    text=cleaned_text,
                    category=category,
                    location=location,
                    metadata={"source": "namuwiki"},
                )

        logger.info(f"Parsed {count:,} travel articles from Namuwiki dump")

    def collect(self) -> List[NamuwikiArticle]:
        """Collect all travel articles.

        Returns:
            List of NamuwikiArticle objects
        """
        return list(self.iterate())

    def save(self) -> Optional[Path]:
        """Save parsed articles to JSONL file.

        Returns:
            Path to output file or None if no dump
        """
        if self.dump_path is None or not self.dump_path.exists():
            logger.warning("No Namuwiki dump available. Skipping.")
            return None

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

        if not articles:
            logger.warning("No articles collected from Namuwiki dump")
            return None

        with open(output_file, "w", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(articles):,} articles to {output_file}")
        return output_file

    def get_stats(self, articles: Optional[List[Dict]] = None) -> Dict:
        """Get collection statistics."""
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
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        dump_path = sys.argv[1]
        parser = NamuwikiDumpParser(dump_path=dump_path)
        output_path = parser.save()
        if output_path:
            print(f"Saved to: {output_path}")
    else:
        print("Usage: python namuwiki_dump.py <dump_path>")
        print("Download dump from: https://mu-star.net/wikidb")
