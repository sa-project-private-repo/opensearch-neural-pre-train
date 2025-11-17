"""Wikipedia XML dump parser for extracting clean text from latest dumps."""

import bz2
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from urllib.request import urlopen, urlretrieve

import mwparserfromhell
from tqdm import tqdm


class WikipediaXMLParser:
    """Parser for Wikipedia XML dumps directly from Wikimedia."""

    DUMP_BASE_URL = "https://dumps.wikimedia.org/{lang}wiki/{date}/"
    DUMP_FILE_PATTERN = "{lang}wiki-{date}-pages-articles-multistream.xml.bz2"

    def __init__(
        self,
        language: str = "ko",
        date: str = "latest",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize Wikipedia XML parser.

        Args:
            language: Language code (ko, en, etc.)
            date: Wikipedia dump date (format: YYYYMMDD or "latest")
            cache_dir: Directory to cache downloaded dumps
        """
        self.language = language
        self.date = date
        self.cache_dir = Path(cache_dir) if cache_dir else Path("dataset/wikipedia/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_dump_url(self) -> str:
        """
        Get URL for Wikipedia dump file.

        Returns:
            URL to download dump
        """
        base_url = self.DUMP_BASE_URL.format(lang=self.language, date=self.date)
        filename = self.DUMP_FILE_PATTERN.format(lang=self.language, date=self.date)
        return base_url + filename

    def get_latest_dump_date(self) -> str:
        """
        Get the latest available dump date from Wikimedia.

        Returns:
            Latest dump date in YYYYMMDD format
        """
        import urllib.request
        from html.parser import HTMLParser

        class DumpDateParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.dates = []

            def handle_starttag(self, tag, attrs):
                if tag == "a":
                    for attr, value in attrs:
                        if attr == "href" and re.match(r"^\d{8}/$", value):
                            self.dates.append(value.rstrip("/"))

        url = f"https://dumps.wikimedia.org/{self.language}wiki/"
        with urllib.request.urlopen(url) as response:
            html = response.read().decode("utf-8")

        parser = DumpDateParser()
        parser.feed(html)

        if not parser.dates:
            raise ValueError(f"No dumps found for {self.language}")

        # Return the most recent date
        return sorted(parser.dates)[-1]

    def download_dump(self, force: bool = False) -> Path:
        """
        Download Wikipedia dump file.

        Args:
            force: Force re-download even if cached

        Returns:
            Path to downloaded dump file
        """
        # Get actual date if "latest"
        if self.date == "latest":
            self.date = self.get_latest_dump_date()
            print(f"Latest dump date: {self.date}")

        dump_path = self.cache_dir / f"{self.language}wiki-{self.date}.xml.bz2"

        if dump_path.exists() and not force:
            print(f"Using cached dump: {dump_path}")
            return dump_path

        url = self.get_dump_url()
        print(f"Downloading Wikipedia dump from: {url}")
        print("This may take a while (several GB)...")

        try:
            # Download with progress bar
            def reporthook(count, block_size, total_size):
                if hasattr(reporthook, "pbar"):
                    reporthook.pbar.update(block_size)
                else:
                    reporthook.pbar = tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc="Downloading",
                    )

            urlretrieve(url, dump_path, reporthook=reporthook)
            if hasattr(reporthook, "pbar"):
                reporthook.pbar.close()

            print(f"Downloaded to: {dump_path}")
            return dump_path

        except Exception as e:
            if dump_path.exists():
                dump_path.unlink()
            raise RuntimeError(f"Failed to download dump: {e}")

    def parse_wikitext(self, wikitext: str) -> str:
        """
        Parse MediaWiki markup to plain text.

        Args:
            wikitext: Raw MediaWiki markup

        Returns:
            Plain text
        """
        try:
            # Parse wikitext
            wikicode = mwparserfromhell.parse(wikitext)

            # Strip templates, tags, and get plain text
            text = wikicode.strip_code()

            # Additional cleanup
            text = self.clean_text(text)

            return text

        except Exception as e:
            # Fallback to basic text extraction
            return self.clean_text(wikitext)

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove citations like [1], [2], etc.
        text = re.sub(r"\[\d+\]", "", text)

        # Remove special markup remnants
        text = re.sub(r"\{\{[^}]+\}\}", "", text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove URLs
        text = re.sub(r"http\S+", "", text)

        # Normalize quotation marks
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")

        # Remove multiple periods
        text = re.sub(r"\.{3,}", "...", text)

        return text.strip()

    def iter_articles(self, dump_path: Path) -> Iterator[Dict]:
        """
        Iterate over articles in Wikipedia dump.

        Args:
            dump_path: Path to Wikipedia XML dump

        Yields:
            Article dicts with id, title, text
        """
        print(f"Parsing Wikipedia dump: {dump_path}")

        # XML namespaces - Updated to 0.11 for 2025 dumps
        ns = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}

        # Open compressed dump
        with bz2.open(dump_path, "rt", encoding="utf-8") as f:
            # Streaming XML parsing
            context = ET.iterparse(f, events=("start", "end"))
            context = iter(context)

            current_page = {}
            is_redirect = False

            for event, elem in context:
                if event == "end":
                    tag = elem.tag.replace("{" + ns["mw"] + "}", "")

                    if tag == "title":
                        current_page["title"] = elem.text or ""
                    elif tag == "ns":
                        # Namespace - 0 is main article namespace
                        current_page["ns"] = elem.text or "0"
                    elif tag == "id" and "id" not in current_page:
                        # First ID is page ID
                        current_page["id"] = elem.text or ""
                    elif tag == "redirect":
                        # Mark as redirect page
                        is_redirect = True
                    elif tag == "text":
                        current_page["wikitext"] = elem.text or ""
                    elif tag == "page":
                        # End of page - yield article if valid
                        # Skip redirect pages and non-main namespace
                        if (
                            not is_redirect
                            and "wikitext" in current_page
                            and current_page.get("ns") == "0"
                            and current_page.get("wikitext", "").strip()
                        ):
                            yield {
                                "id": current_page.get("id", ""),
                                "title": current_page.get("title", ""),
                                "wikitext": current_page.get("wikitext", ""),
                                "url": f"https://{self.language}.wikipedia.org/wiki/{current_page.get('title', '').replace(' ', '_')}",
                            }

                        # Reset for next page
                        current_page = {}
                        is_redirect = False

                    # Clear element to save memory
                    elem.clear()

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
        if title.startswith(("Wikipedia:", "Template:", "Category:", "Help:", "Portal:")):
            return False

        # Skip disambiguation pages
        if any(
            keyword in title.lower()
            for keyword in ["(disambiguation)", "목록", "list of", "분류:", "틀:"]
        ):
            return False

        return True

    def process_wikipedia(
        self,
        output_path: str,
        max_articles: Optional[int] = None,
        min_length: int = 100,
        max_length: int = 10000,
        force_download: bool = False,
    ) -> List[Dict]:
        """
        Process Wikipedia dump and save cleaned articles.

        Args:
            output_path: Output file path
            max_articles: Maximum number of articles to process
            min_length: Minimum article length
            max_length: Maximum article length
            force_download: Force re-download of dump

        Returns:
            List of processed articles
        """
        # Download dump
        dump_path = self.download_dump(force=force_download)

        articles = []
        processed = 0

        print(f"Processing Wikipedia articles...")

        iterator = self.iter_articles(dump_path)
        if max_articles:
            iterator = tqdm(iterator, total=max_articles, desc="Processing")
        else:
            iterator = tqdm(iterator, desc="Processing")

        for raw_article in iterator:
            # Parse wikitext to plain text
            text = self.parse_wikitext(raw_article["wikitext"])

            article = {
                "id": raw_article["id"],
                "url": raw_article["url"],
                "title": raw_article["title"],
                "text": text,
                "language": self.language,
            }

            if self.filter_article(article, min_length, max_length):
                articles.append(article)

            processed += 1
            if max_articles and len(articles) >= max_articles:
                break

        # Save articles
        self.save_articles(articles, output_path)

        return articles

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


if __name__ == "__main__":
    # Example usage
    parser = WikipediaXMLParser(language="ko", date="latest")

    # Process Korean Wikipedia (sample)
    articles = parser.process_wikipedia(
        output_path="dataset/wikipedia/ko_articles.jsonl",
        max_articles=100,
    )

    print(f"Processed {len(articles)} Korean articles")
    print(f"Sample: {articles[0]['title']}")
