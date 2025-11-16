#!/usr/bin/env python3
"""
Wikipedia Dump Downloader and Parser

Downloads and parses the latest Korean Wikipedia dump from Wikimedia.

Requirements:
    pip install requests tqdm mwparserfromhell
"""

import os
import re
import json
import bz2
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, Dict, Any
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import time

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 0)
            self.desc = kwargs.get('desc', '')
            self.n = 0

        def update(self, n=1):
            self.n += n
            if self.total:
                print(f"\r{self.desc}: {self.n}/{self.total} ({self.n/self.total*100:.1f}%)", end='')
            else:
                print(f"\r{self.desc}: {self.n}", end='')

        def close(self):
            print()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()


try:
    import mwparserfromhell
    HAS_MWPARSER = True
except ImportError:
    HAS_MWPARSER = False
    print("⚠️  mwparserfromhell not installed. Install with: pip install mwparserfromhell")


class WikipediaDownloader:
    """
    Downloads and parses Korean Wikipedia dumps.

    Example:
        >>> downloader = WikipediaDownloader()
        >>> downloader.download_latest()
        >>> documents = list(downloader.parse_dump())
    """

    BASE_URL = "https://dumps.wikimedia.org/kowiki"

    def __init__(self, dump_dir: str = "dataset/wikipedia_dumps"):
        """
        Initialize downloader.

        Args:
            dump_dir: Directory to store dump files
        """
        self.dump_dir = Path(dump_dir)
        self.dump_dir.mkdir(parents=True, exist_ok=True)

    def get_latest_dump_date(self) -> str:
        """
        Get the latest available dump date.

        Returns:
            Date string (YYYYMMDD format)
        """
        url = f"{self.BASE_URL}/latest/dumpstatus.json"

        try:
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))
                # Extract date from jobs information
                for job_name, job_info in data.get('jobs', {}).items():
                    if 'updated' in job_info:
                        # Parse date from updated timestamp
                        updated = job_info['updated']
                        # Format: YYYY-MM-DD HH:MM:SS
                        date_str = updated.split()[0].replace('-', '')
                        return date_str[:8]
        except Exception as e:
            print(f"⚠️  Could not fetch latest dump date: {e}")
            # Fallback to current date
            return "20251101"

        return "20251101"

    def download_dump(
        self,
        dump_date: str = "latest",
        filename: str = "kowiki-latest-pages-articles.xml.bz2"
    ) -> Path:
        """
        Download Wikipedia dump file.

        Args:
            dump_date: Dump date (YYYYMMDD) or "latest"
            filename: Dump filename

        Returns:
            Path to downloaded file
        """
        url = f"{self.BASE_URL}/{dump_date}/{filename}"
        output_path = self.dump_dir / filename

        # Skip if already downloaded
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"✓ Dump already exists: {output_path} ({file_size_mb:.1f} MB)")
            return output_path

        print(f"📥 Downloading Korean Wikipedia dump...")
        print(f"   URL: {url}")
        print(f"   Output: {output_path}")

        try:
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req) as response:
                total_size = int(response.headers.get('content-length', 0))

                with open(output_path, 'wb') as f:
                    with tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        desc='Downloading'
                    ) as pbar:
                        chunk_size = 8192
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))

            print(f"✅ Download complete!")
            return output_path

        except HTTPError as e:
            if e.code == 404:
                print(f"❌ Dump not found at {url}")
                print(f"   Try using dump_date='latest' or check available dates at:")
                print(f"   https://dumps.wikimedia.org/kowiki/")
            raise

    def parse_dump(
        self,
        dump_path: Path = None,
        max_documents: int = None,
        min_length: int = 100
    ) -> Iterator[Dict[str, Any]]:
        """
        Parse Wikipedia dump and yield clean documents.

        Args:
            dump_path: Path to dump file (default: latest downloaded)
            max_documents: Maximum number of documents to parse
            min_length: Minimum text length to include

        Yields:
            Dictionary with 'title' and 'text' keys
        """
        if dump_path is None:
            # Find latest dump file
            dump_files = list(self.dump_dir.glob("*.xml.bz2"))
            if not dump_files:
                raise FileNotFoundError("No dump file found. Run download_dump() first.")
            dump_path = dump_files[0]

        print(f"📖 Parsing Wikipedia dump: {dump_path}")

        if not HAS_MWPARSER:
            print("⚠️  Using basic XML parsing (install mwparserfromhell for better results)")

        count = 0

        # Parse compressed XML
        with bz2.open(dump_path, 'rt', encoding='utf-8') as f:
            # Use iterparse for memory efficiency
            context = ET.iterparse(f, events=('end',))

            for event, elem in context:
                if elem.tag.endswith('page'):
                    # Extract title and text
                    title = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}title')
                    text_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}text')

                    if title is not None and text_elem is not None and text_elem.text:
                        title_text = title.text
                        wiki_text = text_elem.text

                        # Skip special pages
                        if self._should_skip(title_text):
                            elem.clear()
                            continue

                        # Clean wiki markup
                        clean_text = self._clean_wiki_text(wiki_text)

                        if len(clean_text) >= min_length:
                            yield {
                                'title': title_text,
                                'text': clean_text,
                                'length': len(clean_text)
                            }

                            count += 1
                            if count % 1000 == 0:
                                print(f"   Processed {count:,} documents...")

                            if max_documents and count >= max_documents:
                                print(f"✓ Reached max_documents limit: {max_documents:,}")
                                break

                    # Clear element to free memory
                    elem.clear()

        print(f"✅ Parsing complete: {count:,} documents extracted")

    def _should_skip(self, title: str) -> bool:
        """Check if page should be skipped."""
        skip_prefixes = [
            '파일:', '분류:', '틀:', '위키백과:', 'Wikipedia:',
            'File:', 'Category:', 'Template:', 'Help:', 'Portal:',
            'Special:', 'Media:', 'MediaWiki:', 'User:', 'Talk:'
        ]

        return any(title.startswith(prefix) for prefix in skip_prefixes)

    def _clean_wiki_text(self, text: str) -> str:
        """
        Clean wiki markup from text.

        Args:
            text: Raw wiki text

        Returns:
            Cleaned plain text
        """
        if HAS_MWPARSER:
            # Use mwparserfromhell for better cleaning
            try:
                wikicode = mwparserfromhell.parse(text)
                return wikicode.strip_code()
            except Exception as e:
                # Fallback to basic cleaning
                pass

        # Basic wiki markup removal
        # Remove templates {{...}}
        text = re.sub(r'\{\{[^}]*\}\}', '', text)

        # Remove references <ref>...</ref>
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^>]*\/>', '', text)

        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

        # Remove file/image links
        text = re.sub(r'\[\[파일:.*?\]\]', '', text)
        text = re.sub(r'\[\[File:.*?\]\]', '', text)
        text = re.sub(r'\[\[Image:.*?\]\]', '', text)

        # Convert wiki links [[Link|Text]] -> Text
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)

        # Remove headings markup
        text = re.sub(r'={2,}([^=]+)={2,}', r'\1', text)

        # Remove bold/italic
        text = re.sub(r"'{2,}", '', text)

        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)

        return text.strip()

    def save_to_json(
        self,
        output_path: str,
        max_documents: int = 100000,
        min_length: int = 100
    ):
        """
        Parse dump and save to JSON file.

        Args:
            output_path: Output JSON file path
            max_documents: Maximum documents to save
            min_length: Minimum text length
        """
        print(f"💾 Saving to JSON: {output_path}")

        documents = []
        for doc in self.parse_dump(max_documents=max_documents, min_length=min_length):
            documents.append({
                'title': doc['title'],
                'text': doc['text'][:2000]  # Limit to 2000 chars
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved {len(documents):,} documents to {output_path}")


def main():
    """
    Command-line interface for Wikipedia downloader.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Download and parse Korean Wikipedia')
    parser.add_argument('--download', action='store_true', help='Download dump file')
    parser.add_argument('--parse', action='store_true', help='Parse dump file')
    parser.add_argument('--output', default='dataset/wikipedia_ko.json', help='Output JSON file')
    parser.add_argument('--max-docs', type=int, default=100000, help='Maximum documents')
    parser.add_argument('--dump-date', default='latest', help='Dump date (YYYYMMDD or "latest")')

    args = parser.parse_args()

    downloader = WikipediaDownloader()

    if args.download:
        downloader.download_dump(dump_date=args.dump_date)

    if args.parse:
        downloader.save_to_json(args.output, max_documents=args.max_docs)

    if not args.download and not args.parse:
        print("Usage: python src/wikipedia_downloader.py --download --parse")
        print("   or: python src/wikipedia_downloader.py --download")
        print("   or: python src/wikipedia_downloader.py --parse")


if __name__ == "__main__":
    main()
