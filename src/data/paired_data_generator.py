"""Generate (Query, Document) paired data from raw text articles."""

import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from tqdm import tqdm


class PairedDataGenerator:
    """Generate (Query, Document) pairs from articles."""

    def __init__(
        self,
        min_summary_sentences: int = 2,
        max_summary_sentences: int = 3,
        min_paragraph_length: int = 100,
        max_paragraph_length: int = 1000,
    ):
        """
        Initialize paired data generator.

        Args:
            min_summary_sentences: Minimum sentences for summary
            max_summary_sentences: Maximum sentences for summary
            min_paragraph_length: Minimum paragraph length
            max_paragraph_length: Maximum paragraph length
        """
        self.min_summary_sentences = min_summary_sentences
        self.max_summary_sentences = max_summary_sentences
        self.min_paragraph_length = min_paragraph_length
        self.max_paragraph_length = max_paragraph_length

    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Split by period, exclamation, question mark
        # Handle Korean and English punctuation
        pattern = r"([.!?。!?])\s+"
        sentences = re.split(pattern, text)

        # Combine sentences with their punctuation
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sent = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            sent = sent.strip()
            if sent:
                result.append(sent)

        # Handle last sentence if no punctuation
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())

        return result

    def extract_first_paragraph(self, text: str) -> str:
        """
        Extract first paragraph from text.

        Args:
            text: Input text

        Returns:
            First paragraph
        """
        # Split by double newline or empty line
        paragraphs = re.split(r"\n\s*\n", text)

        for para in paragraphs:
            para = para.strip()
            if len(para) >= self.min_paragraph_length:
                return para

        # If no valid paragraph, return first N sentences
        sentences = self.split_sentences(text)
        return " ".join(sentences[: self.max_summary_sentences])

    def generate_title_summary_pairs(
        self, articles_path: str, max_articles: Optional[int] = None
    ) -> Iterator[Dict]:
        """
        Generate (Title, Summary) pairs.

        Args:
            articles_path: Path to articles JSONL file
            max_articles: Maximum number of articles to process

        Yields:
            Dict with 'query' (title) and 'document' (summary)
        """
        count = 0

        with open(articles_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Generating title-summary pairs"):
                if max_articles and count >= max_articles:
                    break

                article = json.loads(line)
                title = article["title"]
                text = article["text"]

                # Extract summary (first N sentences)
                sentences = self.split_sentences(text)
                if len(sentences) >= self.min_summary_sentences:
                    summary = " ".join(
                        sentences[: self.max_summary_sentences]
                    )

                    yield {
                        "query": title,
                        "document": summary,
                        "query_type": "title",
                        "doc_type": "summary",
                        "language": article.get("language", "unknown"),
                        "source_id": article.get("id", ""),
                        "source_url": article.get("url", ""),
                    }

                    count += 1

    def generate_title_paragraph_pairs(
        self, articles_path: str, max_articles: Optional[int] = None
    ) -> Iterator[Dict]:
        """
        Generate (Title, Paragraph) pairs.

        Args:
            articles_path: Path to articles JSONL file
            max_articles: Maximum number of articles to process

        Yields:
            Dict with 'query' (title) and 'document' (paragraph)
        """
        count = 0

        with open(articles_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Generating title-paragraph pairs"):
                if max_articles and count >= max_articles:
                    break

                article = json.loads(line)
                title = article["title"]
                text = article["text"]

                # Extract first paragraph
                paragraph = self.extract_first_paragraph(text)

                if (
                    self.min_paragraph_length
                    <= len(paragraph)
                    <= self.max_paragraph_length
                ):
                    yield {
                        "query": title,
                        "document": paragraph,
                        "query_type": "title",
                        "doc_type": "paragraph",
                        "language": article.get("language", "unknown"),
                        "source_id": article.get("id", ""),
                        "source_url": article.get("url", ""),
                    }

                    count += 1

    def generate_sentence_context_pairs(
        self,
        articles_path: str,
        context_sentences: int = 3,
        max_articles: Optional[int] = None,
    ) -> Iterator[Dict]:
        """
        Generate (Sentence, Context) pairs.

        Args:
            articles_path: Path to articles JSONL file
            context_sentences: Number of surrounding sentences for context
            max_articles: Maximum number of articles to process

        Yields:
            Dict with 'query' (sentence) and 'document' (context)
        """
        count = 0

        with open(articles_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Generating sentence-context pairs"):
                if max_articles and count >= max_articles:
                    break

                article = json.loads(line)
                text = article["text"]
                sentences = self.split_sentences(text)

                # Generate pairs for each sentence
                for i, sent in enumerate(sentences):
                    # Get context (surrounding sentences)
                    start = max(0, i - context_sentences)
                    end = min(len(sentences), i + context_sentences + 1)
                    context = " ".join(sentences[start:end])

                    if (
                        len(context) >= self.min_paragraph_length
                        and len(context) <= self.max_paragraph_length
                    ):
                        yield {
                            "query": sent,
                            "document": context,
                            "query_type": "sentence",
                            "doc_type": "context",
                            "language": article.get("language", "unknown"),
                            "source_id": article.get("id", ""),
                            "source_url": article.get("url", ""),
                        }

                count += 1

    def save_pairs(
        self,
        pairs: Iterator[Dict],
        output_path: str,
        chunk_size: int = 100000,
    ) -> int:
        """
        Save paired data to JSONL files.

        Args:
            pairs: Iterator of paired data
            output_path: Output file path (will be chunked)
            chunk_size: Number of pairs per chunk

        Returns:
            Total number of pairs saved
        """
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        total_count = 0
        chunk_num = 0
        current_chunk = []

        for pair in pairs:
            current_chunk.append(pair)

            if len(current_chunk) >= chunk_size:
                chunk_num += 1
                chunk_file = (
                    output_dir / f"{output_path.stem}_chunk_{chunk_num:03d}.jsonl"
                )

                with open(chunk_file, "w", encoding="utf-8") as f:
                    for p in current_chunk:
                        f.write(json.dumps(p, ensure_ascii=False) + "\n")

                total_count += len(current_chunk)
                print(
                    f"Saved chunk {chunk_num}: {len(current_chunk)} pairs to {chunk_file.name}"
                )
                current_chunk = []

        # Save remaining pairs
        if current_chunk:
            chunk_num += 1
            chunk_file = (
                output_dir / f"{output_path.stem}_chunk_{chunk_num:03d}.jsonl"
            )

            with open(chunk_file, "w", encoding="utf-8") as f:
                for p in current_chunk:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

            total_count += len(current_chunk)
            print(
                f"Saved chunk {chunk_num}: {len(current_chunk)} pairs to {chunk_file.name}"
            )

        print(f"\nTotal: {total_count} pairs saved in {chunk_num} chunks")
        return total_count


if __name__ == "__main__":
    # Example usage
    generator = PairedDataGenerator(
        min_summary_sentences=2,
        max_summary_sentences=3,
        min_paragraph_length=100,
        max_paragraph_length=1000,
    )

    # Generate title-summary pairs
    pairs = generator.generate_title_summary_pairs(
        articles_path="../../dataset/wikipedia/ko_articles_chunk_001.jsonl",
        max_articles=100,
    )

    # Save pairs
    generator.save_pairs(
        pairs=pairs,
        output_path="../../dataset/paired_data/ko_wiki_title_summary.jsonl",
        chunk_size=10000,
    )
