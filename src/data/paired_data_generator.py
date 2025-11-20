"""Generate (Query, Document) paired data from raw text articles."""

import json
import random
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

    def split_and_save_pairs(
        self,
        pairs: Iterator[Dict],
        output_dir: str,
        prefix: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        chunk_size: int = 100000,
        seed: int = 42,
    ) -> Tuple[int, int, int]:
        """
        Split paired data into train/val/test sets and save to separate files.

        Args:
            pairs: Iterator of paired data
            output_dir: Output directory
            prefix: File prefix (e.g., "ko_wiki_title_summary")
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            chunk_size: Number of pairs per chunk
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_count, val_count, test_count)
        """
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed for reproducibility
        random.seed(seed)

        # Load all pairs into memory for shuffling
        print(f"Loading pairs into memory for splitting...")
        all_pairs = list(pairs)
        total_count = len(all_pairs)

        print(f"Loaded {total_count:,} pairs")
        print(f"Split ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")

        # Shuffle for random split
        random.shuffle(all_pairs)

        # Calculate split indices
        train_end = int(total_count * train_ratio)
        val_end = train_end + int(total_count * val_ratio)

        # Split data
        train_pairs = all_pairs[:train_end]
        val_pairs = all_pairs[train_end:val_end]
        test_pairs = all_pairs[val_end:]

        print(f"\nSplit sizes:")
        print(f"  Train: {len(train_pairs):,} ({len(train_pairs)/total_count:.1%})")
        print(f"  Val:   {len(val_pairs):,} ({len(val_pairs)/total_count:.1%})")
        print(f"  Test:  {len(test_pairs):,} ({len(test_pairs)/total_count:.1%})")

        # Save each split
        train_count = self._save_split(
            train_pairs, output_dir, f"{prefix}_train", chunk_size
        )
        val_count = self._save_split(
            val_pairs, output_dir, f"{prefix}_val", chunk_size
        )
        test_count = self._save_split(
            test_pairs, output_dir, f"{prefix}_test", chunk_size
        )

        print(f"\n✓ Saved splits to {output_dir}")
        return train_count, val_count, test_count

    def _save_split(
        self,
        pairs: List[Dict],
        output_dir: Path,
        prefix: str,
        chunk_size: int,
    ) -> int:
        """
        Save a data split to chunked JSONL files.

        Args:
            pairs: List of paired data
            output_dir: Output directory
            prefix: File prefix
            chunk_size: Number of pairs per chunk

        Returns:
            Number of pairs saved
        """
        total_count = 0
        chunk_num = 0

        for i in range(0, len(pairs), chunk_size):
            chunk = pairs[i:i + chunk_size]
            chunk_num += 1

            chunk_file = output_dir / f"{prefix}_chunk_{chunk_num:03d}.jsonl"

            with open(chunk_file, "w", encoding="utf-8") as f:
                for pair in chunk:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")

            total_count += len(chunk)

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
