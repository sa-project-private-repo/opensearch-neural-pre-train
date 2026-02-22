"""Prepare Korean text data for XLM-R masked language modeling pre-training.

Loads Korean Wikipedia and mC4 Korean subset, cleans, deduplicates,
and saves as line-delimited text files for MLM pre-training.
"""

import argparse
import hashlib
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def is_korean_char(char: str) -> bool:
    """Return True if character is in the Hangul syllables Unicode block."""
    return "\uAC00" <= char <= "\uD7AF"


def korean_ratio(text: str) -> float:
    """Return the fraction of characters that are Hangul syllables."""
    if not text:
        return 0.0
    korean_count = sum(1 for ch in text if is_korean_char(ch))
    return korean_count / len(text)


def clean_text(text: str) -> str:
    """Normalize whitespace and strip leading/trailing spaces."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    """Split text on double newlines and return non-empty paragraphs."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def is_valid_paragraph(
    paragraph: str,
    min_length: int,
    min_korean_ratio: float,
) -> bool:
    """Return True if paragraph meets length and Korean content thresholds."""
    if len(paragraph) < min_length:
        return False
    if korean_ratio(paragraph) < min_korean_ratio:
        return False
    return True


def text_hash(text: str) -> str:
    """Return a SHA-256 hex digest of the text for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_wikipedia(
    max_samples: int,
    min_length: int,
    min_korean_ratio: float,
) -> list[str]:
    """Load Korean Wikipedia and extract valid paragraphs.

    Args:
        max_samples: Maximum number of Wikipedia articles to process.
            0 means process all articles.
        min_length: Minimum character length for a paragraph to be kept.
        min_korean_ratio: Minimum fraction of Hangul characters required.

    Returns:
        List of valid, cleaned paragraphs.
    """
    print("Loading Korean Wikipedia (20231101.ko)...", flush=True)
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.ko",
        split="train",
        trust_remote_code=True,
    )

    paragraphs: list[str] = []
    iterator = dataset if max_samples == 0 else dataset.select(range(min(max_samples, len(dataset))))
    desc = "Wikipedia"

    for article in tqdm(iterator, desc=desc, unit="article"):
        text = clean_text(article.get("text", ""))
        if not text:
            continue
        for para in split_paragraphs(text):
            if is_valid_paragraph(para, min_length, min_korean_ratio):
                paragraphs.append(para)

    print(f"  Wikipedia: {len(paragraphs):,} paragraphs collected", flush=True)
    return paragraphs


def collect_mc4(
    max_samples: int,
    min_length: int,
    min_korean_ratio: float,
) -> list[str]:
    """Load Korean mC4 in streaming mode and extract valid paragraphs.

    Args:
        max_samples: Maximum number of mC4 documents to process.
        min_length: Minimum character length for a paragraph to be kept.
        min_korean_ratio: Minimum fraction of Hangul characters required.

    Returns:
        List of valid, cleaned paragraphs.
    """
    print(f"Loading Korean mC4 (streaming, up to {max_samples:,} docs)...", flush=True)
    dataset = load_dataset(
        "mc4",
        "ko",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    paragraphs: list[str] = []

    for doc in tqdm(
        dataset,
        desc="mC4",
        unit="doc",
        total=max_samples,
    ):
        if len(paragraphs) >= max_samples * 5:
            # Avoid unbounded growth; we'll deduplicate later anyway
            break
        text = clean_text(doc.get("text", ""))
        if not text:
            continue
        for para in split_paragraphs(text):
            if is_valid_paragraph(para, min_length, min_korean_ratio):
                paragraphs.append(para)

        # Stop after consuming max_samples documents
        if hasattr(doc, "__iter__"):
            pass

    # The streaming iterator doesn't expose a document count easily,
    # so we cap by iterating with an index.
    print(f"  mC4: {len(paragraphs):,} paragraphs collected", flush=True)
    return paragraphs


def collect_mc4_capped(
    max_docs: int,
    min_length: int,
    min_korean_ratio: float,
) -> list[str]:
    """Load Korean mC4 with a hard cap on document count.

    Args:
        max_docs: Stop after processing this many documents.
        min_length: Minimum character length for a paragraph to be kept.
        min_korean_ratio: Minimum fraction of Hangul characters required.

    Returns:
        List of valid, cleaned paragraphs.
    """
    print(f"Loading Korean mC4 (streaming, max {max_docs:,} docs)...", flush=True)
    dataset = load_dataset(
        "mc4",
        "ko",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    paragraphs: list[str] = []
    processed = 0

    for doc in tqdm(
        dataset,
        desc="mC4",
        unit="doc",
        total=max_docs,
    ):
        if processed >= max_docs:
            break
        text = clean_text(doc.get("text", ""))
        if not text:
            processed += 1
            continue
        for para in split_paragraphs(text):
            if is_valid_paragraph(para, min_length, min_korean_ratio):
                paragraphs.append(para)
        processed += 1

    print(f"  mC4: {len(paragraphs):,} paragraphs collected from {processed:,} docs", flush=True)
    return paragraphs


def deduplicate(paragraphs: list[str]) -> list[str]:
    """Remove duplicate paragraphs using SHA-256 hashing.

    Args:
        paragraphs: List of text paragraphs, possibly containing duplicates.

    Returns:
        Deduplicated list preserving first-occurrence order.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for para in tqdm(paragraphs, desc="Deduplicating", unit="para"):
        h = text_hash(para)
        if h not in seen:
            seen.add(h)
            unique.append(para)
    return unique


def split_train_val(
    paragraphs: list[str],
    val_ratio: float,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Randomly shuffle and split paragraphs into train and validation sets.

    Args:
        paragraphs: Full list of deduplicated paragraphs.
        val_ratio: Fraction to reserve for validation (0 < val_ratio < 1).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_paragraphs, val_paragraphs).
    """
    random.seed(seed)
    shuffled = paragraphs.copy()
    random.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1 - val_ratio)))
    return shuffled[:split_idx], shuffled[split_idx:]


def save_lines(paragraphs: list[str], path: Path) -> None:
    """Write paragraphs as a line-delimited text file.

    Args:
        paragraphs: List of text strings, one per line.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for para in tqdm(paragraphs, desc=f"Writing {path.name}", unit="line"):
            f.write(para.replace("\n", " ") + "\n")


def print_stats(train: list[str], val: list[str]) -> None:
    """Print summary statistics for the prepared dataset.

    Args:
        train: Training set paragraphs.
        val: Validation set paragraphs.
    """
    total = len(train) + len(val)
    all_lines = train + val
    avg_len = sum(len(p) for p in all_lines) / total if total else 0
    avg_korean = (
        sum(korean_ratio(p) for p in all_lines) / total if total else 0
    )

    print("\n=== Dataset Statistics ===")
    print(f"Total lines  : {total:>12,}")
    print(f"  Train      : {len(train):>12,} ({len(train)/total*100:.1f}%)")
    print(f"  Val        : {len(val):>12,} ({len(val)/total*100:.1f}%)")
    print(f"Avg length   : {avg_len:>12.1f} chars")
    print(f"Avg Korean % : {avg_korean*100:>11.1f}%")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare Korean text data for XLM-R MLM pre-training."
    )
    parser.add_argument(
        "--output-dir",
        default="data/mlm_korean",
        help="Directory to write train.txt and val.txt (default: data/mlm_korean)",
    )
    parser.add_argument(
        "--max-wiki",
        type=int,
        default=0,
        help="Max Wikipedia articles to process; 0 = all (default: 0)",
    )
    parser.add_argument(
        "--max-mc4",
        type=int,
        default=500_000,
        help="Max mC4 documents to process (default: 500000)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum paragraph character length (default: 50)",
    )
    parser.add_argument(
        "--korean-ratio",
        type=float,
        default=0.3,
        help="Minimum fraction of Hangul characters (default: 0.3)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Fraction of data reserved for validation (default: 0.05)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: collect, clean, deduplicate, and save Korean MLM data."""
    args = parse_args()
    output_dir = Path(args.output_dir)

    # Collect from sources
    wiki_paras = collect_wikipedia(
        max_samples=args.max_wiki,
        min_length=args.min_length,
        min_korean_ratio=args.korean_ratio,
    )
    mc4_paras = collect_mc4_capped(
        max_docs=args.max_mc4,
        min_length=args.min_length,
        min_korean_ratio=args.korean_ratio,
    )

    all_paras = wiki_paras + mc4_paras
    print(f"\nTotal before dedup: {len(all_paras):,}", flush=True)

    # Deduplicate
    unique_paras = deduplicate(all_paras)
    print(f"Total after dedup : {len(unique_paras):,}", flush=True)

    if not unique_paras:
        print("ERROR: No valid paragraphs collected. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Split
    train, val = split_train_val(unique_paras, val_ratio=args.val_ratio)

    # Save
    save_lines(train, output_dir / "train.txt")
    save_lines(val, output_dir / "val.txt")

    # Stats
    print_stats(train, val)
    print(f"\nOutput written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
