"""
Term Pair Extraction from Parallel Corpus

Extracts KO-EN term pairs from parallel sentences using:
1. NER for named entities
2. N-gram matching for technical terms
3. Word alignment heuristics

Target: 1M+ high-quality term pairs from 19M sentence pairs.
"""

import json
import re
from pathlib import Path
from typing import Iterator, Optional
from collections import Counter
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class TermPair:
    """A Korean-English term pair."""
    ko_term: str
    en_term: str
    source: str = "ccmatrix"
    score: float = 1.0


def is_korean(text: str) -> bool:
    """Check if text contains Korean characters."""
    return bool(re.search(r'[가-힣]', text))


def is_english(text: str) -> bool:
    """Check if text contains English letters."""
    return bool(re.search(r'[a-zA-Z]', text))


def extract_korean_terms(text: str) -> list[str]:
    """Extract Korean noun-like terms from text."""
    # Pattern for Korean word sequences (2-10 chars)
    pattern = r'[가-힣]{2,10}'
    terms = re.findall(pattern, text)
    return [t for t in terms if len(t) >= 2]


def extract_english_terms(text: str) -> list[str]:
    """Extract English terms from text."""
    # Pattern for English words and compound terms
    pattern = r'\b[A-Za-z][a-z]*(?:\s+[A-Za-z][a-z]*){0,3}\b'
    terms = re.findall(pattern, text)
    return [t.strip() for t in terms if len(t) >= 2 and len(t.split()) <= 3]


def extract_technical_terms(ko_text: str, en_text: str) -> list[TermPair]:
    """
    Extract technical term pairs using pattern matching.

    Looks for:
    - Parenthetical translations: 머신러닝(Machine Learning)
    - English in Korean text that matches English text
    - Capitalized terms in both
    """
    pairs = []

    # Pattern 1: Korean (English) format
    ko_en_pattern = r'([가-힣]+)\s*[\(（]([A-Za-z][A-Za-z\s]+)[\)）]'
    matches = re.findall(ko_en_pattern, ko_text)
    for ko, en in matches:
        if len(ko) >= 2 and len(en) >= 2:
            pairs.append(TermPair(ko, en.strip()))

    # Pattern 2: English term in Korean text
    en_in_ko = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', ko_text)
    for en_term in en_in_ko:
        if en_term.lower() in en_text.lower():
            # Find corresponding Korean term nearby
            idx = ko_text.find(en_term)
            if idx > 0:
                # Look for Korean term before English
                before = ko_text[:idx]
                ko_terms = extract_korean_terms(before)
                if ko_terms:
                    pairs.append(TermPair(ko_terms[-1], en_term))

    return pairs


def extract_aligned_terms(
    ko_sentences: list[str],
    en_sentences: list[str],
    min_cooccurrence: int = 3,
    max_pairs: int = 1_000_000,
) -> dict[str, Counter]:
    """
    Extract term pairs using co-occurrence alignment.

    Key insight: If a Korean term and English term frequently
    appear in the same sentence pairs, they're likely translations.
    """
    # Count co-occurrences
    ko_en_cooccur: dict[str, Counter] = {}

    print("Counting co-occurrences...")
    for ko_sent, en_sent in tqdm(zip(ko_sentences, en_sentences),
                                   total=len(ko_sentences)):
        ko_terms = set(extract_korean_terms(ko_sent))
        en_terms = set(extract_english_terms(en_sent))

        for ko in ko_terms:
            if ko not in ko_en_cooccur:
                ko_en_cooccur[ko] = Counter()
            for en in en_terms:
                ko_en_cooccur[ko][en] += 1

    return ko_en_cooccur


def filter_term_pairs(
    cooccur: dict[str, Counter],
    min_count: int = 5,
    min_ratio: float = 0.3,
) -> list[TermPair]:
    """
    Filter co-occurrence counts to high-confidence pairs.

    A pair is confident if:
    1. Count >= min_count
    2. The English term is the top translation for Korean term
    3. Ratio of top translation >= min_ratio
    """
    pairs = []

    for ko_term, en_counts in cooccur.items():
        if not en_counts:
            continue

        total = sum(en_counts.values())
        top_en, top_count = en_counts.most_common(1)[0]

        if top_count >= min_count and top_count / total >= min_ratio:
            pairs.append(TermPair(
                ko_term=ko_term,
                en_term=top_en,
                score=top_count / total,
            ))

    return pairs


def process_parallel_corpus(
    ko_file: str,
    en_file: str,
    output_file: str,
    sample_size: Optional[int] = None,
    batch_size: int = 100_000,
) -> int:
    """
    Process parallel corpus and extract term pairs.

    Args:
        ko_file: Path to Korean sentences file
        en_file: Path to English sentences file
        output_file: Path for output JSONL
        sample_size: Number of sentences to process (None = all)
        batch_size: Batch size for processing

    Returns:
        Number of term pairs extracted
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count total lines
    print("Counting lines...")
    with open(ko_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    if sample_size:
        total_lines = min(total_lines, sample_size)

    print(f"Processing {total_lines:,} sentence pairs...")

    # Process in batches
    all_pairs = []
    seen = set()

    with open(ko_file, 'r', encoding='utf-8') as ko_f, \
         open(en_file, 'r', encoding='utf-8') as en_f:

        batch_ko = []
        batch_en = []
        processed = 0

        pbar = tqdm(total=total_lines, desc="Processing")

        for ko_line, en_line in zip(ko_f, en_f):
            if sample_size and processed >= sample_size:
                break

            batch_ko.append(ko_line.strip())
            batch_en.append(en_line.strip())
            processed += 1

            if len(batch_ko) >= batch_size:
                # Extract technical terms
                for ko, en in zip(batch_ko, batch_en):
                    tech_pairs = extract_technical_terms(ko, en)
                    for pair in tech_pairs:
                        key = (pair.ko_term, pair.en_term)
                        if key not in seen:
                            seen.add(key)
                            all_pairs.append(pair)

                pbar.update(len(batch_ko))
                batch_ko = []
                batch_en = []

        # Process remaining
        if batch_ko:
            for ko, en in zip(batch_ko, batch_en):
                tech_pairs = extract_technical_terms(ko, en)
                for pair in tech_pairs:
                    key = (pair.ko_term, pair.en_term)
                    if key not in seen:
                        seen.add(key)
                        all_pairs.append(pair)
            pbar.update(len(batch_ko))

        pbar.close()

    # Co-occurrence based extraction for larger corpus
    if total_lines > 100_000:
        print("\nPerforming co-occurrence analysis on sample...")

        # Sample for co-occurrence
        sample_ko = []
        sample_en = []
        sample_n = min(1_000_000, total_lines)

        with open(ko_file, 'r', encoding='utf-8') as ko_f, \
             open(en_file, 'r', encoding='utf-8') as en_f:

            for i, (ko, en) in enumerate(zip(ko_f, en_f)):
                if i >= sample_n:
                    break
                sample_ko.append(ko.strip())
                sample_en.append(en.strip())

        cooccur = extract_aligned_terms(sample_ko, sample_en)
        cooccur_pairs = filter_term_pairs(cooccur, min_count=10, min_ratio=0.4)

        for pair in cooccur_pairs:
            key = (pair.ko_term, pair.en_term)
            if key not in seen:
                seen.add(key)
                all_pairs.append(pair)

        print(f"Co-occurrence pairs added: {len(cooccur_pairs):,}")

    # Save results
    print(f"\nSaving {len(all_pairs):,} term pairs to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            f.write(json.dumps({
                'ko_term': pair.ko_term,
                'en_term': pair.en_term,
                'source': pair.source,
                'score': pair.score,
            }, ensure_ascii=False) + '\n')

    return len(all_pairs)


def merge_datasets(
    input_files: list[str],
    output_file: str,
    deduplicate: bool = True,
) -> int:
    """
    Merge multiple term pair datasets.

    Args:
        input_files: List of JSONL files to merge
        output_file: Output merged JSONL file
        deduplicate: Remove duplicates

    Returns:
        Total number of pairs in merged file
    """
    seen = set()
    total = 0

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            if not Path(input_file).exists():
                print(f"Skipping missing file: {input_file}")
                continue

            print(f"Processing: {input_file}")
            with open(input_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    try:
                        data = json.loads(line)
                        key = (data['ko_term'], data['en_term'])

                        if deduplicate and key in seen:
                            continue

                        seen.add(key)
                        out_f.write(line)
                        total += 1

                    except json.JSONDecodeError:
                        continue

    print(f"Merged {total:,} pairs to {output_file}")
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract term pairs from parallel corpus")
    parser.add_argument("--ko-file", default="dataset/large_scale/CCMatrix.en-ko.ko")
    parser.add_argument("--en-file", default="dataset/large_scale/CCMatrix.en-ko.en")
    parser.add_argument("--output", default="dataset/large_scale/ccmatrix_terms.jsonl")
    parser.add_argument("--sample", type=int, default=None, help="Sample size (None=all)")

    args = parser.parse_args()

    process_parallel_corpus(
        args.ko_file,
        args.en_file,
        args.output,
        sample_size=args.sample,
    )
