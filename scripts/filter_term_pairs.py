"""
Filter and clean term pairs for high quality.

Quality criteria:
1. Korean term must be mostly Korean characters
2. English term must be mostly English letters
3. Both terms should be reasonable length (2-50 chars)
4. Remove common words and stop words
5. Prioritize noun-like terms
"""

import json
import re
from pathlib import Path
from typing import Iterator
from tqdm import tqdm


# Korean stop words (particles, endings)
KO_STOPWORDS = {
    "이다", "있다", "하다", "되다", "않다", "없다", "같다", "보다",
    "때문", "그것", "것이", "수있", "있는", "있어", "하는", "되는",
    "것은", "것을", "것이", "수가", "들이", "에서", "으로", "하고",
    "라고", "에게", "처럼", "만큼", "대해", "위해", "통해", "따라",
}

# English stop words
EN_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "shall", "it", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "what", "which", "who", "whom", "why", "how", "when", "where",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "also", "now", "here", "there", "then",
    "if", "or", "and", "but", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
}


def has_cyrillic(text: str) -> bool:
    """Check if text contains Cyrillic characters (including extended)."""
    # Main Cyrillic range: U+0400-U+04FF
    # Extended Cyrillic: U+0500-U+052F
    return bool(re.search(r'[\u0400-\u052F]', text))


def has_special_noise(text: str) -> bool:
    """Check if text contains special noise patterns."""
    noise_patterns = [
        r'[║╔╗╚╝═─┌┐└┘├┤┬┴┼]',  # Box drawing
        r'[ღƸ̵̡Ӝ̵̨̄]',  # Special unicode
        r'카톡:',  # Spam patterns
        r'соm|сom',  # Cyrillic 'com'
        r'[〖〗【】]',  # CJK brackets often in spam
        r'\d{5,}',  # Long numbers
    ]
    for pattern in noise_patterns:
        if re.search(pattern, text):
            return True
    return False


def is_valid_korean(text: str, min_ratio: float = 0.5) -> bool:
    """Check if text is valid Korean term."""
    if len(text) < 2 or len(text) > 50:
        return False

    # Reject if contains Cyrillic
    if has_cyrillic(text):
        return False

    # Reject if contains special noise
    if has_special_noise(text):
        return False

    # Count Korean characters
    korean_chars = len(re.findall(r'[가-힣]', text))
    total_chars = len(text.replace(' ', ''))

    if total_chars == 0:
        return False

    ratio = korean_chars / total_chars
    return ratio >= min_ratio


def is_valid_english(text: str, min_ratio: float = 0.7) -> bool:
    """Check if text is valid English term."""
    if len(text) < 2 or len(text) > 50:
        return False

    # Reject if contains Cyrillic
    if has_cyrillic(text):
        return False

    # Reject if contains special noise
    if has_special_noise(text):
        return False

    # Count English letters
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    total_chars = len(text.replace(' ', ''))

    if total_chars == 0:
        return False

    ratio = english_chars / total_chars
    return ratio >= min_ratio


def is_stopword(ko_term: str, en_term: str) -> bool:
    """Check if terms are stop words."""
    ko_lower = ko_term.lower()
    en_lower = en_term.lower()

    # Check Korean stop words
    for sw in KO_STOPWORDS:
        if ko_lower == sw or ko_lower.endswith(sw):
            return True

    # Check English stop words
    en_words = set(en_lower.split())
    if en_words.issubset(EN_STOPWORDS):
        return True

    if en_lower in EN_STOPWORDS:
        return True

    return False


def is_quality_pair(ko_term: str, en_term: str) -> bool:
    """
    Check if a term pair meets quality criteria.
    """
    # 1. Basic validation
    if not is_valid_korean(ko_term):
        return False

    if not is_valid_english(en_term):
        return False

    # 2. Not stop words
    if is_stopword(ko_term, en_term):
        return False

    # 3. Reasonable length ratio (not too different)
    len_ratio = len(ko_term) / max(1, len(en_term))
    if len_ratio < 0.1 or len_ratio > 10:
        return False

    # 4. English should not be too long (likely a sentence)
    if len(en_term.split()) > 4:
        return False

    # 5. Korean should not contain too many English characters
    en_in_ko = len(re.findall(r'[a-zA-Z]', ko_term))
    if en_in_ko > len(ko_term) * 0.3:
        return False

    return True


def filter_term_pairs(
    input_file: str,
    output_file: str,
    min_score: float = 0.0,
) -> tuple[int, int]:
    """
    Filter term pairs file for quality.

    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file
        min_score: Minimum score threshold

    Returns:
        (original_count, filtered_count)
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original = 0
    filtered = 0
    seen = set()

    with open(input_path, 'r', encoding='utf-8') as in_f, \
         open(output_path, 'w', encoding='utf-8') as out_f:

        for line in tqdm(in_f, desc="Filtering"):
            original += 1

            try:
                data = json.loads(line)
                ko_term = data.get('ko_term', '').strip()
                en_term = data.get('en_term', '').strip()
                score = data.get('score', 1.0)

                # Skip low score
                if score < min_score:
                    continue

                # Quality check
                if not is_quality_pair(ko_term, en_term):
                    continue

                # Deduplicate
                key = (ko_term.lower(), en_term.lower())
                if key in seen:
                    continue
                seen.add(key)

                # Write
                out_f.write(json.dumps({
                    'ko_term': ko_term,
                    'en_term': en_term,
                    'source': data.get('source', 'unknown'),
                }, ensure_ascii=False) + '\n')
                filtered += 1

            except json.JSONDecodeError:
                continue

    print(f"Filtered: {original:,} -> {filtered:,} ({filtered/original*100:.1f}%)")
    return original, filtered


def process_xlent_data(
    ko_file: str,
    en_file: str,
    output_file: str,
) -> int:
    """
    Process XLEnt moses format data.

    XLEnt is specifically designed for entity alignment,
    so quality is generally higher.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    seen = set()

    with open(ko_file, 'r', encoding='utf-8') as ko_f, \
         open(en_file, 'r', encoding='utf-8') as en_f, \
         open(output_path, 'w', encoding='utf-8') as out_f:

        for ko_line, en_line in tqdm(zip(ko_f, en_f), desc="Processing XLEnt"):
            ko_term = ko_line.strip()
            en_term = en_line.strip()

            if not ko_term or not en_term:
                continue

            # Basic quality check (XLEnt is already curated)
            if len(ko_term) < 2 or len(en_term) < 2:
                continue

            if len(ko_term) > 50 or len(en_term) > 50:
                continue

            # Deduplicate
            key = (ko_term.lower(), en_term.lower())
            if key in seen:
                continue
            seen.add(key)

            out_f.write(json.dumps({
                'ko_term': ko_term,
                'en_term': en_term,
                'source': 'xlent',
            }, ensure_ascii=False) + '\n')
            count += 1

    print(f"Processed XLEnt: {count:,} pairs")
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--min-score", type=float, default=0.0)

    args = parser.parse_args()

    filter_term_pairs(args.input, args.output, args.min_score)
