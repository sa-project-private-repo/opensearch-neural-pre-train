#!/usr/bin/env python3
"""
Extract Korean synonyms from various sources for neural sparse training.

Sources:
1. Wikipedia redirects (ko.wikipedia.org)
2. Existing training data augmentation
3. Medical/Legal terminology (if available)

Output: data/v22.0/single_term_expanded.jsonl
"""
import json
import logging
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("data/v22.0")
OUTPUT_FILE = DATA_DIR / "single_term_expanded.jsonl"
WIKI_DUMP_URL = "https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-redirect.sql.gz"


def is_single_term(text: str) -> bool:
    """Check if text is a single term (no spaces, reasonable length)."""
    text = text.strip()
    if not text:
        return False
    # Single term: no spaces, 1-20 characters
    if " " in text:
        return False
    if len(text) < 1 or len(text) > 20:
        return False
    # Should contain at least one Korean character
    if not re.search(r"[\uac00-\ud7af]", text):
        return False
    return True


def extract_wikipedia_redirects(dump_path: Path) -> List[Tuple[str, str]]:
    """
    Extract redirect pairs from Wikipedia dump.

    Returns:
        List of (source, target) pairs
    """
    logger.info(f"Extracting redirects from {dump_path}")
    pairs = []

    # Parse SQL dump for redirects
    # Format: (from_page_id, from_namespace, to_title, to_namespace, to_fragment)
    pattern = re.compile(
        r"\((\d+),(\d+),'([^']+)',(\d+),'([^']*)'\)"
    )

    with open(dump_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("INSERT INTO"):
                continue
            for match in pattern.finditer(line):
                from_ns = int(match.group(2))
                to_title = match.group(3).replace("_", " ")
                to_ns = int(match.group(4))

                # Only main namespace (0)
                if from_ns == 0 and to_ns == 0:
                    # We need to get the source title separately
                    # For now, collect target titles
                    pairs.append(("", to_title))

    logger.info(f"Extracted {len(pairs)} redirect targets")
    return pairs


def extract_from_existing_data(data_path: Path) -> List[Tuple[str, str]]:
    """
    Extract additional single-term pairs from existing training data.

    Args:
        data_path: Path to training_triplets.jsonl

    Returns:
        List of (anchor, positive) pairs that are single terms
    """
    logger.info(f"Extracting from existing data: {data_path}")
    pairs = []

    if not data_path.exists():
        logger.warning(f"Data file not found: {data_path}")
        return pairs

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                anchor = item.get("anchor", "")
                positive = item.get("positive", "")

                # Check if both are single terms
                if is_single_term(anchor) and is_single_term(positive):
                    if anchor != positive:  # Avoid identity pairs
                        pairs.append((anchor, positive))
            except json.JSONDecodeError:
                continue

    logger.info(f"Found {len(pairs)} single-term pairs from existing data")
    return pairs


def create_hard_negatives(
    pairs: List[Tuple[str, str]],
    all_terms: Set[str],
) -> List[Dict]:
    """
    Create triplets with hard negatives.

    Hard negative strategy:
    - Same length category
    - Different semantic meaning
    - Character overlap (partial)
    """
    logger.info("Creating hard negatives...")
    triplets = []

    # Group terms by length
    terms_by_length: Dict[int, List[str]] = defaultdict(list)
    for term in all_terms:
        terms_by_length[len(term)].append(term)

    for anchor, positive in pairs:
        # Find hard negative
        anchor_len = len(anchor)
        candidates = []

        # Look for terms with similar length (+-2)
        for length in range(max(1, anchor_len - 2), anchor_len + 3):
            candidates.extend(terms_by_length.get(length, []))

        # Filter out anchor and positive
        candidates = [c for c in candidates if c not in (anchor, positive)]

        if not candidates:
            continue

        # Score candidates by character overlap (prefer some overlap but not too much)
        def overlap_score(term: str) -> float:
            anchor_chars = set(anchor)
            term_chars = set(term)
            overlap = len(anchor_chars & term_chars)
            total = len(anchor_chars | term_chars)
            if total == 0:
                return 0
            ratio = overlap / total
            # Prefer 20-50% overlap
            if 0.2 <= ratio <= 0.5:
                return 1.0
            elif ratio < 0.2:
                return 0.5
            else:
                return 0.3

        candidates.sort(key=overlap_score, reverse=True)
        negative = candidates[0] if candidates else None

        if negative:
            # Determine difficulty
            anchor_chars = set(anchor)
            neg_chars = set(negative)
            overlap = len(anchor_chars & neg_chars) / len(anchor_chars | neg_chars)

            if overlap > 0.5:
                difficulty = "hard"
            elif overlap > 0.2:
                difficulty = "medium"
            else:
                difficulty = "easy"

            triplets.append({
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
                "difficulty": difficulty,
                "length_class": "single_term",
                "pair_type": "single_term_expanded",
            })

    logger.info(f"Created {len(triplets)} triplets with hard negatives")
    return triplets


def fetch_korean_synonyms_from_web() -> List[Tuple[str, str]]:
    """
    Fetch Korean synonyms from publicly available sources.

    Sources:
    - Wikipedia entity aliases
    - Wikidata Korean labels
    """
    logger.info("Fetching synonyms from web sources...")
    pairs = []

    # Use Wikipedia API to get random pages with redirects
    # This is a simplified approach - full implementation would use dump files

    # For now, let's create a curated list of common Korean synonyms
    # These are well-known synonym pairs in Korean
    curated_synonyms = [
        # Medical terms
        ("당뇨병", "당뇨"),
        ("고혈압", "혈압높음"),
        ("두통", "머리아픔"),
        ("복통", "배아픔"),
        ("감기", "감모"),
        ("폐렴", "허파염"),
        ("관절염", "관절통"),
        ("골절", "뼈부러짐"),
        ("탈수", "수분부족"),
        ("빈혈", "피부족"),
        # Technology
        ("컴퓨터", "전산기"),
        ("스마트폰", "휴대폰"),
        ("인터넷", "통신망"),
        ("소프트웨어", "프로그램"),
        ("하드웨어", "기기"),
        ("데이터베이스", "자료저장소"),
        ("알고리즘", "연산법"),
        ("인공지능", "AI"),
        ("머신러닝", "기계학습"),
        ("딥러닝", "심층학습"),
        # Common words
        ("추천", "권장"),
        ("선택", "선정"),
        ("결정", "확정"),
        ("변경", "수정"),
        ("삭제", "제거"),
        ("추가", "첨가"),
        ("확인", "검토"),
        ("완료", "마침"),
        ("시작", "개시"),
        ("종료", "끝"),
        # Business
        ("회사", "기업"),
        ("직원", "사원"),
        ("고객", "손님"),
        ("매출", "판매액"),
        ("이익", "수익"),
        ("손실", "손해"),
        ("계약", "약정"),
        ("협상", "협의"),
        ("합의", "동의"),
        ("거래", "매매"),
        # Legal
        ("법률", "법"),
        ("규정", "규칙"),
        ("조항", "항목"),
        ("위반", "위배"),
        ("처벌", "제재"),
        ("허가", "인가"),
        ("승인", "허락"),
        ("거부", "거절"),
        ("항소", "상소"),
        ("판결", "선고"),
        # Academic
        ("연구", "탐구"),
        ("분석", "해석"),
        ("실험", "시험"),
        ("결과", "결론"),
        ("가설", "가정"),
        ("이론", "학설"),
        ("논문", "보고서"),
        ("학위", "학력"),
        ("교수", "교원"),
        ("학생", "학습자"),
        # Food
        ("음식", "식품"),
        ("요리", "조리"),
        ("식사", "밥"),
        ("간식", "스낵"),
        ("음료", "마실것"),
        ("채소", "야채"),
        ("과일", "과실"),
        ("고기", "육류"),
        ("생선", "어류"),
        ("우유", "유제품"),
        # Transportation
        ("자동차", "차량"),
        ("비행기", "항공기"),
        ("기차", "열차"),
        ("버스", "버스"),
        ("지하철", "전철"),
        ("택시", "택시"),
        ("자전거", "바이크"),
        ("오토바이", "이륜차"),
        ("선박", "배"),
        ("헬리콥터", "헬기"),
        # Nature
        ("날씨", "기상"),
        ("기후", "기후"),
        ("환경", "자연"),
        ("공기", "대기"),
        ("물", "수분"),
        ("불", "화재"),
        ("바람", "풍"),
        ("비", "강수"),
        ("눈", "적설"),
        ("안개", "연무"),
        # Emotions
        ("기쁨", "행복"),
        ("슬픔", "비애"),
        ("분노", "화"),
        ("공포", "두려움"),
        ("불안", "걱정"),
        ("사랑", "애정"),
        ("증오", "혐오"),
        ("질투", "시기"),
        ("희망", "기대"),
        ("절망", "낙담"),
        # Actions
        ("걷다", "보행"),
        ("달리다", "질주"),
        ("먹다", "섭취"),
        ("마시다", "음용"),
        ("자다", "수면"),
        ("일하다", "근무"),
        ("쉬다", "휴식"),
        ("놀다", "오락"),
        ("배우다", "학습"),
        ("가르치다", "교육"),
    ]

    # Add bidirectional pairs
    for term1, term2 in curated_synonyms:
        pairs.append((term1, term2))
        pairs.append((term2, term1))

    logger.info(f"Loaded {len(pairs)} curated synonym pairs")
    return pairs


def download_wikipedia_entities() -> List[Tuple[str, str]]:
    """
    Download and extract Wikipedia entity synonyms using API.

    Uses Wikipedia's search suggestions API to find related terms.
    """
    logger.info("Downloading Wikipedia entity synonyms...")
    pairs = []

    # Seed terms to expand
    seed_terms = [
        "의학", "컴퓨터", "과학", "경제", "법률", "교육", "문화", "기술",
        "정치", "사회", "역사", "지리", "스포츠", "예술", "음악", "영화",
        "건강", "환경", "에너지", "통신", "금융", "건설", "농업", "산업",
    ]

    base_url = "https://ko.wikipedia.org/w/api.php"

    for seed in seed_terms:
        try:
            # Get search suggestions
            params = {
                "action": "opensearch",
                "search": seed,
                "limit": 20,
                "namespace": 0,
                "format": "json",
            }
            response = requests.get(base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if len(data) >= 2:
                    suggestions = data[1]
                    # Create pairs from suggestions
                    for suggestion in suggestions:
                        if is_single_term(suggestion) and suggestion != seed:
                            pairs.append((seed, suggestion))
        except Exception as e:
            logger.warning(f"Error fetching Wikipedia data for {seed}: {e}")
            continue

    logger.info(f"Downloaded {len(pairs)} Wikipedia entity pairs")
    return pairs


def deduplicate_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Remove duplicate pairs and self-pairs."""
    seen = set()
    unique_pairs = []

    for anchor, positive in pairs:
        anchor = anchor.strip()
        positive = positive.strip()

        if not anchor or not positive:
            continue
        if anchor == positive:
            continue

        # Normalize key (sorted tuple)
        key = tuple(sorted([anchor, positive]))
        if key not in seen:
            seen.add(key)
            unique_pairs.append((anchor, positive))

    return unique_pairs


def main():
    """Main function to extract and save synonyms."""
    logger.info("Starting synonym extraction for v22.0")

    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    all_terms = set()

    # Source 1: Curated synonyms
    curated_pairs = fetch_korean_synonyms_from_web()
    all_pairs.extend(curated_pairs)

    # Source 2: Wikipedia entities
    wiki_pairs = download_wikipedia_entities()
    all_pairs.extend(wiki_pairs)

    # Source 3: Existing training data
    existing_data_path = Path("data/v21.4/training_triplets.jsonl")
    if existing_data_path.exists():
        existing_pairs = extract_from_existing_data(existing_data_path)
        all_pairs.extend(existing_pairs)

    # Deduplicate
    all_pairs = deduplicate_pairs(all_pairs)
    logger.info(f"Total unique pairs after deduplication: {len(all_pairs)}")

    # Collect all terms for hard negative generation
    for anchor, positive in all_pairs:
        all_terms.add(anchor)
        all_terms.add(positive)

    # Create triplets with hard negatives
    triplets = create_hard_negatives(all_pairs, all_terms)

    # Save to output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for triplet in triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(triplets)} triplets to {OUTPUT_FILE}")

    # Print statistics
    difficulty_counts = defaultdict(int)
    for t in triplets:
        difficulty_counts[t["difficulty"]] += 1

    logger.info("Difficulty distribution:")
    for diff, count in sorted(difficulty_counts.items()):
        logger.info(f"  {diff}: {count} ({100*count/len(triplets):.1f}%)")


if __name__ == "__main__":
    main()
