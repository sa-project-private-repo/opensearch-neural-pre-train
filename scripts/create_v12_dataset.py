#!/usr/bin/env python3
"""
Create v12 dataset with improved data composition.

Key changes from v11:
1. Filter Wikidata - remove person names, keep only useful common terms
2. Oversample MUSE dictionary (3x) - it has the real vocabulary pairs
3. Add subword decomposition for compound Korean words
"""

import json
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).parent.parent


def is_korean_char(c: str) -> bool:
    return '\uac00' <= c <= '\ud7a3' or '\u1100' <= c <= '\u11ff' or '\u3130' <= c <= '\u318f'


def is_english_char(c: str) -> bool:
    return c.isalpha() and c.isascii()


def is_person_name(en_text: str) -> bool:
    """Check if the English text looks like a person name."""
    words = en_text.split()
    if len(words) < 2:
        return False

    # Check if all words start with uppercase (typical for names)
    capitalized_count = sum(1 for w in words if w and w[0].isupper())
    if capitalized_count == len(words):
        # Additional check: names typically have 2-4 words
        if 2 <= len(words) <= 4:
            # Check if it contains common name particles
            name_particles = {'von', 'van', 'de', 'la', 'le', 'di', 'da', 'du', 'el', 'al'}
            non_particle_words = [w for w in words if w.lower() not in name_particles]
            if all(w[0].isupper() for w in non_particle_words if w):
                return True
    return False


def is_movie_or_work_title(en_text: str) -> bool:
    """Check if it's likely a movie/book/work title."""
    # Titles with articles at the start
    if en_text.startswith(('The ', 'A ', 'An ')):
        words = en_text.split()
        if len(words) >= 2:
            return True
    return False


def is_useful_term(ko_text: str, en_text: str) -> bool:
    """Check if this is a useful vocabulary term (not a proper noun)."""
    # Skip if too long (likely sentences or titles)
    if len(ko_text) > 15 or len(en_text) > 30:
        return False

    # Skip person names
    if is_person_name(en_text):
        return False

    # Skip movie/work titles
    if is_movie_or_work_title(en_text):
        return False

    # Skip if English is all uppercase (acronyms) and very short
    if en_text.isupper() and len(en_text) <= 3:
        return False

    # Keep terms that have common vocabulary
    # Check if English contains lowercase letters (not just proper nouns)
    has_lowercase = any(c.islower() for c in en_text)

    # For short Korean words (1-3 characters), require at least one lowercase English letter
    if len(ko_text) <= 3:
        return has_lowercase

    return True


def load_muse_pairs(data_path: Path) -> list:
    """Load and return MUSE dictionary pairs (oversample 3x)."""
    pairs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get('source') == 'muse':
                pairs.append(item)

    # Oversample MUSE data 3x since it's the most valuable
    return pairs * 3


def load_it_terminology(data_path: Path) -> list:
    """Load and return IT terminology pairs (oversample 5x)."""
    pairs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get('source') == 'it_terminology':
                pairs.append(item)

    # Oversample IT terminology 5x since it's highly relevant
    return pairs * 5


def load_filtered_wikidata(data_path: Path) -> list:
    """Load and return filtered Wikidata pairs (only useful terms)."""
    pairs = []
    filtered_count = 0

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get('source') == 'wikidata':
                if is_useful_term(item['ko'], item['en']):
                    pairs.append(item)
                else:
                    filtered_count += 1

    print(f"Wikidata: Kept {len(pairs)}, Filtered out {filtered_count}")
    return pairs


def create_subword_decomposition_pairs() -> list:
    """Create subword decomposition pairs for compound Korean words."""
    compound_pairs = [
        # IT/Tech compound words
        ("추천시스템", [("추천", "recommendation"), ("시스템", "system")]),
        ("검색엔진", [("검색", "search"), ("엔진", "engine")]),
        ("머신러닝", [("머신", "machine"), ("러닝", "learning")]),
        ("딥러닝", [("딥", "deep"), ("러닝", "learning")]),
        ("자연어처리", [("자연어", "natural language"), ("처리", "processing")]),
        ("인공지능", [("인공", "artificial"), ("지능", "intelligence")]),
        ("데이터베이스", [("데이터", "data"), ("베이스", "base")]),
        ("클라우드컴퓨팅", [("클라우드", "cloud"), ("컴퓨팅", "computing")]),
        ("네트워크보안", [("네트워크", "network"), ("보안", "security")]),
        ("웹서버", [("웹", "web"), ("서버", "server")]),
        ("데이터분석", [("데이터", "data"), ("분석", "analysis")]),
        ("정보검색", [("정보", "information"), ("검색", "retrieval")]),
        ("시맨틱검색", [("시맨틱", "semantic"), ("검색", "search")]),
        ("벡터검색", [("벡터", "vector"), ("검색", "search")]),
        ("임베딩모델", [("임베딩", "embedding"), ("모델", "model")]),
        ("언어모델", [("언어", "language"), ("모델", "model")]),
        ("텍스트분류", [("텍스트", "text"), ("분류", "classification")]),
        ("기계번역", [("기계", "machine"), ("번역", "translation")]),
        ("음성인식", [("음성", "speech"), ("인식", "recognition")]),
        ("이미지인식", [("이미지", "image"), ("인식", "recognition")]),
        ("객체탐지", [("객체", "object"), ("탐지", "detection")]),
        ("신경망", [("신경", "neural"), ("망", "network")]),
        ("강화학습", [("강화", "reinforcement"), ("학습", "learning")]),
        ("지도학습", [("지도", "supervised"), ("학습", "learning")]),
        ("비지도학습", [("비지도", "unsupervised"), ("학습", "learning")]),
    ]

    pairs = []
    for compound, components in compound_pairs:
        for ko, en in components:
            pairs.append({
                "ko": ko,
                "en": en,
                "source": "decomposition"
            })

    # Repeat decomposition pairs 10x for strong learning
    return pairs * 10


def add_common_vocabulary_pairs() -> list:
    """Add common vocabulary pairs that are important for cross-lingual search."""
    common_pairs = [
        # Verbs
        ("검색하다", "search"),
        ("추천하다", "recommend"),
        ("분석하다", "analyze"),
        ("학습하다", "learn"),
        ("처리하다", "process"),
        ("예측하다", "predict"),
        ("분류하다", "classify"),
        ("탐지하다", "detect"),
        ("인식하다", "recognize"),
        ("생성하다", "generate"),

        # Nouns
        ("알고리즘", "algorithm"),
        ("데이터", "data"),
        ("모델", "model"),
        ("시스템", "system"),
        ("네트워크", "network"),
        ("서버", "server"),
        ("클라이언트", "client"),
        ("인터페이스", "interface"),
        ("프레임워크", "framework"),
        ("라이브러리", "library"),
        ("패키지", "package"),
        ("모듈", "module"),
        ("함수", "function"),
        ("클래스", "class"),
        ("객체", "object"),
        ("변수", "variable"),
        ("상수", "constant"),
        ("배열", "array"),
        ("리스트", "list"),
        ("딕셔너리", "dictionary"),
        ("튜플", "tuple"),
        ("집합", "set"),
        ("문자열", "string"),
        ("정수", "integer"),
        ("실수", "float"),
        ("불리언", "boolean"),

        # Tech terms
        ("검색", "search"),
        ("추천", "recommendation"),
        ("클러스터", "cluster"),
        ("노드", "node"),
        ("인덱스", "index"),
        ("쿼리", "query"),
        ("필터", "filter"),
        ("정렬", "sort"),
        ("집계", "aggregation"),
        ("매핑", "mapping"),
        ("샤드", "shard"),
        ("레플리카", "replica"),
        ("스냅샷", "snapshot"),
        ("백업", "backup"),
        ("복원", "restore"),
        ("모니터링", "monitoring"),
        ("로깅", "logging"),
        ("디버깅", "debugging"),
        ("테스트", "test"),
        ("배포", "deploy"),
        ("빌드", "build"),
        ("컴파일", "compile"),
        ("실행", "execute"),
        ("종료", "terminate"),
        ("시작", "start"),
        ("중지", "stop"),
        ("재시작", "restart"),

        # ML terms
        ("학습", "learning"),
        ("훈련", "training"),
        ("추론", "inference"),
        ("예측", "prediction"),
        ("정확도", "accuracy"),
        ("정밀도", "precision"),
        ("재현율", "recall"),
        ("손실", "loss"),
        ("최적화", "optimization"),
        ("경사", "gradient"),
        ("역전파", "backpropagation"),
        ("과적합", "overfitting"),
        ("정규화", "regularization"),
        ("배치", "batch"),
        ("에폭", "epoch"),
        ("가중치", "weight"),
        ("편향", "bias"),
        ("활성화", "activation"),
        ("드롭아웃", "dropout"),
        ("어텐션", "attention"),
        ("트랜스포머", "transformer"),
        ("인코더", "encoder"),
        ("디코더", "decoder"),
        ("임베딩", "embedding"),
        ("토큰", "token"),
        ("토크나이저", "tokenizer"),
        ("어휘", "vocabulary"),
        ("벡터", "vector"),
        ("행렬", "matrix"),
        ("텐서", "tensor"),
    ]

    pairs = []
    for ko, en in common_pairs:
        pairs.append({
            "ko": ko,
            "en": en,
            "source": "common_vocab"
        })

    # Repeat common vocabulary 10x for strong learning
    return pairs * 10


def main():
    print("=" * 70)
    print("Creating v12 Dataset")
    print("=" * 70)

    input_path = PROJECT_ROOT / 'dataset' / 'v11_term_pairs' / 'term_pairs.jsonl'
    output_dir = PROJECT_ROOT / 'dataset' / 'v12_improved'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'term_pairs.jsonl'

    # Load data from different sources
    print("\n1. Loading MUSE dictionary pairs (3x oversample)...")
    muse_pairs = load_muse_pairs(input_path)
    print(f"   MUSE pairs: {len(muse_pairs)}")

    print("\n2. Loading IT terminology pairs (5x oversample)...")
    it_pairs = load_it_terminology(input_path)
    print(f"   IT pairs: {len(it_pairs)}")

    print("\n3. Loading filtered Wikidata pairs...")
    wikidata_pairs = load_filtered_wikidata(input_path)
    print(f"   Wikidata pairs: {len(wikidata_pairs)}")

    print("\n4. Creating subword decomposition pairs...")
    decomp_pairs = create_subword_decomposition_pairs()
    print(f"   Decomposition pairs: {len(decomp_pairs)}")

    print("\n5. Adding common vocabulary pairs...")
    common_pairs = add_common_vocabulary_pairs()
    print(f"   Common vocab pairs: {len(common_pairs)}")

    # Combine all pairs
    all_pairs = muse_pairs + it_pairs + wikidata_pairs + decomp_pairs + common_pairs

    print(f"\n{'=' * 70}")
    print(f"Total pairs: {len(all_pairs)}")
    print(f"{'=' * 70}")

    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"\nSaved to: {output_path}")

    # Print composition
    sources = defaultdict(int)
    for pair in all_pairs:
        sources[pair['source']] += 1

    print("\nDataset composition:")
    for source, count in sorted(sources.items(), key=lambda x: -x[1]):
        pct = count / len(all_pairs) * 100
        print(f"  {source}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
