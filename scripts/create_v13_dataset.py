#!/usr/bin/env python3
"""
Create v13 dataset: Noun-focused KO-EN term pairs.

This script filters term pairs to keep only nouns, which are the most
important for information retrieval and neural sparse encoding.

Uses:
- NLTK for English POS tagging
- Heuristic filtering for Korean (verb endings, etc.)
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Optional

import nltk
from nltk import pos_tag, word_tokenize

# Download NLTK data if needed
print("Initializing POS taggers...")
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

PROJECT_ROOT = Path(__file__).parent.parent

# Korean verb/adjective endings to filter out
KOREAN_VERB_ENDINGS = [
    '하다', '되다', '시키다', '받다', '주다', '내다', '지다',
    '한다', '된다', '는다', '습니다', '합니다', '입니다',
    '했다', '됐다', '였다', '았다', '었다',
    '하는', '되는', '있는', '없는', '같은',
    '하게', '되게', '있게', '없게',
    '하며', '되며', '하고', '되고',
    '하여', '되어', '해서', '되서',
]

# Korean adverb/adjective patterns
KOREAN_NON_NOUN_PATTERNS = [
    r'.*적으로$',  # ~적으로 (adverb)
    r'.*스럽게$',  # ~스럽게 (adverb)
    r'.*롭게$',    # ~롭게 (adverb)
    r'.*하게$',    # ~하게 (adverb)
]


def is_korean_noun(text: str) -> bool:
    """Check if Korean text looks like a noun (heuristic)."""
    # Filter out verb/adjective endings
    for ending in KOREAN_VERB_ENDINGS:
        if text.endswith(ending):
            return False

    # Filter out adverb patterns
    for pattern in KOREAN_NON_NOUN_PATTERNS:
        if re.match(pattern, text):
            return False

    # Very short text (1 char) might be particles
    if len(text) == 1:
        return False

    return True


def is_english_noun(text: str) -> bool:
    """Check if English text is or contains a noun."""
    try:
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        # English noun tags: NN, NNS, NNP, NNPS
        noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
        for word, tag in pos_tags:
            if tag in noun_tags:
                return True
        return False
    except Exception:
        return False


def is_valid_noun_pair(ko_text: str, en_text: str) -> bool:
    """Check if both Korean and English are nouns."""
    return is_korean_noun(ko_text) and is_english_noun(en_text)


def load_and_filter_muse(data_path: Path) -> list:
    """Load MUSE dictionary and filter for noun pairs."""
    pairs = []
    noun_count = 0
    total_count = 0

    print("\nProcessing MUSE dictionary...")

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get('source') == 'muse':
                total_count += 1
                if is_valid_noun_pair(item['ko'], item['en']):
                    pairs.append(item)
                    noun_count += 1

                if total_count % 5000 == 0:
                    print(f"  Processed {total_count}, Nouns: {noun_count}")

    print(f"  MUSE: {noun_count}/{total_count} noun pairs")

    # Oversample 3x
    return pairs * 3


def load_and_filter_wikidata(data_path: Path) -> list:
    """Load filtered Wikidata and keep noun pairs."""
    pairs = []
    noun_count = 0
    total_count = 0

    print("\nProcessing Wikidata...")

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get('source') == 'wikidata':
                total_count += 1
                if is_valid_noun_pair(item['ko'], item['en']):
                    pairs.append(item)
                    noun_count += 1

                if total_count % 5000 == 0:
                    print(f"  Processed {total_count}, Nouns: {noun_count}")

    print(f"  Wikidata: {noun_count}/{total_count} noun pairs")
    return pairs


def load_it_terminology(data_path: Path) -> list:
    """Load IT terminology (already nouns)."""
    pairs = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get('source') == 'it_terminology':
                pairs.append(item)

    print(f"  IT terminology: {len(pairs)} pairs")

    # Oversample 10x
    return pairs * 10


def create_tech_noun_pairs() -> list:
    """Create tech/IT noun pairs manually verified."""
    tech_nouns = [
        # Computing
        ("컴퓨터", "computer"),
        ("서버", "server"),
        ("클라이언트", "client"),
        ("네트워크", "network"),
        ("데이터베이스", "database"),
        ("알고리즘", "algorithm"),
        ("프로그램", "program"),
        ("소프트웨어", "software"),
        ("하드웨어", "hardware"),
        ("인터페이스", "interface"),
        ("프레임워크", "framework"),
        ("라이브러리", "library"),
        ("모듈", "module"),
        ("패키지", "package"),
        ("클래스", "class"),
        ("객체", "object"),
        ("함수", "function"),
        ("변수", "variable"),
        ("상수", "constant"),
        ("배열", "array"),
        ("리스트", "list"),
        ("딕셔너리", "dictionary"),
        ("스택", "stack"),
        ("큐", "queue"),
        ("트리", "tree"),
        ("그래프", "graph"),
        ("노드", "node"),

        # ML/AI
        ("머신러닝", "machine learning"),
        ("딥러닝", "deep learning"),
        ("인공지능", "artificial intelligence"),
        ("신경망", "neural network"),
        ("모델", "model"),
        ("학습", "learning"),
        ("훈련", "training"),
        ("추론", "inference"),
        ("예측", "prediction"),
        ("분류", "classification"),
        ("클러스터링", "clustering"),
        ("회귀", "regression"),
        ("최적화", "optimization"),
        ("경사", "gradient"),
        ("손실", "loss"),
        ("정확도", "accuracy"),
        ("정밀도", "precision"),
        ("재현율", "recall"),
        ("임베딩", "embedding"),
        ("벡터", "vector"),
        ("행렬", "matrix"),
        ("텐서", "tensor"),
        ("가중치", "weight"),
        ("편향", "bias"),
        ("레이어", "layer"),
        ("어텐션", "attention"),
        ("트랜스포머", "transformer"),
        ("인코더", "encoder"),
        ("디코더", "decoder"),
        ("토큰", "token"),
        ("토크나이저", "tokenizer"),

        # NLP
        ("자연어처리", "natural language processing"),
        ("텍스트", "text"),
        ("문장", "sentence"),
        ("단어", "word"),
        ("형태소", "morpheme"),
        ("품사", "part of speech"),
        ("구문", "syntax"),
        ("의미", "semantics"),
        ("번역", "translation"),
        ("요약", "summarization"),
        ("질의응답", "question answering"),

        # Search/IR
        ("검색", "search"),
        ("검색엔진", "search engine"),
        ("추천", "recommendation"),
        ("추천시스템", "recommender system"),
        ("색인", "index"),
        ("쿼리", "query"),
        ("문서", "document"),
        ("랭킹", "ranking"),
        ("관련성", "relevance"),
        ("유사도", "similarity"),
        ("벡터검색", "vector search"),
        ("키워드", "keyword"),

        # Cloud/Infra
        ("클라우드", "cloud"),
        ("컨테이너", "container"),
        ("쿠버네티스", "kubernetes"),
        ("도커", "docker"),
        ("마이크로서비스", "microservice"),
        ("로드밸런서", "load balancer"),
        ("프록시", "proxy"),
        ("캐시", "cache"),
        ("스토리지", "storage"),
        ("메모리", "memory"),
        ("프로세서", "processor"),
        ("스레드", "thread"),
        ("프로세스", "process"),

        # Data
        ("데이터", "data"),
        ("데이터셋", "dataset"),
        ("피처", "feature"),
        ("라벨", "label"),
        ("스키마", "schema"),
        ("테이블", "table"),
        ("컬럼", "column"),
        ("로우", "row"),
        ("레코드", "record"),
        ("필드", "field"),

        # Security
        ("보안", "security"),
        ("인증", "authentication"),
        ("암호화", "encryption"),
        ("권한", "authorization"),
        ("토큰", "token"),
        ("인증서", "certificate"),
    ]

    pairs = []
    for ko, en in tech_nouns:
        pairs.append({"ko": ko, "en": en, "source": "tech_nouns"})

    # Oversample 15x for strong learning
    return pairs * 15


def create_common_noun_pairs() -> list:
    """Create common noun pairs."""
    common_nouns = [
        # Common concepts
        ("사람", "person"),
        ("시간", "time"),
        ("장소", "place"),
        ("사물", "thing"),
        ("방법", "method"),
        ("문제", "problem"),
        ("해결", "solution"),
        ("결과", "result"),
        ("원인", "cause"),
        ("목표", "goal"),
        ("계획", "plan"),
        ("과정", "process"),
        ("시스템", "system"),
        ("구조", "structure"),
        ("형태", "form"),
        ("종류", "type"),
        ("크기", "size"),
        ("속도", "speed"),
        ("성능", "performance"),
        ("품질", "quality"),
        ("가치", "value"),
        ("비용", "cost"),
        ("효율", "efficiency"),
        ("효과", "effect"),
        ("영향", "impact"),
        ("변화", "change"),
        ("발전", "development"),
        ("성장", "growth"),
        ("감소", "decrease"),
        ("증가", "increase"),
    ]

    pairs = []
    for ko, en in common_nouns:
        pairs.append({"ko": ko, "en": en, "source": "common_nouns"})

    # Oversample 10x
    return pairs * 10


def main():
    print("=" * 70)
    print("Creating v13 Dataset: Noun-Focused KO-EN Pairs")
    print("=" * 70)

    input_path = PROJECT_ROOT / 'dataset' / 'v12_improved' / 'term_pairs.jsonl'
    output_dir = PROJECT_ROOT / 'dataset' / 'v13_nouns'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'term_pairs.jsonl'

    # Load and filter data
    print("\n1. Loading and filtering MUSE dictionary (nouns only)...")
    muse_pairs = load_and_filter_muse(input_path)
    print(f"   MUSE noun pairs (3x): {len(muse_pairs)}")

    print("\n2. Loading and filtering Wikidata (nouns only)...")
    wikidata_pairs = load_and_filter_wikidata(input_path)
    print(f"   Wikidata noun pairs: {len(wikidata_pairs)}")

    print("\n3. Loading IT terminology...")
    it_pairs = load_it_terminology(input_path)
    print(f"   IT pairs (10x): {len(it_pairs)}")

    print("\n4. Creating tech noun pairs...")
    tech_pairs = create_tech_noun_pairs()
    print(f"   Tech noun pairs (15x): {len(tech_pairs)}")

    print("\n5. Creating common noun pairs...")
    common_pairs = create_common_noun_pairs()
    print(f"   Common noun pairs (10x): {len(common_pairs)}")

    # Combine all pairs
    all_pairs = muse_pairs + wikidata_pairs + it_pairs + tech_pairs + common_pairs

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
