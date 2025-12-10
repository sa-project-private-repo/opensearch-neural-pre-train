#!/usr/bin/env python3
"""
Create v14 dataset with improved term coverage.

Improvements over v13:
1. Add missing key IT/AI terms (recommend, natural, neural, etc.)
2. Add abbreviation mappings (AI, ML, NLP, etc.)
3. Add compound term decomposition
4. Increase oversampling of important terms
"""

import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = DATASET_DIR / "v14_augmented"


# Core IT/AI terminology that MUST be included
CORE_IT_TERMS = [
    # Recommendation
    ("추천", "recommend"),
    ("추천", "recommendation"),
    ("추천시스템", "recommendation system"),
    ("추천 시스템", "recommender system"),
    ("협업필터링", "collaborative filtering"),
    ("콘텐츠기반필터링", "content-based filtering"),

    # NLP
    ("자연어", "natural language"),
    ("자연어처리", "natural language processing"),
    ("자연어이해", "natural language understanding"),
    ("자연어생성", "natural language generation"),
    ("형태소분석", "morphological analysis"),
    ("개체명인식", "named entity recognition"),
    ("감성분석", "sentiment analysis"),
    ("텍스트마이닝", "text mining"),
    ("토큰화", "tokenization"),
    ("임베딩", "embedding"),
    ("워드임베딩", "word embedding"),
    ("문장임베딩", "sentence embedding"),

    # AI/ML Core
    ("인공지능", "artificial intelligence"),
    ("인공 지능", "artificial intelligence"),
    ("기계학습", "machine learning"),
    ("기계 학습", "machine learning"),
    ("심층학습", "deep learning"),
    ("딥러닝", "deep learning"),
    ("신경망", "neural network"),
    ("인공신경망", "artificial neural network"),
    ("강화학습", "reinforcement learning"),
    ("지도학습", "supervised learning"),
    ("비지도학습", "unsupervised learning"),
    ("준지도학습", "semi-supervised learning"),
    ("전이학습", "transfer learning"),
    ("메타학습", "meta learning"),

    # Neural Networks
    ("트랜스포머", "transformer"),
    ("어텐션", "attention"),
    ("셀프어텐션", "self-attention"),
    ("멀티헤드어텐션", "multi-head attention"),
    ("컨볼루션", "convolution"),
    ("합성곱", "convolution"),
    ("순환신경망", "recurrent neural network"),
    ("장단기메모리", "long short-term memory"),
    ("게이트순환유닛", "gated recurrent unit"),
    ("오토인코더", "autoencoder"),
    ("변분오토인코더", "variational autoencoder"),
    ("생성적적대신경망", "generative adversarial network"),

    # Computer Vision
    ("컴퓨터비전", "computer vision"),
    ("이미지인식", "image recognition"),
    ("객체탐지", "object detection"),
    ("이미지분류", "image classification"),
    ("이미지분할", "image segmentation"),
    ("얼굴인식", "face recognition"),
    ("광학문자인식", "optical character recognition"),

    # Speech
    ("음성인식", "speech recognition"),
    ("음성합성", "speech synthesis"),
    ("음성처리", "speech processing"),
    ("텍스트음성변환", "text to speech"),
    ("음성텍스트변환", "speech to text"),

    # Search/IR
    ("검색엔진", "search engine"),
    ("검색", "search"),
    ("정보검색", "information retrieval"),
    ("시맨틱검색", "semantic search"),
    ("벡터검색", "vector search"),
    ("유사도검색", "similarity search"),
    ("전문검색", "full-text search"),
    ("역색인", "inverted index"),
    ("랭킹", "ranking"),
    ("리랭킹", "re-ranking"),

    # Data
    ("데이터", "data"),
    ("데이터베이스", "database"),
    ("빅데이터", "big data"),
    ("데이터마이닝", "data mining"),
    ("데이터분석", "data analysis"),
    ("데이터전처리", "data preprocessing"),
    ("데이터증강", "data augmentation"),
    ("특성추출", "feature extraction"),
    ("특성공학", "feature engineering"),

    # Cloud/Infra
    ("클라우드", "cloud"),
    ("클라우드컴퓨팅", "cloud computing"),
    ("서버", "server"),
    ("서버리스", "serverless"),
    ("컨테이너", "container"),
    ("마이크로서비스", "microservice"),
    ("로드밸런싱", "load balancing"),
    ("스케일링", "scaling"),
    ("오케스트레이션", "orchestration"),

    # General Tech
    ("알고리즘", "algorithm"),
    ("프레임워크", "framework"),
    ("라이브러리", "library"),
    ("모듈", "module"),
    ("아키텍처", "architecture"),
    ("인터페이스", "interface"),
    ("프로토콜", "protocol"),
    ("플랫폼", "platform"),
    ("파이프라인", "pipeline"),
    ("워크플로우", "workflow"),

    # LLM specific
    ("대규모언어모델", "large language model"),
    ("언어모델", "language model"),
    ("사전학습", "pre-training"),
    ("파인튜닝", "fine-tuning"),
    ("프롬프트", "prompt"),
    ("프롬프트엔지니어링", "prompt engineering"),
    ("컨텍스트", "context"),
    ("토큰", "token"),
    ("추론", "inference"),
    ("생성", "generation"),
]

# Abbreviation mappings (high frequency)
ABBREVIATIONS = [
    ("AI", "artificial intelligence"),
    ("ML", "machine learning"),
    ("DL", "deep learning"),
    ("NLP", "natural language processing"),
    ("NLU", "natural language understanding"),
    ("NLG", "natural language generation"),
    ("CV", "computer vision"),
    ("RL", "reinforcement learning"),
    ("CNN", "convolutional neural network"),
    ("RNN", "recurrent neural network"),
    ("LSTM", "long short-term memory"),
    ("GRU", "gated recurrent unit"),
    ("GAN", "generative adversarial network"),
    ("VAE", "variational autoencoder"),
    ("LLM", "large language model"),
    ("GPT", "generative pre-trained transformer"),
    ("BERT", "bidirectional encoder representations from transformers"),
    ("API", "application programming interface"),
    ("DB", "database"),
    ("SQL", "structured query language"),
    ("OCR", "optical character recognition"),
    ("TTS", "text to speech"),
    ("STT", "speech to text"),
    ("IR", "information retrieval"),
]

# Compound term decomposition (Korean compound -> individual English words)
COMPOUND_DECOMPOSITION = [
    # 추천시스템 -> recommend + system
    ("추천시스템", ["recommend", "system", "recommendation"]),
    ("검색엔진", ["search", "engine"]),
    ("머신러닝", ["machine", "learning"]),
    ("딥러닝", ["deep", "learning"]),
    ("자연어처리", ["natural", "language", "processing"]),
    ("인공지능", ["artificial", "intelligence"]),
    ("신경망", ["neural", "network"]),
    ("강화학습", ["reinforcement", "learning"]),
    ("컴퓨터비전", ["computer", "vision"]),
    ("음성인식", ["speech", "recognition"]),
    ("데이터베이스", ["data", "database"]),
    ("클라우드컴퓨팅", ["cloud", "computing"]),
    ("객체탐지", ["object", "detection"]),
    ("이미지분류", ["image", "classification"]),
    ("텍스트마이닝", ["text", "mining"]),
    ("빅데이터", ["big", "data"]),
]


def load_v13_dataset() -> list[dict]:
    """Load v13 dataset as base."""
    v13_path = DATASET_DIR / "v13_nouns" / "term_pairs.jsonl"

    if not v13_path.exists():
        print(f"Warning: v13 dataset not found at {v13_path}")
        return []

    data = []
    with open(v13_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


def create_core_term_pairs() -> list[dict]:
    """Create pairs from core IT terminology."""
    pairs = []

    for ko, en in CORE_IT_TERMS:
        pairs.append({
            "ko_text": ko,
            "en_text": en,
            "source": "core_it_terms",
        })

    return pairs


def create_abbreviation_pairs() -> list[dict]:
    """Create pairs from abbreviations."""
    pairs = []

    for abbr, full in ABBREVIATIONS:
        # Abbreviation -> Full form
        pairs.append({
            "ko_text": abbr,
            "en_text": full,
            "source": "abbreviations",
        })

    return pairs


def create_decomposition_pairs() -> list[dict]:
    """Create pairs from compound decomposition."""
    pairs = []

    for ko, en_words in COMPOUND_DECOMPOSITION:
        for en in en_words:
            pairs.append({
                "ko_text": ko,
                "en_text": en,
                "source": "decomposition",
            })

    return pairs


def oversample_important_terms(
    data: list[dict],
    important_sources: list[str],
    factor: int,
) -> list[dict]:
    """Oversample important term categories."""
    result = []

    for item in data:
        source = item.get("source", "")
        if source in important_sources:
            result.extend([item] * factor)
        else:
            result.append(item)

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Creating v14 Augmented Dataset")
    print("=" * 60)

    # 1. Load v13 base
    print("\n1. Loading v13 base dataset...")
    base_data = load_v13_dataset()
    print(f"   Loaded {len(base_data):,} pairs from v13")

    # 2. Add core IT terms
    print("\n2. Adding core IT terminology...")
    core_pairs = create_core_term_pairs()
    print(f"   Added {len(core_pairs):,} core IT term pairs")

    # 3. Add abbreviations
    print("\n3. Adding abbreviation mappings...")
    abbr_pairs = create_abbreviation_pairs()
    print(f"   Added {len(abbr_pairs):,} abbreviation pairs")

    # 4. Add decomposition pairs
    print("\n4. Adding compound decomposition pairs...")
    decomp_pairs = create_decomposition_pairs()
    print(f"   Added {len(decomp_pairs):,} decomposition pairs")

    # 5. Combine all
    all_data = base_data + core_pairs + abbr_pairs + decomp_pairs
    print(f"\n5. Combined: {len(all_data):,} total pairs")

    # 6. Oversample important terms
    print("\n6. Oversampling important terms...")
    important_sources = ["core_it_terms", "abbreviations", "decomposition"]
    all_data = oversample_important_terms(all_data, important_sources, factor=10)
    print(f"   After oversampling: {len(all_data):,} pairs")

    # 7. Statistics
    print("\n7. Dataset composition:")
    sources = Counter(item.get("source", "unknown") for item in all_data)
    for source, count in sources.most_common():
        pct = count / len(all_data) * 100
        print(f"   {source:25s}: {count:8,} ({pct:5.1f}%)")

    # 8. Save
    output_path = OUTPUT_DIR / "term_pairs.jsonl"
    print(f"\n8. Saving to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Created v14 dataset: {len(all_data):,} pairs")

    # Verify key terms
    print("\n9. Verifying key terms...")
    ko_terms = set(item["ko_text"] for item in all_data)
    en_terms = set(item["en_text"].lower() for item in all_data)

    key_checks = [
        ("추천", "recommend"),
        ("자연어", "natural"),
        ("신경망", "neural"),
        ("강화학습", "reinforcement"),
        ("컴퓨터비전", "vision"),
        ("음성인식", "recognition"),
    ]

    for ko, en in key_checks:
        ko_found = any(ko in t for t in ko_terms)
        en_found = any(en in t for t in en_terms)
        status = "✅" if (ko_found and en_found) else "❌"
        print(f"   {status} {ko} → {en}")


if __name__ == "__main__":
    main()
