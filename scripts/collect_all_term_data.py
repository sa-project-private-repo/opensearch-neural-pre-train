#!/usr/bin/env python3
"""
Collect term-level KO-EN parallel data from multiple sources.

Sources:
1. Wikipedia interlanguage links
2. Wikidata labels
3. MUSE bilingual dictionary
4. IT/Tech terminology

Target: 1M+ high-quality term pairs
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional
import requests
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "v11_term_pairs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_valid_korean(text: str) -> bool:
    """Check if text contains Korean characters."""
    return any('\uac00' <= c <= '\ud7a3' for c in text)


def is_valid_english(text: str) -> bool:
    """Check if text contains English characters."""
    return any(c.isalpha() and c.isascii() for c in text)


def clean_text(text: str) -> str:
    """Clean text for consistency."""
    text = text.strip()
    # Remove parenthetical content for cleaner terms
    if '(' in text and ')' in text:
        # Keep the main part before parenthesis
        main_part = text.split('(')[0].strip()
        if main_part:
            return main_part
    return text


# ============================================================
# 1. Wikipedia Interlanguage Links
# ============================================================

def collect_wikipedia_links() -> list:
    """
    Collect KO-EN term pairs from Wikipedia interlanguage links.
    Uses Wikipedia API to get article titles that exist in both languages.
    """
    print("\n" + "="*70)
    print("1. COLLECTING WIKIPEDIA INTERLANGUAGE LINKS")
    print("="*70)

    pairs = []

    # We'll use Wikipedia dumps or API
    # For efficiency, we'll use the Wikipedia API with continue parameter

    api_url = "https://ko.wikipedia.org/w/api.php"

    # Get random articles and their English equivalents
    params = {
        "action": "query",
        "format": "json",
        "generator": "random",
        "grnnamespace": 0,  # Main namespace only
        "grnlimit": 50,
        "prop": "langlinks",
        "lllang": "en",
        "lllimit": 1,
    }

    print("Fetching Wikipedia interlanguage links...")

    # Collect multiple batches
    target_pairs = 100000  # Target 100K from Wikipedia

    with tqdm(total=target_pairs, desc="Wikipedia") as pbar:
        while len(pairs) < target_pairs:
            try:
                response = requests.get(api_url, params=params, timeout=30)
                data = response.json()

                if "query" not in data or "pages" not in data["query"]:
                    continue

                for page_id, page_data in data["query"]["pages"].items():
                    ko_title = page_data.get("title", "")

                    if "langlinks" in page_data:
                        for ll in page_data["langlinks"]:
                            if ll.get("lang") == "en":
                                en_title = ll.get("*", "")

                                if ko_title and en_title:
                                    ko_clean = clean_text(ko_title)
                                    en_clean = clean_text(en_title)

                                    if (is_valid_korean(ko_clean) and
                                        is_valid_english(en_clean) and
                                        len(ko_clean) >= 2 and
                                        len(en_clean) >= 2):
                                        pairs.append({
                                            "ko": ko_clean,
                                            "en": en_clean,
                                            "source": "wikipedia"
                                        })
                                        pbar.update(1)

            except Exception as e:
                print(f"Error: {e}")
                continue

            if len(pairs) >= target_pairs:
                break

    print(f"Collected {len(pairs):,} pairs from Wikipedia")
    return pairs


def collect_wikipedia_from_dump() -> list:
    """
    Alternative: Use pre-extracted Wikipedia titles.
    Downloads and parses Wikipedia language links dump.
    """
    print("\n" + "="*70)
    print("1. COLLECTING WIKIPEDIA LINKS (from HuggingFace)")
    print("="*70)

    pairs = []

    try:
        from datasets import load_dataset

        # Try to load Wikipedia dataset with titles
        print("Loading Wikipedia dataset...")

        # Load Korean Wikipedia
        wiki_ko = load_dataset("wikipedia", "20220301.ko", split="train", streaming=True)

        # We can't directly get interlanguage links from HuggingFace Wikipedia
        # So we'll use a different approach - load pre-computed entity alignments

        print("Wikipedia streaming loaded")

        # Extract titles (limited due to no direct interlanguage info)
        count = 0
        for article in tqdm(wiki_ko, desc="Wikipedia titles", total=50000):
            if count >= 50000:
                break
            title = article.get("title", "")
            if is_valid_korean(title) and len(title) >= 2:
                # We don't have EN equivalent here, will rely on other sources
                count += 1

    except Exception as e:
        print(f"Wikipedia dataset error: {e}")

    return pairs


# ============================================================
# 2. Wikidata Labels
# ============================================================

def collect_wikidata_labels() -> list:
    """
    Collect KO-EN term pairs from Wikidata labels.
    Wikidata has millions of entities with multilingual labels.
    """
    print("\n" + "="*70)
    print("2. COLLECTING WIKIDATA LABELS")
    print("="*70)

    pairs = []

    # Use SPARQL endpoint
    sparql_url = "https://query.wikidata.org/sparql"

    # Query for entities with both Korean and English labels
    # Limited to avoid timeout
    query = """
    SELECT ?item ?koLabel ?enLabel WHERE {
        ?item rdfs:label ?koLabel .
        ?item rdfs:label ?enLabel .
        FILTER(LANG(?koLabel) = "ko")
        FILTER(LANG(?enLabel) = "en")
        FILTER(STRLEN(?koLabel) >= 2)
        FILTER(STRLEN(?enLabel) >= 2)
    }
    LIMIT 100000
    """

    print("Querying Wikidata SPARQL endpoint...")

    try:
        response = requests.get(
            sparql_url,
            params={"query": query, "format": "json"},
            headers={"User-Agent": "TermCollector/1.0"},
            timeout=300  # 5 minute timeout
        )

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", {}).get("bindings", [])

            for item in tqdm(results, desc="Wikidata"):
                ko_label = item.get("koLabel", {}).get("value", "")
                en_label = item.get("enLabel", {}).get("value", "")

                if ko_label and en_label:
                    ko_clean = clean_text(ko_label)
                    en_clean = clean_text(en_label)

                    if (is_valid_korean(ko_clean) and
                        is_valid_english(en_clean) and
                        len(ko_clean) >= 2 and
                        len(en_clean) >= 2):
                        pairs.append({
                            "ko": ko_clean,
                            "en": en_clean,
                            "source": "wikidata"
                        })

            print(f"Collected {len(pairs):,} pairs from Wikidata")
        else:
            print(f"Wikidata query failed: {response.status_code}")

    except Exception as e:
        print(f"Wikidata error: {e}")

    return pairs


# ============================================================
# 3. MUSE Bilingual Dictionary
# ============================================================

def collect_muse_dictionary() -> list:
    """
    Collect KO-EN pairs from MUSE bilingual dictionaries.
    Facebook's MUSE project provides high-quality bilingual word pairs.
    """
    print("\n" + "="*70)
    print("3. COLLECTING MUSE DICTIONARY")
    print("="*70)

    pairs = []

    # MUSE dictionaries are available on GitHub
    muse_url = "https://dl.fbaipublicfiles.com/arrival/dictionaries/ko-en.txt"

    print(f"Downloading MUSE dictionary from {muse_url}...")

    try:
        response = requests.get(muse_url, timeout=60)

        if response.status_code == 200:
            lines = response.text.strip().split('\n')

            for line in tqdm(lines, desc="MUSE"):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    ko_word = parts[0].strip()
                    en_word = parts[1].strip()

                    if (is_valid_korean(ko_word) and
                        is_valid_english(en_word) and
                        len(ko_word) >= 1 and
                        len(en_word) >= 1):
                        pairs.append({
                            "ko": ko_word,
                            "en": en_word,
                            "source": "muse"
                        })

            print(f"Collected {len(pairs):,} pairs from MUSE")
        else:
            print(f"MUSE download failed: {response.status_code}")
            # Try alternative source
            print("Trying alternative MUSE source...")

    except Exception as e:
        print(f"MUSE error: {e}")
        print("Trying to load from local cache or alternative...")

    # Also try reverse dictionary (en-ko)
    muse_url_reverse = "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-ko.txt"

    try:
        response = requests.get(muse_url_reverse, timeout=60)

        if response.status_code == 200:
            lines = response.text.strip().split('\n')

            for line in tqdm(lines, desc="MUSE (reverse)"):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    en_word = parts[0].strip()
                    ko_word = parts[1].strip()

                    if (is_valid_korean(ko_word) and
                        is_valid_english(en_word) and
                        len(ko_word) >= 1 and
                        len(en_word) >= 1):
                        pairs.append({
                            "ko": ko_word,
                            "en": en_word,
                            "source": "muse"
                        })

            print(f"Total MUSE pairs: {len(pairs):,}")

    except Exception as e:
        print(f"MUSE reverse error: {e}")

    return pairs


# ============================================================
# 4. IT/Tech Terminology
# ============================================================

def collect_it_terminology() -> list:
    """
    Collect IT and technical terminology.
    Includes common programming, ML/AI, and tech terms.
    """
    print("\n" + "="*70)
    print("4. COLLECTING IT/TECH TERMINOLOGY")
    print("="*70)

    # Curated IT terminology list
    it_terms = [
        # Machine Learning / AI
        ("머신러닝", "machine learning"),
        ("머신 러닝", "machine learning"),
        ("기계학습", "machine learning"),
        ("딥러닝", "deep learning"),
        ("딥 러닝", "deep learning"),
        ("심층학습", "deep learning"),
        ("인공지능", "artificial intelligence"),
        ("인공 지능", "artificial intelligence"),
        ("자연어처리", "natural language processing"),
        ("자연어 처리", "natural language processing"),
        ("신경망", "neural network"),
        ("컴퓨터비전", "computer vision"),
        ("컴퓨터 비전", "computer vision"),
        ("강화학습", "reinforcement learning"),
        ("강화 학습", "reinforcement learning"),
        ("지도학습", "supervised learning"),
        ("비지도학습", "unsupervised learning"),
        ("전이학습", "transfer learning"),
        ("트랜스포머", "transformer"),
        ("어텐션", "attention"),
        ("임베딩", "embedding"),
        ("벡터", "vector"),
        ("텐서", "tensor"),
        ("그래디언트", "gradient"),
        ("역전파", "backpropagation"),
        ("손실함수", "loss function"),
        ("최적화", "optimization"),
        ("정규화", "regularization"),
        ("드롭아웃", "dropout"),
        ("배치", "batch"),
        ("에폭", "epoch"),
        ("학습률", "learning rate"),
        ("하이퍼파라미터", "hyperparameter"),
        ("오버피팅", "overfitting"),
        ("언더피팅", "underfitting"),
        ("교차검증", "cross validation"),
        ("정확도", "accuracy"),
        ("정밀도", "precision"),
        ("재현율", "recall"),
        ("분류", "classification"),
        ("회귀", "regression"),
        ("클러스터링", "clustering"),
        ("군집화", "clustering"),
        ("차원축소", "dimensionality reduction"),
        ("특성추출", "feature extraction"),
        ("특징추출", "feature extraction"),
        ("데이터증강", "data augmentation"),

        # Programming
        ("프로그래밍", "programming"),
        ("코딩", "coding"),
        ("알고리즘", "algorithm"),
        ("자료구조", "data structure"),
        ("함수", "function"),
        ("변수", "variable"),
        ("클래스", "class"),
        ("객체", "object"),
        ("메서드", "method"),
        ("인스턴스", "instance"),
        ("상속", "inheritance"),
        ("캡슐화", "encapsulation"),
        ("다형성", "polymorphism"),
        ("인터페이스", "interface"),
        ("추상화", "abstraction"),
        ("모듈", "module"),
        ("패키지", "package"),
        ("라이브러리", "library"),
        ("프레임워크", "framework"),
        ("컴파일러", "compiler"),
        ("인터프리터", "interpreter"),
        ("디버깅", "debugging"),
        ("디버그", "debug"),
        ("테스트", "test"),
        ("유닛테스트", "unit test"),
        ("통합테스트", "integration test"),
        ("배포", "deployment"),
        ("버전관리", "version control"),
        ("깃", "git"),
        ("리포지토리", "repository"),
        ("브랜치", "branch"),
        ("커밋", "commit"),
        ("머지", "merge"),
        ("풀리퀘스트", "pull request"),

        # Database
        ("데이터베이스", "database"),
        ("데이터", "data"),
        ("쿼리", "query"),
        ("테이블", "table"),
        ("인덱스", "index"),
        ("키", "key"),
        ("기본키", "primary key"),
        ("외래키", "foreign key"),
        ("조인", "join"),
        ("트랜잭션", "transaction"),
        ("스키마", "schema"),
        ("정규화", "normalization"),
        ("비정규화", "denormalization"),

        # Web / Network
        ("웹", "web"),
        ("웹사이트", "website"),
        ("웹페이지", "webpage"),
        ("웹서버", "web server"),
        ("서버", "server"),
        ("클라이언트", "client"),
        ("프론트엔드", "frontend"),
        ("백엔드", "backend"),
        ("풀스택", "full stack"),
        ("API", "API"),
        ("REST", "REST"),
        ("GraphQL", "GraphQL"),
        ("HTTP", "HTTP"),
        ("HTTPS", "HTTPS"),
        ("URL", "URL"),
        ("도메인", "domain"),
        ("호스팅", "hosting"),
        ("네트워크", "network"),
        ("프로토콜", "protocol"),
        ("라우터", "router"),
        ("스위치", "switch"),
        ("방화벽", "firewall"),
        ("로드밸런서", "load balancer"),
        ("프록시", "proxy"),
        ("캐시", "cache"),
        ("CDN", "CDN"),
        ("DNS", "DNS"),
        ("IP", "IP"),
        ("TCP", "TCP"),
        ("UDP", "UDP"),

        # Cloud
        ("클라우드", "cloud"),
        ("클라우드컴퓨팅", "cloud computing"),
        ("가상화", "virtualization"),
        ("가상머신", "virtual machine"),
        ("컨테이너", "container"),
        ("도커", "docker"),
        ("쿠버네티스", "kubernetes"),
        ("마이크로서비스", "microservice"),
        ("서버리스", "serverless"),
        ("스케일링", "scaling"),
        ("오토스케일링", "autoscaling"),

        # Security
        ("보안", "security"),
        ("사이버보안", "cybersecurity"),
        ("암호화", "encryption"),
        ("복호화", "decryption"),
        ("해시", "hash"),
        ("인증", "authentication"),
        ("권한", "authorization"),
        ("토큰", "token"),
        ("세션", "session"),
        ("쿠키", "cookie"),
        ("SSL", "SSL"),
        ("TLS", "TLS"),
        ("인증서", "certificate"),
        ("방화벽", "firewall"),
        ("취약점", "vulnerability"),
        ("악성코드", "malware"),
        ("랜섬웨어", "ransomware"),
        ("피싱", "phishing"),

        # Hardware
        ("하드웨어", "hardware"),
        ("소프트웨어", "software"),
        ("펌웨어", "firmware"),
        ("컴퓨터", "computer"),
        ("프로세서", "processor"),
        ("CPU", "CPU"),
        ("GPU", "GPU"),
        ("메모리", "memory"),
        ("RAM", "RAM"),
        ("ROM", "ROM"),
        ("저장장치", "storage"),
        ("SSD", "SSD"),
        ("HDD", "HDD"),
        ("마더보드", "motherboard"),
        ("그래픽카드", "graphics card"),
        ("모니터", "monitor"),
        ("키보드", "keyboard"),
        ("마우스", "mouse"),

        # Search / IR
        ("검색", "search"),
        ("검색엔진", "search engine"),
        ("정보검색", "information retrieval"),
        ("인덱싱", "indexing"),
        ("랭킹", "ranking"),
        ("쿼리", "query"),
        ("문서", "document"),
        ("토큰", "token"),
        ("토크나이저", "tokenizer"),
        ("형태소분석", "morphological analysis"),
        ("불용어", "stopword"),
        ("어간추출", "stemming"),
        ("표제어추출", "lemmatization"),
        ("TF-IDF", "TF-IDF"),
        ("BM25", "BM25"),
        ("시맨틱검색", "semantic search"),
        ("벡터검색", "vector search"),
        ("유사도", "similarity"),
        ("코사인유사도", "cosine similarity"),
        ("추천", "recommendation"),
        ("추천시스템", "recommendation system"),
        ("협업필터링", "collaborative filtering"),
        ("콘텐츠기반필터링", "content-based filtering"),

        # Data Science
        ("데이터사이언스", "data science"),
        ("데이터분석", "data analysis"),
        ("데이터마이닝", "data mining"),
        ("빅데이터", "big data"),
        ("데이터시각화", "data visualization"),
        ("대시보드", "dashboard"),
        ("리포트", "report"),
        ("통계", "statistics"),
        ("확률", "probability"),
        ("분포", "distribution"),
        ("평균", "mean"),
        ("중앙값", "median"),
        ("표준편차", "standard deviation"),
        ("분산", "variance"),
        ("상관관계", "correlation"),
        ("회귀분석", "regression analysis"),

        # Business / General
        ("비즈니스", "business"),
        ("스타트업", "startup"),
        ("플랫폼", "platform"),
        ("서비스", "service"),
        ("사용자", "user"),
        ("고객", "customer"),
        ("프로젝트", "project"),
        ("팀", "team"),
        ("리더", "leader"),
        ("관리", "management"),
        ("전략", "strategy"),
        ("분석", "analysis"),
        ("성능", "performance"),
        ("효율", "efficiency"),
        ("생산성", "productivity"),
        ("자동화", "automation"),
        ("워크플로우", "workflow"),
        ("프로세스", "process"),
        ("솔루션", "solution"),
        ("모델", "model"),
        ("시스템", "system"),
        ("아키텍처", "architecture"),
        ("설계", "design"),
        ("개발", "development"),
        ("구현", "implementation"),
        ("운영", "operation"),
        ("유지보수", "maintenance"),
        ("업데이트", "update"),
        ("업그레이드", "upgrade"),
        ("마이그레이션", "migration"),
        ("통합", "integration"),
        ("연동", "integration"),
        ("확장", "extension"),
        ("확장성", "scalability"),
        ("안정성", "stability"),
        ("신뢰성", "reliability"),
        ("가용성", "availability"),
    ]

    pairs = []
    for ko, en in it_terms:
        pairs.append({
            "ko": ko,
            "en": en,
            "source": "it_terminology"
        })

    # Also add variations
    additional_pairs = []
    for pair in pairs:
        ko = pair["ko"]
        en = pair["en"]

        # Add with different spacing/formatting
        if " " not in ko and len(ko) > 3:
            # Add some common suffixes
            for suffix_ko, suffix_en in [("하다", ""), ("하기", ""), ("화", "ization")]:
                if not ko.endswith(suffix_ko):
                    additional_pairs.append({
                        "ko": ko + suffix_ko,
                        "en": en + suffix_en if suffix_en else en,
                        "source": "it_terminology"
                    })

    pairs.extend(additional_pairs)

    print(f"Collected {len(pairs):,} IT/Tech terms")
    return pairs


# ============================================================
# 5. Additional sources - XLEnt Dataset
# ============================================================

def collect_xlent_data() -> list:
    """
    Collect from XLEnt (Cross-Lingual Entity) dataset if available.
    """
    print("\n" + "="*70)
    print("5. COLLECTING XLENT DATA (if available)")
    print("="*70)

    pairs = []

    try:
        from datasets import load_dataset

        # Try to load XLEnt
        print("Trying to load XLEnt dataset...")
        xlent = load_dataset("franbvp/xlent", split="train", streaming=True)

        count = 0
        max_samples = 500000

        for sample in tqdm(xlent, desc="XLEnt", total=max_samples):
            if count >= max_samples:
                break

            # XLEnt format may vary
            try:
                # Try to extract KO-EN pairs
                if "ko" in sample and "en" in sample:
                    ko = sample["ko"].strip()
                    en = sample["en"].strip()

                    if (is_valid_korean(ko) and
                        is_valid_english(en) and
                        len(ko) >= 1 and
                        len(en) >= 1):
                        pairs.append({
                            "ko": ko,
                            "en": en,
                            "source": "xlent"
                        })
                        count += 1
            except:
                continue

        print(f"Collected {len(pairs):,} pairs from XLEnt")

    except Exception as e:
        print(f"XLEnt not available: {e}")

    return pairs


# ============================================================
# Main Collection Pipeline
# ============================================================

def main():
    print("="*70)
    print("TERM-LEVEL KO-EN PARALLEL DATA COLLECTION")
    print("="*70)

    all_pairs = []

    # 1. Wikipedia (may take time, so we'll use API carefully)
    # wikipedia_pairs = collect_wikipedia_links()
    # all_pairs.extend(wikipedia_pairs)
    # For now, skip Wikipedia API due to rate limiting

    # 2. Wikidata
    wikidata_pairs = collect_wikidata_labels()
    all_pairs.extend(wikidata_pairs)

    # 3. MUSE Dictionary
    muse_pairs = collect_muse_dictionary()
    all_pairs.extend(muse_pairs)

    # 4. IT Terminology
    it_pairs = collect_it_terminology()
    all_pairs.extend(it_pairs)

    # 5. XLEnt (if available)
    xlent_pairs = collect_xlent_data()
    all_pairs.extend(xlent_pairs)

    # Deduplicate
    print("\n" + "="*70)
    print("DEDUPLICATION")
    print("="*70)

    seen = set()
    unique_pairs = []

    for pair in tqdm(all_pairs, desc="Deduplicating"):
        key = (pair["ko"].lower(), pair["en"].lower())
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    print(f"Before deduplication: {len(all_pairs):,}")
    print(f"After deduplication: {len(unique_pairs):,}")

    # Save
    print("\n" + "="*70)
    print("SAVING DATA")
    print("="*70)

    output_path = OUTPUT_DIR / "term_pairs.jsonl"

    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in tqdm(unique_pairs, desc="Saving"):
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"Saved to: {output_path}")
    print(f"Total pairs: {len(unique_pairs):,}")

    # Statistics
    sources = defaultdict(int)
    for pair in unique_pairs:
        sources[pair["source"]] += 1

    print("\nPairs by source:")
    for source, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count:,}")

    # Save metadata
    metadata = {
        "total_pairs": len(unique_pairs),
        "sources": dict(sources),
        "output_file": str(output_path),
    }

    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*70)
    print("DATA COLLECTION COMPLETE")
    print("="*70)

    return unique_pairs


if __name__ == "__main__":
    main()
