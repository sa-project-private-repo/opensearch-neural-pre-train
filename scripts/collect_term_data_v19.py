#!/usr/bin/env python3
"""
Enhanced term-level KO-EN parallel data collection for v19.

Sources:
1. MUSE bilingual dictionary (high-quality, ~20K pairs)
2. Wikidata labels (multiple queries, ~50K+ pairs)
3. Wikipedia API (with rate limiting, ~20K pairs)
4. IT/Tech terminology (curated, ~500 pairs)
5. OPUS parallel corpus terms (if available)

Target: 50K+ high-quality term pairs
"""

import json
import os
import sys
import time
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Optional, Tuple
import requests
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "v19_high_quality"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_valid_korean(text: str) -> bool:
    """Check if text contains Korean characters."""
    return any('\uac00' <= c <= '\ud7a3' for c in text)


def is_valid_english(text: str) -> bool:
    """Check if text is valid English (letters only, no special chars)."""
    if not text:
        return False
    # Must contain at least one ASCII letter
    has_letter = any(c.isalpha() and c.isascii() for c in text)
    # Should not be all uppercase abbreviations longer than 5 chars
    if text.isupper() and len(text) > 5:
        return False
    return has_letter


def clean_text(text: str) -> str:
    """Clean text for consistency."""
    text = text.strip()
    # Remove parenthetical content
    if '(' in text and ')' in text:
        main_part = text.split('(')[0].strip()
        if main_part:
            return main_part
    return text


def is_single_word(text: str) -> bool:
    """Check if text is a single word (no spaces)."""
    return ' ' not in text.strip()


def extract_english_words(text: str) -> List[str]:
    """Extract individual English words from a phrase."""
    words = []
    for word in text.split():
        word = word.strip().lower()
        if word and word.isalpha() and word.isascii() and len(word) >= 2:
            words.append(word)
    return words


# ============================================================
# 1. MUSE Bilingual Dictionary
# ============================================================

def collect_muse_dictionary() -> List[Dict]:
    """Collect KO-EN pairs from MUSE bilingual dictionaries."""
    print("\n" + "=" * 70)
    print("1. COLLECTING MUSE DICTIONARY")
    print("=" * 70)

    pairs = []

    # KO -> EN dictionary
    muse_urls = [
        ("https://dl.fbaipublicfiles.com/arrival/dictionaries/ko-en.txt", "ko", "en"),
        ("https://dl.fbaipublicfiles.com/arrival/dictionaries/en-ko.txt", "en", "ko"),
    ]

    for url, src_lang, tgt_lang in muse_urls:
        print(f"Downloading from {url}...")
        try:
            response = requests.get(url, timeout=120, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                # Force UTF-8 encoding for Korean characters
                response.encoding = 'utf-8'
                content = response.text.strip()
                if not content:
                    print(f"Empty response from {url}")
                    continue

                lines = content.split('\n')
                print(f"Got {len(lines)} lines")

                for line in tqdm(lines, desc=f"MUSE ({src_lang}->{tgt_lang})"):
                    parts = line.strip().split()  # Split by whitespace instead of tab
                    if len(parts) >= 2:
                        if src_lang == "ko":
                            ko_word, en_word = parts[0].strip(), parts[1].strip()
                        else:
                            en_word, ko_word = parts[0].strip(), parts[1].strip()

                        if (is_valid_korean(ko_word) and
                            is_valid_english(en_word) and
                            len(ko_word) >= 2 and
                            len(en_word) >= 2):
                            pairs.append({
                                "ko": ko_word,
                                "en": en_word.lower(),
                                "source": "muse"
                            })
            else:
                print(f"Failed: {response.status_code}")
        except Exception as e:
            print(f"Error: {e}")

    print(f"Collected {len(pairs):,} pairs from MUSE")
    return pairs


# ============================================================
# 2. Wikidata Labels (Multiple Queries)
# ============================================================

def collect_wikidata_labels() -> List[Dict]:
    """Collect KO-EN term pairs from Wikidata with multiple queries."""
    print("\n" + "=" * 70)
    print("2. COLLECTING WIKIDATA LABELS")
    print("=" * 70)

    pairs = []
    sparql_url = "https://query.wikidata.org/sparql"

    # Multiple queries for different entity types
    queries = [
        # General entities
        """
        SELECT ?koLabel ?enLabel WHERE {
            ?item wdt:P31 ?type .
            ?item rdfs:label ?koLabel . FILTER(LANG(?koLabel) = "ko")
            ?item rdfs:label ?enLabel . FILTER(LANG(?enLabel) = "en")
            FILTER(STRLEN(?koLabel) >= 2 && STRLEN(?koLabel) <= 20)
            FILTER(STRLEN(?enLabel) >= 2 && STRLEN(?enLabel) <= 30)
        } LIMIT 30000
        """,
        # Organizations
        """
        SELECT ?koLabel ?enLabel WHERE {
            ?item wdt:P31/wdt:P279* wd:Q43229 .
            ?item rdfs:label ?koLabel . FILTER(LANG(?koLabel) = "ko")
            ?item rdfs:label ?enLabel . FILTER(LANG(?enLabel) = "en")
            FILTER(STRLEN(?koLabel) >= 2)
        } LIMIT 20000
        """,
        # Scientific concepts
        """
        SELECT ?koLabel ?enLabel WHERE {
            ?item wdt:P31/wdt:P279* wd:Q35120 .
            ?item rdfs:label ?koLabel . FILTER(LANG(?koLabel) = "ko")
            ?item rdfs:label ?enLabel . FILTER(LANG(?enLabel) = "en")
            FILTER(STRLEN(?koLabel) >= 2)
        } LIMIT 20000
        """,
        # Software/Technology
        """
        SELECT ?koLabel ?enLabel WHERE {
            ?item wdt:P31/wdt:P279* wd:Q7397 .
            ?item rdfs:label ?koLabel . FILTER(LANG(?koLabel) = "ko")
            ?item rdfs:label ?enLabel . FILTER(LANG(?enLabel) = "en")
            FILTER(STRLEN(?koLabel) >= 2)
        } LIMIT 10000
        """,
    ]

    for i, query in enumerate(queries):
        print(f"Executing Wikidata query {i + 1}/{len(queries)}...")
        try:
            response = requests.get(
                sparql_url,
                params={"query": query, "format": "json"},
                headers={"User-Agent": "TermCollector/2.0 (v19 data collection)"},
                timeout=300
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", {}).get("bindings", [])

                for item in tqdm(results, desc=f"Wikidata Q{i + 1}"):
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
                                "en": en_clean.lower(),
                                "source": "wikidata"
                            })
            else:
                print(f"Query {i + 1} failed: {response.status_code}")

            # Rate limiting
            time.sleep(2)

        except Exception as e:
            print(f"Wikidata error: {e}")

    print(f"Collected {len(pairs):,} pairs from Wikidata")
    return pairs


# ============================================================
# 3. Wikipedia API (with rate limiting)
# ============================================================

def collect_wikipedia_links(target_pairs: int = 20000) -> List[Dict]:
    """Collect KO-EN term pairs from Wikipedia interlanguage links."""
    print("\n" + "=" * 70)
    print("3. COLLECTING WIKIPEDIA INTERLANGUAGE LINKS")
    print("=" * 70)

    pairs = []
    api_url = "https://ko.wikipedia.org/w/api.php"

    # Use category-based approach for more structured data
    categories = [
        "분류:컴퓨터_과학",
        "분류:인공지능",
        "분류:기계_학습",
        "분류:프로그래밍_언어",
        "분류:소프트웨어",
        "분류:인터넷",
        "분류:전자_공학",
        "분류:정보_기술",
        "분류:과학",
        "분류:기술",
    ]

    # Also use random pages
    print("Fetching Wikipedia pages...")

    # Method 1: Category members
    for category in categories[:5]:  # Limit to avoid too many requests
        try:
            params = {
                "action": "query",
                "format": "json",
                "list": "categorymembers",
                "cmtitle": category,
                "cmlimit": 500,
                "cmtype": "page",
            }

            response = requests.get(api_url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                pages = data.get("query", {}).get("categorymembers", [])

                # Get English links for these pages
                page_ids = [str(p["pageid"]) for p in pages[:100]]

                if page_ids:
                    link_params = {
                        "action": "query",
                        "format": "json",
                        "pageids": "|".join(page_ids),
                        "prop": "langlinks",
                        "lllang": "en",
                        "lllimit": 500,
                    }

                    link_response = requests.get(api_url, params=link_params, timeout=30)
                    if link_response.status_code == 200:
                        link_data = link_response.json()
                        pages_data = link_data.get("query", {}).get("pages", {})

                        for page_id, page_info in pages_data.items():
                            ko_title = page_info.get("title", "")
                            langlinks = page_info.get("langlinks", [])

                            for ll in langlinks:
                                if ll.get("lang") == "en":
                                    en_title = ll.get("*", "")

                                    if ko_title and en_title:
                                        ko_clean = clean_text(ko_title)
                                        en_clean = clean_text(en_title)

                                        if (is_valid_korean(ko_clean) and
                                            is_valid_english(en_clean) and
                                            len(ko_clean) >= 2):
                                            pairs.append({
                                                "ko": ko_clean,
                                                "en": en_clean.lower(),
                                                "source": "wikipedia"
                                            })

            time.sleep(1)  # Rate limiting

        except Exception as e:
            print(f"Category error: {e}")

    # Method 2: Random pages
    print("Fetching random Wikipedia pages...")
    random_params = {
        "action": "query",
        "format": "json",
        "generator": "random",
        "grnnamespace": 0,
        "grnlimit": 50,
        "prop": "langlinks",
        "lllang": "en",
        "lllimit": 1,
    }

    with tqdm(total=target_pairs, desc="Wikipedia Random") as pbar:
        attempts = 0
        max_attempts = target_pairs * 3

        while len(pairs) < target_pairs and attempts < max_attempts:
            try:
                response = requests.get(api_url, params=random_params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    pages = data.get("query", {}).get("pages", {})

                    for page_id, page_data in pages.items():
                        ko_title = page_data.get("title", "")
                        langlinks = page_data.get("langlinks", [])

                        for ll in langlinks:
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
                                            "en": en_clean.lower(),
                                            "source": "wikipedia"
                                        })
                                        pbar.update(1)

                time.sleep(0.5)  # Rate limiting
                attempts += 50

            except Exception as e:
                print(f"Wikipedia error: {e}")
                time.sleep(2)
                attempts += 50

    print(f"Collected {len(pairs):,} pairs from Wikipedia")
    return pairs


# ============================================================
# 4. IT/Tech Terminology (Extended)
# ============================================================

def collect_it_terminology() -> List[Dict]:
    """Collect extensive IT and technical terminology."""
    print("\n" + "=" * 70)
    print("4. COLLECTING IT/TECH TERMINOLOGY")
    print("=" * 70)

    # Extended IT terminology
    it_terms = [
        # Machine Learning / AI
        ("머신러닝", "machine learning"), ("기계학습", "machine learning"),
        ("딥러닝", "deep learning"), ("심층학습", "deep learning"),
        ("인공지능", "artificial intelligence"), ("자연어처리", "natural language processing"),
        ("신경망", "neural network"), ("컴퓨터비전", "computer vision"),
        ("강화학습", "reinforcement learning"), ("지도학습", "supervised learning"),
        ("비지도학습", "unsupervised learning"), ("전이학습", "transfer learning"),
        ("트랜스포머", "transformer"), ("어텐션", "attention"),
        ("임베딩", "embedding"), ("벡터", "vector"), ("텐서", "tensor"),
        ("그래디언트", "gradient"), ("역전파", "backpropagation"),
        ("손실함수", "loss function"), ("최적화", "optimization"),
        ("정규화", "regularization"), ("드롭아웃", "dropout"),
        ("배치", "batch"), ("에폭", "epoch"), ("학습률", "learning rate"),
        ("하이퍼파라미터", "hyperparameter"), ("오버피팅", "overfitting"),
        ("언더피팅", "underfitting"), ("교차검증", "cross validation"),
        ("정확도", "accuracy"), ("정밀도", "precision"), ("재현율", "recall"),
        ("분류", "classification"), ("회귀", "regression"),
        ("클러스터링", "clustering"), ("군집화", "clustering"),
        ("차원축소", "dimensionality reduction"), ("특성추출", "feature extraction"),
        ("데이터증강", "data augmentation"), ("파인튜닝", "fine tuning"),
        ("사전학습", "pretraining"), ("제로샷", "zero shot"),
        ("퓨샷", "few shot"), ("프롬프트", "prompt"),
        ("토크나이저", "tokenizer"), ("토큰화", "tokenization"),
        ("어휘", "vocabulary"), ("컨볼루션", "convolution"),
        ("풀링", "pooling"), ("활성화함수", "activation function"),
        ("소프트맥스", "softmax"), ("시그모이드", "sigmoid"),
        ("렐루", "relu"), ("생성모델", "generative model"),
        ("판별모델", "discriminative model"), ("오토인코더", "autoencoder"),
        ("변분오토인코더", "variational autoencoder"),
        ("생성적적대신경망", "generative adversarial network"),
        ("순환신경망", "recurrent neural network"),
        ("장단기메모리", "long short term memory"),
        ("게이트순환유닛", "gated recurrent unit"),
        ("셀프어텐션", "self attention"), ("멀티헤드어텐션", "multi head attention"),
        ("포지셔널인코딩", "positional encoding"), ("레이어정규화", "layer normalization"),
        ("배치정규화", "batch normalization"), ("잔차연결", "residual connection"),
        ("스킵커넥션", "skip connection"), ("드롭아웃", "dropout"),
        ("가중치감쇠", "weight decay"), ("모멘텀", "momentum"),
        ("아담", "adam"), ("경사하강법", "gradient descent"),
        ("확률적경사하강법", "stochastic gradient descent"),

        # Programming
        ("프로그래밍", "programming"), ("코딩", "coding"),
        ("알고리즘", "algorithm"), ("자료구조", "data structure"),
        ("함수", "function"), ("변수", "variable"), ("클래스", "class"),
        ("객체", "object"), ("메서드", "method"), ("인스턴스", "instance"),
        ("상속", "inheritance"), ("캡슐화", "encapsulation"),
        ("다형성", "polymorphism"), ("인터페이스", "interface"),
        ("추상화", "abstraction"), ("모듈", "module"), ("패키지", "package"),
        ("라이브러리", "library"), ("프레임워크", "framework"),
        ("컴파일러", "compiler"), ("인터프리터", "interpreter"),
        ("디버깅", "debugging"), ("테스트", "test"),
        ("유닛테스트", "unit test"), ("통합테스트", "integration test"),
        ("배포", "deployment"), ("버전관리", "version control"),
        ("리포지토리", "repository"), ("브랜치", "branch"),
        ("커밋", "commit"), ("머지", "merge"), ("풀리퀘스트", "pull request"),
        ("코드리뷰", "code review"), ("리팩토링", "refactoring"),
        ("디자인패턴", "design pattern"), ("싱글톤", "singleton"),
        ("팩토리", "factory"), ("옵저버", "observer"),
        ("데코레이터", "decorator"), ("어댑터", "adapter"),
        ("이터레이터", "iterator"), ("제너레이터", "generator"),
        ("람다", "lambda"), ("클로저", "closure"),
        ("콜백", "callback"), ("프로미스", "promise"),
        ("비동기", "asynchronous"), ("동기", "synchronous"),
        ("스레드", "thread"), ("프로세스", "process"),
        ("동시성", "concurrency"), ("병렬성", "parallelism"),

        # Database
        ("데이터베이스", "database"), ("쿼리", "query"),
        ("테이블", "table"), ("인덱스", "index"), ("키", "key"),
        ("기본키", "primary key"), ("외래키", "foreign key"),
        ("조인", "join"), ("트랜잭션", "transaction"),
        ("스키마", "schema"), ("정규화", "normalization"),
        ("비정규화", "denormalization"), ("샤딩", "sharding"),
        ("복제", "replication"), ("파티셔닝", "partitioning"),
        ("캐싱", "caching"), ("레디스", "redis"),
        ("몽고디비", "mongodb"), ("포스트그레스", "postgresql"),
        ("마이에스큐엘", "mysql"), ("오라클", "oracle"),

        # Web / Network
        ("웹", "web"), ("웹사이트", "website"), ("웹페이지", "webpage"),
        ("서버", "server"), ("클라이언트", "client"),
        ("프론트엔드", "frontend"), ("백엔드", "backend"),
        ("풀스택", "full stack"), ("API", "api"),
        ("REST", "rest"), ("GraphQL", "graphql"),
        ("HTTP", "http"), ("HTTPS", "https"),
        ("URL", "url"), ("도메인", "domain"),
        ("호스팅", "hosting"), ("네트워크", "network"),
        ("프로토콜", "protocol"), ("라우터", "router"),
        ("방화벽", "firewall"), ("로드밸런서", "load balancer"),
        ("프록시", "proxy"), ("캐시", "cache"),
        ("DNS", "dns"), ("TCP", "tcp"), ("UDP", "udp"),
        ("웹소켓", "websocket"), ("쿠키", "cookie"),
        ("세션", "session"), ("토큰", "token"),

        # Cloud
        ("클라우드", "cloud"), ("클라우드컴퓨팅", "cloud computing"),
        ("가상화", "virtualization"), ("가상머신", "virtual machine"),
        ("컨테이너", "container"), ("도커", "docker"),
        ("쿠버네티스", "kubernetes"), ("마이크로서비스", "microservice"),
        ("서버리스", "serverless"), ("스케일링", "scaling"),
        ("오토스케일링", "autoscaling"), ("로드밸런싱", "load balancing"),
        ("인스턴스", "instance"), ("이미지", "image"),
        ("스토리지", "storage"), ("오브젝트스토리지", "object storage"),
        ("블록스토리지", "block storage"),

        # Security
        ("보안", "security"), ("암호화", "encryption"),
        ("복호화", "decryption"), ("해시", "hash"),
        ("인증", "authentication"), ("권한", "authorization"),
        ("인증서", "certificate"), ("취약점", "vulnerability"),
        ("악성코드", "malware"), ("랜섬웨어", "ransomware"),
        ("피싱", "phishing"), ("방화벽", "firewall"),
        ("침입탐지", "intrusion detection"), ("백신", "antivirus"),

        # Search / IR
        ("검색", "search"), ("검색엔진", "search engine"),
        ("정보검색", "information retrieval"), ("인덱싱", "indexing"),
        ("랭킹", "ranking"), ("문서", "document"),
        ("형태소분석", "morphological analysis"), ("불용어", "stopword"),
        ("어간추출", "stemming"), ("표제어추출", "lemmatization"),
        ("시맨틱검색", "semantic search"), ("벡터검색", "vector search"),
        ("유사도", "similarity"), ("코사인유사도", "cosine similarity"),
        ("추천", "recommendation"), ("추천시스템", "recommendation system"),
        ("협업필터링", "collaborative filtering"),
        ("콘텐츠기반필터링", "content based filtering"),
        ("하이브리드필터링", "hybrid filtering"),
        ("임베딩검색", "embedding search"), ("리랭킹", "reranking"),
        ("쿼리확장", "query expansion"), ("문서확장", "document expansion"),

        # Data Science
        ("데이터사이언스", "data science"), ("데이터분석", "data analysis"),
        ("데이터마이닝", "data mining"), ("빅데이터", "big data"),
        ("데이터시각화", "data visualization"), ("대시보드", "dashboard"),
        ("통계", "statistics"), ("확률", "probability"),
        ("분포", "distribution"), ("평균", "mean"),
        ("중앙값", "median"), ("표준편차", "standard deviation"),
        ("분산", "variance"), ("상관관계", "correlation"),
        ("회귀분석", "regression analysis"), ("가설검정", "hypothesis testing"),
        ("신뢰구간", "confidence interval"), ("표본", "sample"),
        ("모집단", "population"), ("편향", "bias"),

        # General Tech Terms
        ("기술", "technology"), ("혁신", "innovation"),
        ("디지털", "digital"), ("스마트", "smart"),
        ("자동화", "automation"), ("효율", "efficiency"),
        ("성능", "performance"), ("품질", "quality"),
        ("안정성", "stability"), ("확장성", "scalability"),
        ("신뢰성", "reliability"), ("가용성", "availability"),
        ("지연시간", "latency"), ("처리량", "throughput"),
        ("대역폭", "bandwidth"), ("용량", "capacity"),
        ("메트릭", "metric"), ("모니터링", "monitoring"),
        ("로깅", "logging"), ("추적", "tracing"),
        ("프로파일링", "profiling"), ("벤치마크", "benchmark"),
        ("부하테스트", "load test"), ("스트레스테스트", "stress test"),
    ]

    pairs = []
    for ko, en in it_terms:
        pairs.append({
            "ko": ko,
            "en": en.lower(),
            "source": "it_terminology"
        })

        # Also add individual English words for multi-word terms
        if ' ' in en:
            for word in extract_english_words(en):
                if len(word) >= 3:
                    pairs.append({
                        "ko": ko,
                        "en": word,
                        "source": "it_terminology"
                    })

    print(f"Collected {len(pairs):,} IT/Tech terms")
    return pairs


# ============================================================
# 5. Quality Filtering and Deduplication
# ============================================================

def filter_and_deduplicate(pairs: List[Dict]) -> List[Dict]:
    """Filter low-quality pairs and deduplicate."""
    print("\n" + "=" * 70)
    print("FILTERING AND DEDUPLICATION")
    print("=" * 70)

    # Quality filtering
    filtered = []
    rejected = defaultdict(int)

    for pair in tqdm(pairs, desc="Filtering"):
        ko = pair["ko"]
        en = pair["en"]

        # Length checks
        if len(ko) < 2:
            rejected["ko_too_short"] += 1
            continue
        if len(en) < 2:
            rejected["en_too_short"] += 1
            continue
        if len(ko) > 30:
            rejected["ko_too_long"] += 1
            continue
        if len(en) > 50:
            rejected["en_too_long"] += 1
            continue

        # Korean validation
        if not is_valid_korean(ko):
            rejected["no_korean"] += 1
            continue

        # English validation
        if not is_valid_english(en):
            rejected["invalid_english"] += 1
            continue

        # Skip if English is just numbers or special chars
        clean_en = re.sub(r'[^a-zA-Z]', '', en)
        if len(clean_en) < 2:
            rejected["en_no_letters"] += 1
            continue

        # Skip very generic single characters
        if len(en) == 1:
            rejected["en_single_char"] += 1
            continue

        filtered.append(pair)

    print(f"Filtered: {len(pairs):,} -> {len(filtered):,}")
    print("Rejection reasons:")
    for reason, count in sorted(rejected.items(), key=lambda x: -x[1])[:10]:
        print(f"  {reason}: {count:,}")

    # Deduplication
    seen = set()
    unique = []

    for pair in tqdm(filtered, desc="Deduplicating"):
        # Normalize for dedup
        key = (pair["ko"].strip(), pair["en"].strip().lower())
        if key not in seen:
            seen.add(key)
            unique.append(pair)

    print(f"After deduplication: {len(unique):,}")
    return unique


# ============================================================
# Main Collection Pipeline
# ============================================================

def main():
    print("=" * 70)
    print("V19 TERM-LEVEL KO-EN DATA COLLECTION")
    print("=" * 70)

    all_pairs = []

    # 1. MUSE Dictionary (highest quality)
    muse_pairs = collect_muse_dictionary()
    all_pairs.extend(muse_pairs)

    # 2. Wikidata (large-scale)
    wikidata_pairs = collect_wikidata_labels()
    all_pairs.extend(wikidata_pairs)

    # 3. IT Terminology (curated)
    it_pairs = collect_it_terminology()
    all_pairs.extend(it_pairs)

    # 4. Wikipedia (optional, skip if already have enough data)
    if len(all_pairs) < 50000:
        wikipedia_pairs = collect_wikipedia_links(target_pairs=10000)
        all_pairs.extend(wikipedia_pairs)

    # Filter and deduplicate
    unique_pairs = filter_and_deduplicate(all_pairs)

    # Statistics
    sources = defaultdict(int)
    for pair in unique_pairs:
        sources[pair["source"]] += 1

    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    print(f"Total unique pairs: {len(unique_pairs):,}")
    print("\nPairs by source:")
    for source, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count:,} ({count/len(unique_pairs)*100:.1f}%)")

    # Save to term_pairs.jsonl
    output_path = OUTPUT_DIR / "term_pairs.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in tqdm(unique_pairs, desc="Saving"):
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"\nSaved to: {output_path}")

    # Save metadata
    metadata = {
        "version": "v19",
        "total_pairs": len(unique_pairs),
        "sources": dict(sources),
        "output_file": str(output_path),
    }

    with open(OUTPUT_DIR / "collection_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("DATA COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Next: Run 00_data_preparation.ipynb to process this data")

    return unique_pairs


if __name__ == "__main__":
    main()
