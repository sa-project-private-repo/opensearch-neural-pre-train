#!/usr/bin/env python3
"""
Collect term-level KO-EN parallel data - Version 2 (Fixed)

Sources:
1. MUSE bilingual dictionary (Direct download)
2. Wikidata labels (SPARQL with smaller queries)
3. Wikipedia titles (from dumps)
4. IT/Tech terminology (Curated)

Target: 500K+ high-quality term pairs
"""

import json
import os
import sys
import subprocess
from pathlib import Path
from collections import defaultdict
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "v11_term_pairs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_valid_korean(text: str) -> bool:
    return any('\uac00' <= c <= '\ud7a3' for c in text)


def is_valid_english(text: str) -> bool:
    return any(c.isalpha() and c.isascii() for c in text)


def clean_text(text: str) -> str:
    return text.strip()


# ============================================================
# 1. MUSE Dictionary (Fixed)
# ============================================================

def collect_muse_dictionary() -> list:
    """Download and parse MUSE bilingual dictionaries."""
    print("\n" + "="*70)
    print("1. COLLECTING MUSE DICTIONARY")
    print("="*70)

    pairs = []

    # Download ko-en
    print("Downloading ko-en dictionary...")
    try:
        response = requests.get(
            "https://dl.fbaipublicfiles.com/arrival/dictionaries/ko-en.txt",
            timeout=60
        )
        if response.status_code == 200:
            # Fix: Use content.decode('utf-8') instead of text
            content = response.content.decode('utf-8')
            for line in content.strip().split('\n'):
                parts = line.split('\t')
                if len(parts) >= 2:
                    ko = parts[0].strip()
                    en = parts[1].strip()
                    if ko and en and is_valid_korean(ko) and is_valid_english(en):
                        pairs.append({"ko": ko, "en": en, "source": "muse"})
            print(f"  ko-en: {len(pairs):,} pairs")
    except Exception as e:
        print(f"  ko-en error: {e}")

    # Download en-ko (reverse)
    print("Downloading en-ko dictionary...")
    count_before = len(pairs)
    try:
        response = requests.get(
            "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-ko.txt",
            timeout=60
        )
        if response.status_code == 200:
            # Fix: Use content.decode('utf-8') instead of text
            content = response.content.decode('utf-8')
            for line in content.strip().split('\n'):
                parts = line.split('\t')
                if len(parts) >= 2:
                    en = parts[0].strip()
                    ko = parts[1].strip()
                    if ko and en and is_valid_korean(ko) and is_valid_english(en):
                        pairs.append({"ko": ko, "en": en, "source": "muse"})
            print(f"  en-ko: {len(pairs) - count_before:,} pairs")
    except Exception as e:
        print(f"  en-ko error: {e}")

    print(f"Total MUSE: {len(pairs):,} pairs")
    return pairs


# ============================================================
# 2. Wikidata Labels (Fixed - smaller batched queries)
# ============================================================

def collect_wikidata_labels() -> list:
    """Collect from Wikidata using smaller batched queries."""
    print("\n" + "="*70)
    print("2. COLLECTING WIKIDATA LABELS")
    print("="*70)

    pairs = []
    sparql_url = "https://query.wikidata.org/sparql"

    # Use multiple smaller queries for different entity types
    queries = [
        # People
        """
        SELECT ?koLabel ?enLabel WHERE {
            ?item wdt:P31 wd:Q5 .
            ?item rdfs:label ?koLabel . FILTER(LANG(?koLabel) = "ko")
            ?item rdfs:label ?enLabel . FILTER(LANG(?enLabel) = "en")
        } LIMIT 20000
        """,
        # Countries
        """
        SELECT ?koLabel ?enLabel WHERE {
            ?item wdt:P31 wd:Q6256 .
            ?item rdfs:label ?koLabel . FILTER(LANG(?koLabel) = "ko")
            ?item rdfs:label ?enLabel . FILTER(LANG(?enLabel) = "en")
        } LIMIT 5000
        """,
        # Cities
        """
        SELECT ?koLabel ?enLabel WHERE {
            ?item wdt:P31 wd:Q515 .
            ?item rdfs:label ?koLabel . FILTER(LANG(?koLabel) = "ko")
            ?item rdfs:label ?enLabel . FILTER(LANG(?enLabel) = "en")
        } LIMIT 20000
        """,
        # Companies
        """
        SELECT ?koLabel ?enLabel WHERE {
            ?item wdt:P31 wd:Q4830453 .
            ?item rdfs:label ?koLabel . FILTER(LANG(?koLabel) = "ko")
            ?item rdfs:label ?enLabel . FILTER(LANG(?enLabel) = "en")
        } LIMIT 20000
        """,
        # Scientific concepts
        """
        SELECT ?koLabel ?enLabel WHERE {
            ?item wdt:P31 wd:Q35120 .
            ?item rdfs:label ?koLabel . FILTER(LANG(?koLabel) = "ko")
            ?item rdfs:label ?enLabel . FILTER(LANG(?enLabel) = "en")
        } LIMIT 20000
        """,
        # Software
        """
        SELECT ?koLabel ?enLabel WHERE {
            ?item wdt:P31 wd:Q7397 .
            ?item rdfs:label ?koLabel . FILTER(LANG(?koLabel) = "ko")
            ?item rdfs:label ?enLabel . FILTER(LANG(?enLabel) = "en")
        } LIMIT 10000
        """,
        # Films
        """
        SELECT ?koLabel ?enLabel WHERE {
            ?item wdt:P31 wd:Q11424 .
            ?item rdfs:label ?koLabel . FILTER(LANG(?koLabel) = "ko")
            ?item rdfs:label ?enLabel . FILTER(LANG(?enLabel) = "en")
        } LIMIT 30000
        """,
        # Books
        """
        SELECT ?koLabel ?enLabel WHERE {
            ?item wdt:P31 wd:Q571 .
            ?item rdfs:label ?koLabel . FILTER(LANG(?koLabel) = "ko")
            ?item rdfs:label ?enLabel . FILTER(LANG(?enLabel) = "en")
        } LIMIT 20000
        """,
        # Universities
        """
        SELECT ?koLabel ?enLabel WHERE {
            ?item wdt:P31 wd:Q3918 .
            ?item rdfs:label ?koLabel . FILTER(LANG(?koLabel) = "ko")
            ?item rdfs:label ?enLabel . FILTER(LANG(?enLabel) = "en")
        } LIMIT 10000
        """,
        # Diseases
        """
        SELECT ?koLabel ?enLabel WHERE {
            ?item wdt:P31 wd:Q12136 .
            ?item rdfs:label ?koLabel . FILTER(LANG(?koLabel) = "ko")
            ?item rdfs:label ?enLabel . FILTER(LANG(?enLabel) = "en")
        } LIMIT 10000
        """,
    ]

    for i, query in enumerate(queries):
        print(f"  Query {i+1}/{len(queries)}...")
        try:
            response = requests.get(
                sparql_url,
                params={"query": query, "format": "json"},
                headers={"User-Agent": "TermCollector/1.0"},
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", {}).get("bindings", [])

                for item in results:
                    ko = item.get("koLabel", {}).get("value", "")
                    en = item.get("enLabel", {}).get("value", "")

                    if ko and en and is_valid_korean(ko) and is_valid_english(en):
                        pairs.append({"ko": clean_text(ko), "en": clean_text(en), "source": "wikidata"})

                print(f"    Found {len(results):,} items")
            else:
                print(f"    Query failed: {response.status_code}")

        except Exception as e:
            print(f"    Error: {e}")

    print(f"Total Wikidata: {len(pairs):,} pairs")
    return pairs


# ============================================================
# 3. Wikipedia Titles (from API)
# ============================================================

def collect_wikipedia_titles() -> list:
    """Collect Wikipedia article titles with their English equivalents."""
    print("\n" + "="*70)
    print("3. COLLECTING WIKIPEDIA TITLES")
    print("="*70)

    import time

    pairs = []
    api_url = "https://ko.wikipedia.org/w/api.php"

    # Required User-Agent for Wikipedia API
    headers = {
        "User-Agent": "KoEnTermCollector/1.0 (Neural Sparse Training Data)"
    }

    # Use allpages generator to get more systematic coverage
    params = {
        "action": "query",
        "format": "json",
        "generator": "allpages",
        "gapnamespace": 0,
        "gaplimit": 50,
        "prop": "langlinks",
        "lllang": "en",
        "lllimit": 1,
    }

    target = 50000
    continue_param = None
    error_count = 0
    max_errors = 5

    with tqdm(total=target, desc="Wikipedia") as pbar:
        while len(pairs) < target and error_count < max_errors:
            try:
                if continue_param:
                    params["gapcontinue"] = continue_param

                response = requests.get(api_url, params=params, headers=headers, timeout=30)

                if response.status_code != 200:
                    error_count += 1
                    time.sleep(1)
                    continue

                data = response.json()

                if "query" not in data or "pages" not in data["query"]:
                    break

                for page_id, page_data in data["query"]["pages"].items():
                    ko_title = page_data.get("title", "")

                    if "langlinks" in page_data:
                        for ll in page_data["langlinks"]:
                            if ll.get("lang") == "en":
                                en_title = ll.get("*", "")

                                if ko_title and en_title:
                                    if is_valid_korean(ko_title) and is_valid_english(en_title):
                                        pairs.append({
                                            "ko": clean_text(ko_title),
                                            "en": clean_text(en_title),
                                            "source": "wikipedia"
                                        })
                                        pbar.update(1)

                # Get continue parameter
                if "continue" in data and "gapcontinue" in data["continue"]:
                    continue_param = data["continue"]["gapcontinue"]
                else:
                    break

                # Rate limiting
                time.sleep(0.1)
                error_count = 0  # Reset on success

            except Exception as e:
                error_count += 1
                print(f"Error ({error_count}/{max_errors}): {e}")
                time.sleep(2)

            if len(pairs) >= target:
                break

    print(f"Total Wikipedia: {len(pairs):,} pairs")
    return pairs


# ============================================================
# 4. IT/Tech Terminology (Extended)
# ============================================================

def collect_it_terminology() -> list:
    """Extended IT and technical terminology."""
    print("\n" + "="*70)
    print("4. COLLECTING IT/TECH TERMINOLOGY")
    print("="*70)

    # Extended list
    it_terms = [
        # Machine Learning / AI
        ("머신러닝", "machine learning"), ("기계학습", "machine learning"),
        ("딥러닝", "deep learning"), ("심층학습", "deep learning"),
        ("인공지능", "artificial intelligence"), ("AI", "AI"),
        ("자연어처리", "natural language processing"), ("NLP", "NLP"),
        ("신경망", "neural network"), ("뉴럴네트워크", "neural network"),
        ("컴퓨터비전", "computer vision"), ("영상처리", "image processing"),
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
        ("사전학습", "pretraining"), ("언어모델", "language model"),
        ("생성모델", "generative model"), ("판별모델", "discriminative model"),
        ("GAN", "GAN"), ("VAE", "VAE"), ("오토인코더", "autoencoder"),
        ("CNN", "CNN"), ("RNN", "RNN"), ("LSTM", "LSTM"), ("GRU", "GRU"),
        ("BERT", "BERT"), ("GPT", "GPT"), ("T5", "T5"),

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
        ("깃", "git"), ("리포지토리", "repository"), ("브랜치", "branch"),
        ("커밋", "commit"), ("머지", "merge"), ("풀리퀘스트", "pull request"),
        ("파이썬", "python"), ("자바", "java"), ("자바스크립트", "javascript"),
        ("타입스크립트", "typescript"), ("씨플플", "c++"), ("고랭", "golang"),
        ("러스트", "rust"), ("스위프트", "swift"), ("코틀린", "kotlin"),

        # Database
        ("데이터베이스", "database"), ("데이터", "data"), ("쿼리", "query"),
        ("테이블", "table"), ("인덱스", "index"), ("키", "key"),
        ("기본키", "primary key"), ("외래키", "foreign key"),
        ("조인", "join"), ("트랜잭션", "transaction"), ("스키마", "schema"),
        ("SQL", "SQL"), ("NoSQL", "NoSQL"), ("몽고디비", "MongoDB"),
        ("포스트그레스", "PostgreSQL"), ("마이에스큐엘", "MySQL"),
        ("레디스", "Redis"), ("엘라스틱서치", "Elasticsearch"),
        ("오픈서치", "OpenSearch"),

        # Web / Network
        ("웹", "web"), ("웹사이트", "website"), ("웹페이지", "webpage"),
        ("웹서버", "web server"), ("서버", "server"), ("클라이언트", "client"),
        ("프론트엔드", "frontend"), ("백엔드", "backend"),
        ("풀스택", "full stack"), ("API", "API"), ("REST", "REST"),
        ("GraphQL", "GraphQL"), ("HTTP", "HTTP"), ("HTTPS", "HTTPS"),
        ("URL", "URL"), ("도메인", "domain"), ("호스팅", "hosting"),
        ("네트워크", "network"), ("프로토콜", "protocol"),
        ("라우터", "router"), ("스위치", "switch"), ("방화벽", "firewall"),
        ("로드밸런서", "load balancer"), ("프록시", "proxy"),
        ("캐시", "cache"), ("CDN", "CDN"), ("DNS", "DNS"),

        # Cloud
        ("클라우드", "cloud"), ("클라우드컴퓨팅", "cloud computing"),
        ("가상화", "virtualization"), ("가상머신", "virtual machine"),
        ("컨테이너", "container"), ("도커", "docker"),
        ("쿠버네티스", "kubernetes"), ("마이크로서비스", "microservice"),
        ("서버리스", "serverless"), ("스케일링", "scaling"),
        ("오토스케일링", "autoscaling"), ("AWS", "AWS"),
        ("애저", "Azure"), ("GCP", "GCP"),

        # Search / IR
        ("검색", "search"), ("검색엔진", "search engine"),
        ("정보검색", "information retrieval"), ("인덱싱", "indexing"),
        ("랭킹", "ranking"), ("문서", "document"), ("토큰", "token"),
        ("토크나이저", "tokenizer"), ("형태소분석", "morphological analysis"),
        ("불용어", "stopword"), ("어간추출", "stemming"),
        ("표제어추출", "lemmatization"), ("시맨틱검색", "semantic search"),
        ("벡터검색", "vector search"), ("유사도", "similarity"),
        ("코사인유사도", "cosine similarity"), ("추천", "recommendation"),
        ("추천시스템", "recommendation system"), ("협업필터링", "collaborative filtering"),
        ("콘텐츠기반필터링", "content-based filtering"),
        ("희소표현", "sparse representation"), ("밀집표현", "dense representation"),

        # Data Science
        ("데이터사이언스", "data science"), ("데이터분석", "data analysis"),
        ("데이터마이닝", "data mining"), ("빅데이터", "big data"),
        ("데이터시각화", "data visualization"), ("대시보드", "dashboard"),
        ("통계", "statistics"), ("확률", "probability"),
        ("분포", "distribution"), ("평균", "mean"), ("중앙값", "median"),
        ("표준편차", "standard deviation"), ("분산", "variance"),
        ("상관관계", "correlation"), ("회귀분석", "regression analysis"),

        # Security
        ("보안", "security"), ("사이버보안", "cybersecurity"),
        ("암호화", "encryption"), ("복호화", "decryption"),
        ("해시", "hash"), ("인증", "authentication"), ("권한", "authorization"),
        ("토큰", "token"), ("세션", "session"), ("쿠키", "cookie"),
        ("SSL", "SSL"), ("TLS", "TLS"), ("인증서", "certificate"),
        ("취약점", "vulnerability"), ("악성코드", "malware"),

        # Hardware
        ("하드웨어", "hardware"), ("소프트웨어", "software"),
        ("펌웨어", "firmware"), ("컴퓨터", "computer"),
        ("프로세서", "processor"), ("CPU", "CPU"), ("GPU", "GPU"),
        ("메모리", "memory"), ("RAM", "RAM"), ("저장장치", "storage"),
        ("SSD", "SSD"), ("HDD", "HDD"),

        # General Tech
        ("기술", "technology"), ("시스템", "system"), ("플랫폼", "platform"),
        ("서비스", "service"), ("애플리케이션", "application"),
        ("앱", "app"), ("소프트웨어", "software"), ("하드웨어", "hardware"),
        ("인터페이스", "interface"), ("사용자경험", "user experience"),
        ("UX", "UX"), ("UI", "UI"), ("디자인", "design"),
        ("모델", "model"), ("아키텍처", "architecture"),
        ("설계", "design"), ("개발", "development"),
        ("구현", "implementation"), ("운영", "operation"),
        ("유지보수", "maintenance"), ("업데이트", "update"),
        ("업그레이드", "upgrade"), ("마이그레이션", "migration"),
        ("통합", "integration"), ("확장", "extension"),
        ("확장성", "scalability"), ("안정성", "stability"),
        ("신뢰성", "reliability"), ("가용성", "availability"),
        ("성능", "performance"), ("효율", "efficiency"),
        ("자동화", "automation"), ("워크플로우", "workflow"),

        # Business
        ("비즈니스", "business"), ("스타트업", "startup"),
        ("사용자", "user"), ("고객", "customer"),
        ("프로젝트", "project"), ("관리", "management"),
        ("분석", "analysis"), ("전략", "strategy"),
        ("마케팅", "marketing"), ("영업", "sales"),
        ("재무", "finance"), ("회계", "accounting"),
        ("투자", "investment"), ("수익", "revenue"),
        ("이익", "profit"), ("비용", "cost"),
    ]

    pairs = []
    for ko, en in it_terms:
        pairs.append({"ko": ko, "en": en, "source": "it_terminology"})

    print(f"Total IT terms: {len(pairs):,}")
    return pairs


# ============================================================
# Main Pipeline
# ============================================================

def main():
    print("="*70)
    print("TERM-LEVEL KO-EN DATA COLLECTION v2")
    print("="*70)

    all_pairs = []

    # 1. MUSE
    muse_pairs = collect_muse_dictionary()
    all_pairs.extend(muse_pairs)

    # 2. Wikidata
    wikidata_pairs = collect_wikidata_labels()
    all_pairs.extend(wikidata_pairs)

    # 3. Wikipedia
    wikipedia_pairs = collect_wikipedia_titles()
    all_pairs.extend(wikipedia_pairs)

    # 4. IT Terminology
    it_pairs = collect_it_terminology()
    all_pairs.extend(it_pairs)

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

    print(f"Before: {len(all_pairs):,}")
    print(f"After: {len(unique_pairs):,}")

    # Save
    print("\n" + "="*70)
    print("SAVING")
    print("="*70)

    output_path = OUTPUT_DIR / "term_pairs.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in unique_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"Saved to: {output_path}")

    # Stats
    sources = defaultdict(int)
    for pair in unique_pairs:
        sources[pair["source"]] += 1

    print("\nBy source:")
    for source, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count:,}")

    # Metadata
    metadata = {
        "total_pairs": len(unique_pairs),
        "sources": dict(sources),
    }
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTotal: {len(unique_pairs):,} unique term pairs")

    return unique_pairs


if __name__ == "__main__":
    main()
