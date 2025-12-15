#!/usr/bin/env python3
"""
v19 Dataset Creation Script - High Quality Only

Based on v18 learnings:
- EXCLUDE wikidata (caused English activation drop)
- KEEP muse (high quality translation pairs)
- KEEP IT terminology expansion
- KEEP cross-lingual pairs
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "v19_high_quality"


def load_v15_muse_only() -> List[Dict]:
    """Load only MUSE source from v15 (exclude wikidata)."""
    data_path = PROJECT_ROOT / "dataset" / "v15_aggressive" / "term_pairs.jsonl"
    data = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            source = item.get("source", "")

            # Only include high-quality sources (exclude wikidata)
            if source in ["muse", "it_terminology", "core_it_terms",
                          "critical_terms", "decomposition", "tech_nouns",
                          "abbreviations", "common_nouns"]:
                ko = item.get("ko", "")
                en = item.get("en", "")
                if ko and en:
                    data.append({
                        "ko": ko,
                        "en": en.lower(),
                        "source": source
                    })

    print(f"Loaded {len(data):,} high-quality pairs from v15 (excluding wikidata)")
    return data


def load_cross_lingual_pairs() -> List[Dict]:
    """Load cross-lingual term pairs from synonyms directory."""
    data = []

    # cross_lingual_pairs_v2.jsonl
    path = PROJECT_ROOT / "dataset" / "synonyms" / "cross_lingual_pairs_v2.jsonl"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                ko_term = item.get("ko_term", "")
                en_terms = item.get("en_terms", [])
                for en in en_terms:
                    if ko_term and en:
                        data.append({
                            "ko": ko_term,
                            "en": en.lower(),
                            "source": "cross_lingual"
                        })

    # ko_en_terms.jsonl
    path = PROJECT_ROOT / "dataset" / "synonyms" / "ko_en_terms.jsonl"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                ko = item.get("ko_term") or item.get("ko", "")
                en = item.get("en_term") or item.get("en", "")
                if ko and en:
                    data.append({
                        "ko": ko,
                        "en": en.lower(),
                        "source": "ko_en_terms"
                    })

    # abbreviation_pairs.jsonl
    path = PROJECT_ROOT / "dataset" / "synonyms" / "abbreviation_pairs.jsonl"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                ko = item.get("ko", "")
                en = item.get("en", "")
                abbr = item.get("abbreviation", "")
                if ko and en:
                    data.append({"ko": ko, "en": en.lower(), "source": "abbreviations"})
                if ko and abbr:
                    data.append({"ko": ko, "en": abbr.lower(), "source": "abbreviations"})

    print(f"Loaded {len(data):,} pairs from cross-lingual sources")
    return data


def generate_expanded_it_terminology() -> List[Dict]:
    """Generate expanded IT terminology pairs with more variants."""

    it_terms = {
        # Programming
        "프로그래밍": ["programming", "coding", "development"],
        "코딩": ["coding", "programming", "code"],
        "개발": ["development", "dev", "developing"],
        "소프트웨어": ["software", "sw", "application"],
        "하드웨어": ["hardware", "hw", "device"],
        "애플리케이션": ["application", "app", "software"],
        "앱": ["app", "application", "mobile app"],

        # Web
        "웹사이트": ["website", "web site", "site"],
        "웹페이지": ["webpage", "web page", "page"],
        "홈페이지": ["homepage", "home page", "website"],
        "브라우저": ["browser", "web browser"],
        "프론트엔드": ["frontend", "front-end", "front end"],
        "백엔드": ["backend", "back-end", "back end"],
        "풀스택": ["fullstack", "full-stack", "full stack"],

        # Database
        "데이터베이스": ["database", "db", "data store"],
        "쿼리": ["query", "queries", "sql query"],
        "테이블": ["table", "tables", "db table"],
        "인덱스": ["index", "indices", "indexing"],
        "스키마": ["schema", "schemas", "db schema"],

        # Cloud & Infrastructure
        "클라우드": ["cloud", "cloud computing"],
        "서버": ["server", "servers"],
        "클라이언트": ["client", "clients"],
        "호스팅": ["hosting", "host", "web hosting"],
        "배포": ["deployment", "deploy", "deploying"],
        "컨테이너": ["container", "containers", "docker"],
        "쿠버네티스": ["kubernetes", "k8s"],
        "도커": ["docker", "container"],

        # AI/ML
        "인공지능": ["artificial intelligence", "ai"],
        "기계학습": ["machine learning", "ml"],
        "머신러닝": ["machine learning", "ml"],
        "딥러닝": ["deep learning", "dl", "deep neural network"],
        "신경망": ["neural network", "nn", "neural net"],
        "자연어처리": ["natural language processing", "nlp"],
        "컴퓨터비전": ["computer vision", "cv", "vision"],
        "강화학습": ["reinforcement learning", "rl"],
        "트랜스포머": ["transformer", "transformers"],
        "임베딩": ["embedding", "embeddings", "vector embedding"],
        "벡터": ["vector", "vectors"],
        "텐서": ["tensor", "tensors"],

        # Security
        "보안": ["security", "secure", "cybersecurity"],
        "암호화": ["encryption", "encrypt", "cryptography"],
        "복호화": ["decryption", "decrypt"],
        "인증": ["authentication", "auth", "authn"],
        "권한": ["authorization", "authz", "permission"],
        "방화벽": ["firewall", "firewalls"],
        "취약점": ["vulnerability", "vulnerabilities", "security flaw"],

        # DevOps
        "데브옵스": ["devops", "dev ops"],
        "시아이시디": ["cicd", "ci/cd", "continuous integration"],
        "파이프라인": ["pipeline", "pipelines"],
        "모니터링": ["monitoring", "monitor"],
        "로깅": ["logging", "logs", "log"],
        "알림": ["alerting", "alerts", "notification"],

        # Data
        "데이터": ["data", "dataset"],
        "빅데이터": ["big data", "bigdata"],
        "데이터분석": ["data analysis", "analytics"],
        "데이터마이닝": ["data mining", "mining"],
        "데이터웨어하우스": ["data warehouse", "dwh"],

        # Testing
        "테스트": ["test", "testing", "tests"],
        "단위테스트": ["unit test", "unit testing"],
        "통합테스트": ["integration test", "integration testing"],
        "디버깅": ["debugging", "debug", "debugger"],
        "버그": ["bug", "bugs", "defect"],

        # Architecture
        "아키텍처": ["architecture", "arch", "design"],
        "마이크로서비스": ["microservice", "microservices", "micro service"],
        "모놀리식": ["monolithic", "monolith"],

        # Version Control
        "버전관리": ["version control", "vcs", "source control"],
        "깃": ["git", "git vcs"],
        "커밋": ["commit", "commits"],
        "브랜치": ["branch", "branches", "branching"],
        "머지": ["merge", "merging"],
        "풀리퀘스트": ["pull request", "pr", "merge request"],

        # Frameworks & Libraries
        "프레임워크": ["framework", "frameworks"],
        "라이브러리": ["library", "libraries", "lib"],
        "패키지": ["package", "packages", "pkg"],
        "모듈": ["module", "modules"],
        "플러그인": ["plugin", "plugins", "extension"],

        # Performance
        "성능": ["performance", "perf"],
        "최적화": ["optimization", "optimize", "optimizing"],
        "캐싱": ["caching", "cache"],
        "스케일링": ["scaling", "scale", "scalability"],
        "로드밸런싱": ["load balancing", "load balancer"],
        "지연시간": ["latency", "delay", "lag"],
        "처리량": ["throughput", "throughputs"],

        # Search
        "검색": ["search", "searching", "retrieval"],
        "검색엔진": ["search engine", "search engines"],
        "인덱싱": ["indexing", "index", "indices"],
        "랭킹": ["ranking", "rank", "relevance"],
        "추천": ["recommendation", "recommend", "recommender"],
        "필터링": ["filtering", "filter", "filters"],

        # Networking
        "네트워크": ["network", "networking", "networks"],
        "프로토콜": ["protocol", "protocols"],
        "도메인": ["domain", "domains", "domain name"],
        "프록시": ["proxy", "proxies"],
        "게이트웨이": ["gateway", "gateways"],

        # Storage
        "스토리지": ["storage", "store"],
        "파일시스템": ["file system", "filesystem", "fs"],
        "백업": ["backup", "backups"],
        "복구": ["recovery", "restore", "restoration"],

        # Operating System
        "운영체제": ["operating system", "os"],
        "리눅스": ["linux", "linux os"],
        "윈도우": ["windows", "windows os"],
        "커널": ["kernel", "kernels"],
        "프로세스": ["process", "processes"],
        "스레드": ["thread", "threads", "threading"],
        "메모리": ["memory", "ram", "mem"],

        # Algorithms
        "알고리즘": ["algorithm", "algorithms", "algo"],
        "자료구조": ["data structure", "data structures"],
        "정렬": ["sorting", "sort"],
        "탐색": ["search", "searching"],
        "그래프": ["graph", "graphs"],
        "트리": ["tree", "trees"],
        "해시": ["hash", "hashing", "hashtable"],

        # Development terms
        "리팩토링": ["refactoring", "refactor"],
        "코드리뷰": ["code review", "review"],
        "문서화": ["documentation", "docs", "document"],
        "변수": ["variable", "variables", "var"],
        "함수": ["function", "functions", "func"],
        "클래스": ["class", "classes"],
        "객체": ["object", "objects", "obj"],
        "인스턴스": ["instance", "instances"],
        "상속": ["inheritance", "inherit"],
        "인터페이스": ["interface", "interfaces"],

        # Common Terms
        "설정": ["configuration", "config", "settings"],
        "환경": ["environment", "env"],
        "옵션": ["option", "options"],
        "파라미터": ["parameter", "parameters", "param"],
        "반환": ["return", "returns"],
        "입력": ["input", "inputs"],
        "출력": ["output", "outputs"],
        "예외": ["exception", "exceptions", "error"],
        "오류": ["error", "errors", "fault"],
        "로그": ["log", "logs", "logging"],
    }

    data = []
    for ko, en_list in it_terms.items():
        for en in en_list:
            data.append({
                "ko": ko,
                "en": en.lower(),
                "source": "it_expansion"
            })

    print(f"Generated {len(data):,} IT terminology pairs")
    return data


def generate_common_terms() -> List[Dict]:
    """Generate common Korean-English term pairs."""

    common_terms = {
        # Business
        "회사": ["company", "corporation", "business"],
        "기업": ["enterprise", "company", "corporation"],
        "고객": ["customer", "client", "user"],
        "사용자": ["user", "users", "end user"],
        "관리자": ["administrator", "admin", "manager"],
        "팀": ["team", "teams"],
        "프로젝트": ["project", "projects"],

        # Actions
        "생성": ["create", "creation", "generate"],
        "수정": ["edit", "modify", "update"],
        "삭제": ["delete", "remove", "deletion"],
        "조회": ["view", "query", "lookup"],
        "저장": ["save", "store", "saving"],
        "실행": ["execute", "run", "execution"],
        "시작": ["start", "begin", "starting"],
        "종료": ["end", "finish", "terminate"],
        "완료": ["complete", "completion", "done"],
        "취소": ["cancel", "cancellation"],
        "확인": ["confirm", "confirmation", "check"],

        # States
        "활성": ["active", "enabled"],
        "비활성": ["inactive", "disabled"],
        "대기": ["pending", "waiting"],
        "진행": ["progress", "in progress", "processing"],
        "성공": ["success", "successful"],
        "실패": ["failure", "failed", "fail"],

        # UI Elements
        "버튼": ["button", "buttons", "btn"],
        "메뉴": ["menu", "menus"],
        "창": ["window", "windows"],
        "팝업": ["popup", "pop-up", "modal"],
        "알림": ["notification", "alert", "notice"],
        "메시지": ["message", "messages", "msg"],
        "이미지": ["image", "images", "picture"],
        "텍스트": ["text", "texts"],
        "링크": ["link", "links", "hyperlink"],

        # Documents
        "문서": ["document", "documents", "doc"],
        "파일": ["file", "files"],
        "폴더": ["folder", "folders", "directory"],
        "목록": ["list", "lists"],

        # Analysis
        "분석": ["analysis", "analytics", "analyze"],
        "통계": ["statistics", "stats", "statistical"],
        "결과": ["result", "results", "outcome"],
        "보고서": ["report", "reports"],
    }

    data = []
    for ko, en_list in common_terms.items():
        for en in en_list:
            data.append({
                "ko": ko,
                "en": en.lower(),
                "source": "common_expansion"
            })

    print(f"Generated {len(data):,} common term pairs")
    return data


def deduplicate_data(data: List[Dict]) -> List[Dict]:
    """Remove duplicate (ko, en) pairs."""
    seen = set()
    unique_data = []

    for item in data:
        key = (item["ko"], item["en"])
        if key not in seen:
            seen.add(key)
            unique_data.append(item)

    print(f"Deduplicated: {len(data):,} -> {len(unique_data):,} pairs")
    return unique_data


def filter_quality(data: List[Dict]) -> List[Dict]:
    """Filter out low-quality pairs."""
    filtered = []

    for item in data:
        ko = item["ko"]
        en = item["en"]

        # Skip empty
        if not ko or not en:
            continue

        # Skip very short terms
        if len(ko) < 1 or len(en) < 2:
            continue

        # Skip if Korean contains only English characters
        if all(c.isascii() for c in ko):
            continue

        # Skip if English contains Korean characters
        if any('\uac00' <= c <= '\ud7a3' for c in en):
            continue

        # Skip very long terms (likely sentences)
        if len(ko) > 20 or len(en) > 50:
            continue

        filtered.append(item)

    print(f"Quality filtered: {len(data):,} -> {len(filtered):,} pairs")
    return filtered


def main():
    print("=" * 70)
    print("Creating v19 High-Quality Dataset")
    print("=" * 70)
    print("\nStrategy: Exclude wikidata, keep only high-quality sources")

    # Collect all data
    all_data = []

    # 1. Load v15 MUSE data only (exclude wikidata)
    all_data.extend(load_v15_muse_only())

    # 2. Load cross-lingual pairs
    all_data.extend(load_cross_lingual_pairs())

    # 3. Generate IT terminology expansion
    all_data.extend(generate_expanded_it_terminology())

    # 4. Generate common terms
    all_data.extend(generate_common_terms())

    print(f"\nTotal collected: {len(all_data):,} pairs")

    # Deduplicate
    all_data = deduplicate_data(all_data)

    # Quality filter
    all_data = filter_quality(all_data)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save dataset
    output_path = OUTPUT_DIR / "term_pairs.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 70}")
    print(f"Dataset saved to: {output_path}")
    print(f"Total pairs: {len(all_data):,}")

    # Print source distribution
    print(f"\n{'=' * 70}")
    print("Source Distribution:")
    source_counts = defaultdict(int)
    for item in all_data:
        source_counts[item["source"]] += 1

    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count:,}")

    # Print sample data
    print(f"\n{'=' * 70}")
    print("Sample Data:")
    import random
    samples = random.sample(all_data, min(10, len(all_data)))
    for s in samples:
        print(f"  {s['ko']} -> {s['en']} ({s['source']})")


if __name__ == "__main__":
    main()
