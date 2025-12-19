#!/usr/bin/env python3
"""
IT/기술 용어 한영 사전 수집 및 생성 스크립트.

다양한 IT 분야의 용어를 수집하여 term_mappings.jsonl 형식으로 저장.
"""

import json
from pathlib import Path
from typing import Dict, List

# IT 용어 사전 (한국어 -> [영어 번역들])
IT_TERMS: Dict[str, List[str]] = {
    # ==========================================================================
    # 프로그래밍 기초
    # ==========================================================================
    "변수": ["variable", "variables", "var"],
    "상수": ["constant", "constants", "const"],
    "함수": ["function", "functions", "func", "method"],
    "메서드": ["method", "methods", "function"],
    "클래스": ["class", "classes"],
    "객체": ["object", "objects", "instance"],
    "인스턴스": ["instance", "instances", "object"],
    "배열": ["array", "arrays", "list"],
    "리스트": ["list", "lists", "array"],
    "딕셔너리": ["dictionary", "dict", "map", "hashmap"],
    "해시맵": ["hashmap", "hash map", "dictionary", "map"],
    "튜플": ["tuple", "tuples"],
    "집합": ["set", "sets"],
    "문자열": ["string", "strings", "str", "text"],
    "정수": ["integer", "integers", "int", "number"],
    "실수": ["float", "floating point", "double", "decimal"],
    "불리언": ["boolean", "bool", "true false"],
    "타입": ["type", "types", "data type"],
    "자료형": ["data type", "type", "datatype"],
    "반복문": ["loop", "loops", "iteration", "for loop", "while loop"],
    "조건문": ["conditional", "if statement", "condition", "branch"],
    "예외처리": ["exception handling", "try catch", "error handling"],
    "상속": ["inheritance", "inherit", "extends"],
    "다형성": ["polymorphism", "polymorphic"],
    "캡슐화": ["encapsulation", "encapsulate"],
    "추상화": ["abstraction", "abstract"],
    "인터페이스": ["interface", "interfaces"],
    "모듈": ["module", "modules"],
    "패키지": ["package", "packages"],
    "라이브러리": ["library", "libraries", "lib"],
    "프레임워크": ["framework", "frameworks"],
    "런타임": ["runtime", "run time", "execution"],
    "컴파일": ["compile", "compilation", "compiler"],
    "컴파일러": ["compiler", "compilers"],
    "인터프리터": ["interpreter", "interpreters"],
    "디버그": ["debug", "debugging", "debugger"],
    "디버깅": ["debugging", "debug", "troubleshooting"],
    "버그": ["bug", "bugs", "defect", "error"],
    "오류": ["error", "errors", "bug", "exception"],
    "에러": ["error", "errors", "exception"],
    "스택": ["stack", "stacks", "call stack"],
    "힙": ["heap", "heaps", "memory heap"],
    "포인터": ["pointer", "pointers", "reference"],
    "참조": ["reference", "references", "pointer"],
    "재귀": ["recursion", "recursive"],
    "콜백": ["callback", "callbacks", "callback function"],
    "비동기": ["async", "asynchronous", "async await"],
    "동기": ["sync", "synchronous"],
    "스레드": ["thread", "threads", "threading"],
    "멀티스레드": ["multithread", "multithreading", "multi-thread"],
    "프로세스": ["process", "processes"],
    "병렬처리": ["parallel processing", "parallelism", "parallel"],
    "동시성": ["concurrency", "concurrent"],

    # ==========================================================================
    # 웹 개발
    # ==========================================================================
    "프론트엔드": ["frontend", "front-end", "front end", "client side"],
    "백엔드": ["backend", "back-end", "back end", "server side"],
    "풀스택": ["fullstack", "full-stack", "full stack"],
    "웹서버": ["web server", "webserver", "http server"],
    "웹사이트": ["website", "web site", "site"],
    "웹앱": ["web app", "web application", "webapp"],
    "웹페이지": ["web page", "webpage", "html page"],
    "브라우저": ["browser", "browsers", "web browser"],
    "렌더링": ["rendering", "render"],
    "반응형": ["responsive", "responsive design", "responsive web"],
    "스타일시트": ["stylesheet", "style sheet", "css"],
    "마크업": ["markup", "html markup"],
    "템플릿": ["template", "templates"],
    "컴포넌트": ["component", "components"],
    "라우팅": ["routing", "router", "routes"],
    "라우터": ["router", "routers", "routing"],
    "미들웨어": ["middleware", "middlewares"],
    "세션": ["session", "sessions"],
    "쿠키": ["cookie", "cookies"],
    "토큰": ["token", "tokens"],
    "인증": ["authentication", "auth", "authn"],
    "인가": ["authorization", "authz", "permission"],
    "로그인": ["login", "log in", "sign in"],
    "로그아웃": ["logout", "log out", "sign out"],
    "회원가입": ["registration", "sign up", "register"],

    # ==========================================================================
    # API & 네트워크
    # ==========================================================================
    "에이피아이": ["api", "application programming interface"],
    "레스트": ["rest", "restful", "rest api"],
    "그래프큐엘": ["graphql", "graph ql"],
    "엔드포인트": ["endpoint", "endpoints", "api endpoint"],
    "요청": ["request", "requests", "http request"],
    "응답": ["response", "responses", "http response"],
    "헤더": ["header", "headers", "http header"],
    "바디": ["body", "request body", "response body"],
    "파라미터": ["parameter", "parameters", "params"],
    "쿼리": ["query", "queries"],
    "페이로드": ["payload", "payloads", "data payload"],
    "프로토콜": ["protocol", "protocols"],
    "에이치티티피": ["http", "hypertext transfer protocol"],
    "에이치티티피에스": ["https", "http secure", "ssl"],
    "웹소켓": ["websocket", "websockets", "ws"],
    "소켓": ["socket", "sockets", "network socket"],
    "포트": ["port", "ports", "network port"],
    "아이피": ["ip", "ip address", "internet protocol"],
    "도메인": ["domain", "domains", "domain name"],
    "디엔에스": ["dns", "domain name system"],
    "로드밸런서": ["load balancer", "load balancing", "lb"],
    "프록시": ["proxy", "proxies", "proxy server"],
    "리버스프록시": ["reverse proxy", "nginx", "reverse-proxy"],
    "게이트웨이": ["gateway", "gateways", "api gateway"],
    "방화벽": ["firewall", "firewalls"],
    "대역폭": ["bandwidth", "network bandwidth"],
    "레이턴시": ["latency", "delay", "response time"],
    "처리량": ["throughput", "processing capacity"],

    # ==========================================================================
    # 데이터베이스
    # ==========================================================================
    "데이터베이스": ["database", "databases", "db"],
    "테이블": ["table", "tables", "database table"],
    "컬럼": ["column", "columns", "field"],
    "로우": ["row", "rows", "record"],
    "레코드": ["record", "records", "row"],
    "스키마": ["schema", "schemas", "database schema"],
    "인덱스": ["index", "indexes", "indices", "db index"],
    "인덱싱": ["indexing", "index creation"],
    "기본키": ["primary key", "pk", "primary"],
    "외래키": ["foreign key", "fk", "foreign"],
    "조인": ["join", "joins", "sql join"],
    "트랜잭션": ["transaction", "transactions", "tx"],
    "커밋": ["commit", "commits", "db commit"],
    "롤백": ["rollback", "rollbacks", "db rollback"],
    "정규화": ["normalization", "normalize", "db normalization"],
    "비정규화": ["denormalization", "denormalize"],
    "샤딩": ["sharding", "shards", "db sharding"],
    "레플리카": ["replica", "replicas", "replication"],
    "복제": ["replication", "replicate", "db replication"],
    "백업": ["backup", "backups", "db backup"],
    "복구": ["recovery", "restore", "db recovery"],
    "마이그레이션": ["migration", "migrations", "db migration"],
    "에스큐엘": ["sql", "structured query language"],
    "노에스큐엘": ["nosql", "no sql", "non-relational"],
    "관계형": ["relational", "rdbms", "relational database"],
    "문서형": ["document", "document database", "mongodb"],
    "키밸류": ["key-value", "key value", "redis"],
    "그래프디비": ["graph database", "graph db", "neo4j"],
    "캐시": ["cache", "caching", "cached"],
    "캐싱": ["caching", "cache", "cached data"],
    "레디스": ["redis", "redis cache"],
    "멤캐시드": ["memcached", "memcache"],

    # ==========================================================================
    # 클라우드 & 인프라
    # ==========================================================================
    "클라우드": ["cloud", "cloud computing"],
    "서버": ["server", "servers"],
    "클라이언트": ["client", "clients"],
    "호스팅": ["hosting", "host", "web hosting"],
    "가상화": ["virtualization", "virtual", "vm"],
    "가상머신": ["virtual machine", "vm", "virtual server"],
    "컨테이너": ["container", "containers", "docker container"],
    "도커": ["docker", "docker container"],
    "쿠버네티스": ["kubernetes", "k8s", "container orchestration"],
    "오케스트레이션": ["orchestration", "orchestrator"],
    "마이크로서비스": ["microservice", "microservices", "micro service"],
    "모놀리식": ["monolithic", "monolith"],
    "서버리스": ["serverless", "server-less", "faas"],
    "함수형서비스": ["faas", "function as a service", "lambda"],
    "인프라": ["infrastructure", "infra"],
    "프로비저닝": ["provisioning", "provision"],
    "스케일링": ["scaling", "scale", "scalability"],
    "스케일업": ["scale up", "vertical scaling", "scale-up"],
    "스케일아웃": ["scale out", "horizontal scaling", "scale-out"],
    "오토스케일링": ["auto scaling", "autoscaling", "auto-scaling"],
    "고가용성": ["high availability", "ha", "highly available"],
    "장애조치": ["failover", "fail over", "failure recovery"],
    "재해복구": ["disaster recovery", "dr"],
    "로깅": ["logging", "logs", "log"],
    "모니터링": ["monitoring", "monitor", "observability"],
    "알림": ["alert", "alerts", "alerting", "notification"],
    "대시보드": ["dashboard", "dashboards"],
    "메트릭": ["metric", "metrics", "measurements"],
    "추적": ["tracing", "trace", "distributed tracing"],
    "프로파일링": ["profiling", "profile", "profiler"],

    # ==========================================================================
    # DevOps & CI/CD
    # ==========================================================================
    "데브옵스": ["devops", "dev ops", "development operations"],
    "지속적통합": ["ci", "continuous integration"],
    "지속적배포": ["cd", "continuous deployment", "continuous delivery"],
    "파이프라인": ["pipeline", "pipelines", "ci cd pipeline"],
    "빌드": ["build", "builds", "building"],
    "배포": ["deployment", "deploy", "release"],
    "릴리스": ["release", "releases", "version release"],
    "롤아웃": ["rollout", "roll out", "deployment rollout"],
    "카나리": ["canary", "canary deployment", "canary release"],
    "블루그린": ["blue green", "blue-green", "blue green deployment"],
    "버전관리": ["version control", "vcs", "source control"],
    "깃": ["git", "git version control"],
    "브랜치": ["branch", "branches", "git branch"],
    "머지": ["merge", "merging", "git merge"],
    "풀리퀘스트": ["pull request", "pr", "merge request"],
    "커밋": ["commit", "commits", "git commit"],
    "푸시": ["push", "git push"],
    "풀": ["pull", "git pull"],
    "클론": ["clone", "git clone"],
    "포크": ["fork", "forking", "git fork"],
    "리베이스": ["rebase", "git rebase"],
    "태그": ["tag", "tags", "git tag"],
    "코드리뷰": ["code review", "review", "peer review"],
    "테스트": ["test", "tests", "testing"],
    "유닛테스트": ["unit test", "unit testing", "unittest"],
    "통합테스트": ["integration test", "integration testing"],
    "엔드투엔드": ["e2e", "end to end", "end-to-end test"],
    "테스트커버리지": ["test coverage", "code coverage", "coverage"],
    "리팩토링": ["refactoring", "refactor", "code refactoring"],
    "코드품질": ["code quality", "quality", "clean code"],

    # ==========================================================================
    # 보안
    # ==========================================================================
    "보안": ["security", "secure", "cybersecurity"],
    "암호화": ["encryption", "encrypt", "cryptography"],
    "복호화": ["decryption", "decrypt"],
    "해시": ["hash", "hashing", "hash function"],
    "해싱": ["hashing", "hash", "hash algorithm"],
    "솔트": ["salt", "salting", "password salt"],
    "인증서": ["certificate", "certificates", "ssl certificate"],
    "공개키": ["public key", "public-key"],
    "개인키": ["private key", "private-key", "secret key"],
    "대칭키": ["symmetric key", "symmetric encryption"],
    "비대칭키": ["asymmetric key", "asymmetric encryption"],
    "취약점": ["vulnerability", "vulnerabilities", "security flaw"],
    "익스플로잇": ["exploit", "exploits", "security exploit"],
    "패치": ["patch", "patches", "security patch"],
    "방어": ["defense", "protection", "security defense"],
    "침입탐지": ["intrusion detection", "ids", "intrusion"],
    "악성코드": ["malware", "malicious code", "virus"],
    "랜섬웨어": ["ransomware", "ransom"],
    "피싱": ["phishing", "phishing attack"],
    "디도스": ["ddos", "denial of service", "dos attack"],
    "크로스사이트": ["xss", "cross site scripting", "cross-site"],
    "에스큐엘인젝션": ["sql injection", "sqli", "injection attack"],
    "제로데이": ["zero day", "zero-day", "0day"],

    # ==========================================================================
    # 인공지능 & 머신러닝
    # ==========================================================================
    "인공지능": ["ai", "artificial intelligence"],
    "머신러닝": ["machine learning", "ml"],
    "딥러닝": ["deep learning", "dl", "neural network"],
    "신경망": ["neural network", "neural net", "nn"],
    "자연어처리": ["nlp", "natural language processing"],
    "컴퓨터비전": ["computer vision", "cv", "image recognition"],
    "강화학습": ["reinforcement learning", "rl"],
    "지도학습": ["supervised learning", "supervised"],
    "비지도학습": ["unsupervised learning", "unsupervised"],
    "트랜스포머": ["transformer", "transformers", "attention"],
    "어텐션": ["attention", "attention mechanism", "self-attention"],
    "임베딩": ["embedding", "embeddings", "vector embedding"],
    "벡터": ["vector", "vectors"],
    "텐서": ["tensor", "tensors"],
    "모델": ["model", "models", "ml model"],
    "학습": ["training", "train", "learning"],
    "추론": ["inference", "prediction", "predict"],
    "파인튜닝": ["fine-tuning", "fine tuning", "finetuning"],
    "전이학습": ["transfer learning", "pretrained"],
    "사전학습": ["pretraining", "pre-training", "pretrained"],
    "하이퍼파라미터": ["hyperparameter", "hyperparameters", "hyperparam"],
    "배치": ["batch", "batches", "mini-batch"],
    "에폭": ["epoch", "epochs", "training epoch"],
    "손실함수": ["loss function", "loss", "cost function"],
    "옵티마이저": ["optimizer", "optimizers", "optimization"],
    "경사하강법": ["gradient descent", "sgd", "gradient"],
    "역전파": ["backpropagation", "backprop", "back propagation"],
    "과적합": ["overfitting", "overfit", "over-fitting"],
    "과소적합": ["underfitting", "underfit", "under-fitting"],
    "정규화": ["regularization", "regularize", "l1 l2"],
    "드롭아웃": ["dropout", "drop out", "dropout layer"],
    "활성화함수": ["activation function", "activation", "relu"],
    "렐루": ["relu", "rectified linear unit"],
    "소프트맥스": ["softmax", "soft max"],
    "시그모이드": ["sigmoid", "logistic"],
    "생성형": ["generative", "generative ai", "genai"],
    "대규모언어모델": ["llm", "large language model"],
    "프롬프트": ["prompt", "prompts", "prompting"],
    "토큰": ["token", "tokens", "tokenization"],
    "토큰화": ["tokenization", "tokenize", "tokenizer"],
    "검색증강생성": ["rag", "retrieval augmented generation"],
    "파라미터": ["parameter", "parameters", "params"],

    # ==========================================================================
    # 데이터 엔지니어링
    # ==========================================================================
    "데이터파이프라인": ["data pipeline", "pipeline", "etl pipeline"],
    "이티엘": ["etl", "extract transform load"],
    "데이터웨어하우스": ["data warehouse", "dw", "dwh"],
    "데이터레이크": ["data lake", "datalake"],
    "데이터마트": ["data mart", "datamart"],
    "스트리밍": ["streaming", "stream processing", "real-time"],
    "배치처리": ["batch processing", "batch", "batch job"],
    "메시지큐": ["message queue", "mq", "queue"],
    "카프카": ["kafka", "apache kafka"],
    "래빗엠큐": ["rabbitmq", "rabbit mq", "amqp"],
    "스파크": ["spark", "apache spark"],
    "하둡": ["hadoop", "apache hadoop", "hdfs"],
    "에어플로우": ["airflow", "apache airflow"],
    "데이터분석": ["data analysis", "analytics", "data analytics"],
    "비즈니스인텔리전스": ["bi", "business intelligence"],
    "시각화": ["visualization", "visualize", "data visualization"],

    # ==========================================================================
    # 검색 & 정보검색
    # ==========================================================================
    "검색": ["search", "searching", "retrieval"],
    "검색엔진": ["search engine", "search engines"],
    "인덱싱": ["indexing", "index", "inverted index"],
    "역색인": ["inverted index", "inverted-index"],
    "랭킹": ["ranking", "rank", "search ranking"],
    "관련성": ["relevance", "relevant", "relevancy"],
    "리콜": ["recall", "retrieval recall"],
    "정밀도": ["precision", "search precision"],
    "시맨틱검색": ["semantic search", "semantic retrieval"],
    "키워드검색": ["keyword search", "keyword matching"],
    "전문검색": ["full text search", "full-text search", "fts"],
    "유사도": ["similarity", "similar", "cosine similarity"],
    "추천": ["recommendation", "recommend", "recommender"],
    "추천시스템": ["recommender system", "recommendation system"],
    "필터링": ["filtering", "filter", "content filtering"],
    "협업필터링": ["collaborative filtering", "cf"],
    "개인화": ["personalization", "personalize", "personalized"],
    "오픈서치": ["opensearch", "open search"],
    "엘라스틱서치": ["elasticsearch", "elastic search", "es"],
    "루씬": ["lucene", "apache lucene"],
    "솔라": ["solr", "apache solr"],

    # ==========================================================================
    # 소프트웨어 아키텍처
    # ==========================================================================
    "아키텍처": ["architecture", "arch", "software architecture"],
    "디자인패턴": ["design pattern", "design patterns", "pattern"],
    "싱글톤": ["singleton", "singleton pattern"],
    "팩토리": ["factory", "factory pattern"],
    "옵저버": ["observer", "observer pattern"],
    "전략패턴": ["strategy pattern", "strategy"],
    "의존성주입": ["dependency injection", "di", "ioc"],
    "제어역전": ["inversion of control", "ioc"],
    "계층구조": ["layered architecture", "layers", "n-tier"],
    "이벤트기반": ["event driven", "event-driven", "eda"],
    "도메인주도": ["domain driven", "ddd", "domain-driven design"],
    "클린아키텍처": ["clean architecture", "clean arch"],
    "헥사고날": ["hexagonal", "hexagonal architecture", "ports and adapters"],
    "모듈화": ["modular", "modularization", "modularity"],
    "결합도": ["coupling", "loose coupling", "tight coupling"],
    "응집도": ["cohesion", "high cohesion"],
    "확장성": ["scalability", "scalable", "extensibility"],
    "유지보수성": ["maintainability", "maintainable"],
    "재사용성": ["reusability", "reusable"],

    # ==========================================================================
    # 모바일 개발
    # ==========================================================================
    "안드로이드": ["android", "android app"],
    "아이오에스": ["ios", "iphone", "apple"],
    "모바일앱": ["mobile app", "mobile application"],
    "네이티브앱": ["native app", "native application"],
    "하이브리드앱": ["hybrid app", "hybrid application"],
    "크로스플랫폼": ["cross platform", "cross-platform"],
    "리액트네이티브": ["react native", "react-native", "rn"],
    "플러터": ["flutter", "flutter app"],
    "스위프트": ["swift", "swift language"],
    "코틀린": ["kotlin", "kotlin language"],
    "푸시알림": ["push notification", "push notifications", "push"],

    # ==========================================================================
    # 기타 IT 용어
    # ==========================================================================
    "알고리즘": ["algorithm", "algorithms", "algo"],
    "자료구조": ["data structure", "data structures"],
    "최적화": ["optimization", "optimize", "optimizing"],
    "성능": ["performance", "perf"],
    "효율성": ["efficiency", "efficient"],
    "복잡도": ["complexity", "time complexity", "space complexity"],
    "빅오": ["big o", "big-o", "o notation"],
    "코딩": ["coding", "programming", "code"],
    "프로그래밍": ["programming", "coding", "development"],
    "개발": ["development", "dev", "developing"],
    "소프트웨어": ["software", "sw"],
    "하드웨어": ["hardware", "hw"],
    "운영체제": ["operating system", "os"],
    "커널": ["kernel", "os kernel"],
    "드라이버": ["driver", "drivers", "device driver"],
    "펌웨어": ["firmware", "fw"],
    "바이오스": ["bios", "basic input output system"],
    "부트로더": ["bootloader", "boot loader"],
    "메모리": ["memory", "ram", "memory management"],
    "저장소": ["storage", "disk", "storage device"],
    "파일시스템": ["file system", "filesystem", "fs"],
    "터미널": ["terminal", "command line", "cli"],
    "쉘": ["shell", "bash", "command shell"],
    "스크립트": ["script", "scripts", "scripting"],
    "자동화": ["automation", "automate", "automated"],
    "워크플로우": ["workflow", "workflows", "work flow"],
    "애자일": ["agile", "agile development", "scrum"],
    "스크럼": ["scrum", "scrum methodology"],
    "칸반": ["kanban", "kanban board"],
    "스프린트": ["sprint", "sprints"],
    "백로그": ["backlog", "product backlog"],
    "유저스토리": ["user story", "user stories"],
    "기술부채": ["technical debt", "tech debt"],
    "레거시": ["legacy", "legacy system", "legacy code"],
    "마이그레이션": ["migration", "migrate", "system migration"],
    "통합": ["integration", "integrate", "system integration"],
    "호환성": ["compatibility", "compatible"],
    "상호운용성": ["interoperability", "interoperable"],
    "표준": ["standard", "standards", "specification"],
    "문서화": ["documentation", "docs", "document"],
    "주석": ["comment", "comments", "code comment"],
    "로그": ["log", "logs", "logging"],
    "트레이스": ["trace", "tracing", "stack trace"],
    "덤프": ["dump", "memory dump", "core dump"],
    "크래시": ["crash", "crashes", "application crash"],
    "행": ["hang", "hanging", "frozen"],
    "타임아웃": ["timeout", "time out", "request timeout"],
    "재시도": ["retry", "retries", "retry logic"],
    "폴백": ["fallback", "fall back", "fallback mechanism"],
    "서킷브레이커": ["circuit breaker", "circuit-breaker"],
    "헬스체크": ["health check", "healthcheck", "health"],
    "하트비트": ["heartbeat", "heart beat", "keep alive"],
    "핑": ["ping", "ping pong", "network ping"],
    "레이스컨디션": ["race condition", "race", "concurrency bug"],
    "데드락": ["deadlock", "dead lock", "deadlocking"],
    "메모리릭": ["memory leak", "leak", "memory leakage"],
}


def generate_term_mappings(output_path: Path) -> int:
    """IT 용어 매핑을 jsonl 형식으로 생성."""

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for ko_term, en_terms in IT_TERMS.items():
            # 기본 유사도 점수 (첫 번째 번역이 가장 정확)
            terms_with_sim = []
            for i, en_term in enumerate(en_terms):
                # 첫 번째 용어가 가장 높은 유사도
                sim = 0.95 - (i * 0.03)
                sim = max(sim, 0.80)  # 최소 0.80
                terms_with_sim.append({"term": en_term, "sim": round(sim, 2)})

            entry = {
                "ko": ko_term,
                "terms": terms_with_sim,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    return count


def main():
    """메인 함수."""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "dataset" / "v19_high_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    # IT 용어 매핑 생성
    it_terms_path = output_dir / "it_terms.jsonl"
    count = generate_term_mappings(it_terms_path)
    print(f"Generated {count} IT term mappings -> {it_terms_path}")

    # 기존 데이터와 병합
    original_path = output_dir / "term_mappings.jsonl"
    merged_path = output_dir / "term_mappings_merged.jsonl"

    # 기존 데이터 로드
    existing_terms = set()
    existing_data = []
    if original_path.exists():
        with open(original_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                existing_terms.add(item["ko"])
                existing_data.append(item)
        print(f"Loaded {len(existing_data)} existing mappings")

    # IT 용어 추가 (중복 제외)
    added = 0
    with open(it_terms_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            if item["ko"] not in existing_terms:
                existing_data.append(item)
                existing_terms.add(item["ko"])
                added += 1

    # 병합된 데이터 저장
    with open(merged_path, "w", encoding="utf-8") as f:
        for item in existing_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Added {added} new IT terms (skipped {count - added} duplicates)")
    print(f"Total mappings: {len(existing_data)} -> {merged_path}")

    # 원본 파일 백업 및 교체
    if original_path.exists():
        backup_path = output_dir / "term_mappings_backup.jsonl"
        original_path.rename(backup_path)
        print(f"Backed up original to {backup_path}")

    merged_path.rename(original_path)
    print(f"Updated {original_path}")


if __name__ == "__main__":
    main()
