#!/usr/bin/env python3
"""
AI 도메인 특화 용어집 및 동의어 매핑
"""

# AI 도메인 특화 용어집 (2025년 기준)
AI_TERMINOLOGY = {
    # 기초 용어
    "인공지능": ["AI", "Artificial Intelligence", "기계지능"],
    "머신러닝": ["Machine Learning", "ML", "기계학습"],
    "딥러닝": ["Deep Learning", "DL", "심층학습"],
    "자연어처리": ["NLP", "Natural Language Processing", "언어처리", "언어이해"],
    "신경망": ["Neural Network", "인공신경망", "뉴럴넷"],

    # LLM 생태계
    "대규모언어모델": ["LLM", "Large Language Model", "거대언어모델", "언어모델"],
    "소형언어모델": ["SLM", "Small Language Model", "sLLM", "경량모델"],
    "트랜스포머": ["Transformer", "어텐션", "Attention"],
    "생성형AI": ["GenAI", "Generative AI", "생성AI"],

    # 모델 및 서비스명 (분절 방지 필요)
    "ChatGPT": ["챗GPT", "챗지피티", "GPT"],
    "OpenAI": ["오픈AI", "오픈에이아이"],
    "OpenSearch": ["오픈서치", "오픈검색"],
    "Elasticsearch": ["엘라스틱서치", "엘라스틱"],
    "Claude": ["클로드", "Claude AI"],
    "Gemini": ["제미나이", "구글제미나이"],
    "LLaMA": ["라마", "라마모델", "Meta LLaMA"],

    # 실전 구현 기술
    "RAG": ["검색증강생성", "Retrieval Augmented Generation", "검색증강"],
    "임베딩": ["Embedding", "벡터변환", "벡터표현"],
    "벡터데이터베이스": ["Vector Database", "벡터DB", "임베딩저장소"],
    "프롬프트엔지니어링": ["Prompt Engineering", "프롬프트최적화", "질문최적화"],
    "미세조정": ["Fine-tuning", "파인튜닝", "추가학습"],
    "할루시네이션": ["Hallucination", "환각", "사실왜곡"],
    "토큰": ["Token", "처리단위", "토큰화"],
    "토크나이저": ["Tokenizer", "토큰화기"],

    # AI Agent 시대
    "AI에이전트": ["AI Agent", "자율지능시스템", "에이전트"],
    "멀티모달": ["Multimodal", "다중양식", "멀티모달AI"],
    "도구호출": ["Tool Calling", "Function Calling", "API호출"],
    "추론": ["Reasoning", "논리적분석", "사고"],
    "사고연쇄": ["Chain of Thought", "CoT", "단계별추론"],

    # 학습 기법
    "강화학습": ["Reinforcement Learning", "RL", "보상학습"],
    "전이학습": ["Transfer Learning", "모델재사용"],
    "데이터증강": ["Data Augmentation", "데이터확장"],
    "과적합": ["Overfitting", "오버피팅"],
    "정규화": ["Regularization", "일반화"],

    # 평가 및 성능
    "정확도": ["Accuracy", "정밀도"],
    "재현율": ["Recall", "리콜"],
    "F1스코어": ["F1 Score", "F1점수"],
    "손실함수": ["Loss Function", "손실", "Loss"],
    "배치": ["Batch", "배치사이즈"],
    "에포크": ["Epoch", "훈련반복"],
    "학습률": ["Learning Rate", "러닝레이트"],

    # 아키텍처 용어
    "어텐션": ["Attention", "주의메커니즘", "Attention Mechanism"],
    "인코더": ["Encoder", "부호화기"],
    "디코더": ["Decoder", "복호화기"],
    "레이어": ["Layer", "층"],
    "파라미터": ["Parameter", "매개변수", "가중치"],
    "활성화함수": ["Activation Function", "활성함수"],

    # 데이터 관련
    "데이터셋": ["Dataset", "데이터세트", "학습데이터"],
    "라벨링": ["Labeling", "레이블링", "어노테이션", "Annotation"],
    "전처리": ["Preprocessing", "데이터정제"],
    "증강": ["Augmentation", "데이터증강"],

    # 검색 및 랭킹
    "검색": ["Search", "탐색", "조회"],
    "랭킹": ["Ranking", "순위", "정렬"],
    "유사도": ["Similarity", "코사인유사도", "Cosine Similarity"],
    "벡터검색": ["Vector Search", "임베딩검색", "신경검색"],
    "희소벡터": ["Sparse Vector", "희소표현"],
    "밀집벡터": ["Dense Vector", "밀집표현"],

    # OpenSearch 관련
    "인덱스": ["Index", "색인"],
    "샤드": ["Shard", "분산"],
    "클러스터": ["Cluster", "클러스터"],
    "노드": ["Node", "노드"],
    "매핑": ["Mapping", "스키마"],

    # 최신 트렌드
    "AGI": ["Artificial General Intelligence", "범용인공지능", "범용지능"],
    "온디바이스AI": ["On-device AI", "엣지AI", "기기내실행"],
    "멀티에이전트": ["Multi-Agent", "다중에이전트", "협업에이전트"],
    "MoE": ["Mixture of Experts", "전문가혼합"],
    "컨텍스트윈도우": ["Context Window", "문맥길이", "메모리"],
    "제로샷": ["Zero-shot", "제로샷학습"],
    "퓨샷": ["Few-shot", "퓨샷학습"],

    # 책임 있는 AI
    "설명가능AI": ["XAI", "Explainable AI", "해석가능AI"],
    "공정성": ["Fairness", "편향제거"],
    "투명성": ["Transparency", "투명성"],
    "책임AI": ["Responsible AI", "윤리AI"],
}

# Special tokens로 추가할 기술 용어 (분절 방지)
TECHNICAL_SPECIAL_TOKENS = [
    # AI 서비스 및 모델명
    "ChatGPT", "OpenAI", "OpenSearch", "Elasticsearch",
    "Claude", "Gemini", "LLaMA", "GPT-4", "GPT-3",

    # 기술 약어
    "LLM", "NLP", "RAG", "AGI", "MoE", "XAI", "CoT",
    "BERT", "RoBERTa", "T5", "BART",

    # 한글 합성어 (분절되기 쉬운 것들)
    "딥러닝", "머신러닝", "프롬프트", "토크나이저",
    "벡터검색", "임베딩", "파인튜닝",

    # 프레임워크
    "PyTorch", "TensorFlow", "HuggingFace", "Transformers",
    "LangChain", "LlamaIndex",
]

# 동의어 그룹 (양방향 매핑)
def build_bidirectional_synonyms(terminology_dict):
    """
    양방향 동의어 매핑 구성
    예: "검색" <-> "Search", "Search" <-> "검색"
    """
    bidirectional = {}

    for main_term, synonyms in terminology_dict.items():
        # 모든 용어를 소문자로 정규화
        all_terms = [main_term.lower()] + [s.lower() for s in synonyms]

        # 각 용어에 대해 다른 모든 용어를 동의어로 추가
        for term in all_terms:
            bidirectional[term] = [t for t in all_terms if t != term]

    return bidirectional

# 동의어 매핑 생성
AI_SYNONYMS = build_bidirectional_synonyms(AI_TERMINOLOGY)

print(f"✓ AI 용어집 로드 완료:")
print(f"  - 주요 용어: {len(AI_TERMINOLOGY)}개")
print(f"  - Special tokens: {len(TECHNICAL_SPECIAL_TOKENS)}개")
print(f"  - 동의어 매핑: {len(AI_SYNONYMS)}개")
