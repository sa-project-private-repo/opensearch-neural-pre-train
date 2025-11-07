#!/usr/bin/env python3
"""
OpenSearch Inference-Free Neural Sparse - 간단한 IDF 데모
torch 없이 핵심 컨셉만 보여주는 데모입니다.
"""

import json
import math
from collections import Counter

print("=" * 60)
print("OpenSearch Inference-Free Neural Sparse - IDF 데모")
print("=" * 60)

# 한국어 샘플 데이터
SAMPLE_DOCUMENTS = [
    "인공지능은 컴퓨터 시스템이 인간의 지능을 모방하는 기술입니다",
    "머신러닝은 데이터로부터 패턴을 학습하는 인공지능의 한 분야입니다",
    "딥러닝은 인공 신경망을 사용하여 복잡한 문제를 해결합니다",
    "자연어 처리는 컴퓨터가 인간의 언어를 이해하고 처리하는 기술입니다",
    "OpenSearch는 강력한 검색 및 분석 엔진으로 다양한 기능을 제공합니다",
    "벡터 검색은 의미적 유사성을 기반으로 문서를 검색합니다",
    "Neural sparse 검색은 희소 벡터를 사용하여 효율적인 검색을 제공합니다",
    "한국어 자연어 처리는 형태소 분석과 품사 태깅을 포함합니다",
    "트랜스포머 아키텍처는 현대 자연어 처리의 핵심 기술입니다",
    "BERT 모델은 양방향 인코더를 사용하여 문맥을 이해합니다",
    "GPT는 생성형 언어 모델로 다양한 텍스트를 생성할 수 있습니다",
    "LLM은 대규모 언어 모델을 의미하며 ChatGPT가 대표적입니다",
    "임베딩은 텍스트를 벡터 공간으로 변환하는 과정입니다",
    "검색 엔진 최적화는 웹사이트의 가시성을 높이는 작업입니다",
    "데이터베이스는 구조화된 정보를 저장하고 관리하는 시스템입니다",
]

SAMPLE_QUERIES = [
    "인공지능 기술",
    "머신러닝 학습",
    "OpenSearch 검색",
    "neural sparse",
    "한국어 처리",
    "BERT 모델",
    "GPT LLM",
    "벡터 임베딩",
]

print(f"\n📚 샘플 데이터:")
print(f"  문서: {len(SAMPLE_DOCUMENTS)}개")
print(f"  쿼리: {len(SAMPLE_QUERIES)}개")

# 간단한 토크나이저 (공백 기반)
def simple_tokenize(text):
    """간단한 토크나이저 (실제로는 BERT tokenizer 사용)"""
    return text.lower().split()

# Step 1: IDF 계산
print("\n" + "=" * 60)
print("Step 1: IDF (Inverse Document Frequency) 계산")
print("=" * 60)

def calculate_idf(documents):
    """IDF 계산"""
    N = len(documents)
    df = Counter()  # Document frequency

    # 각 토큰이 등장하는 문서 수 계산
    for doc in documents:
        tokens = simple_tokenize(doc)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            df[token] += 1

    # IDF = log(N / df) + 1
    idf_dict = {}
    for token, doc_freq in df.items():
        idf_score = math.log((N + 1) / (doc_freq + 1)) + 1.0
        idf_dict[token] = idf_score

    return idf_dict

idf_dict = calculate_idf(SAMPLE_DOCUMENTS)

print(f"✓ {len(idf_dict)}개 토큰의 IDF 계산 완료")
print(f"  평균 IDF: {sum(idf_dict.values()) / len(idf_dict):.4f}")

# 상위/하위 IDF 출력
sorted_idf = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)

print("\n🔝 IDF 상위 10개 토큰 (희귀한 단어 - 높은 가중치):")
for i, (token, score) in enumerate(sorted_idf[:10], 1):
    print(f"  {i:2d}. {token:15s} - IDF: {score:.4f}")

print("\n🔻 IDF 하위 10개 토큰 (흔한 단어 - 낮은 가중치):")
for i, (token, score) in enumerate(sorted_idf[-10:], 1):
    print(f"  {i:2d}. {token:15s} - IDF: {score:.4f}")

# Step 2: 트렌드 키워드 부스팅
print("\n" + "=" * 60)
print("Step 2: 트렌드 키워드 가중치 부스팅")
print("=" * 60)

TREND_BOOST = {
    'llm': 1.5,
    'gpt': 1.5,
    'chatgpt': 1.5,
    '생성형': 1.4,
    'rag': 1.4,
    'opensearch': 1.3,
    'neural': 1.3,
    'sparse': 1.3,
    '검색': 1.2,
    '인공지능': 1.2,
    'bert': 1.2,
    '임베딩': 1.3,
}

idf_dict_boosted = idf_dict.copy()
boost_count = 0

print("트렌드 키워드 부스팅 적용:")
for keyword, boost_factor in TREND_BOOST.items():
    if keyword in idf_dict_boosted:
        original = idf_dict_boosted[keyword]
        idf_dict_boosted[keyword] = original * boost_factor
        boost_count += 1
        print(f"  ✓ {keyword:15s}: {original:.4f} → {idf_dict_boosted[keyword]:.4f} ({boost_factor}x)")

print(f"\n✓ {boost_count}개 토큰에 부스팅 적용")

# Step 3: Inference-Free 쿼리 인코딩
print("\n" + "=" * 60)
print("Step 3: Inference-Free 쿼리 인코딩 (IDF Lookup)")
print("=" * 60)

def encode_query_inference_free(query, idf_dict):
    """
    쿼리를 sparse vector로 변환 (IDF lookup)
    🔥 이것이 Inference-Free의 핵심입니다!
    """
    tokens = simple_tokenize(query)

    # 토큰별로 IDF 값을 가져옴 (모델 inference 없음!)
    sparse_vec = {}
    for token in tokens:
        if token in idf_dict:
            sparse_vec[token] = idf_dict[token]

    return sparse_vec

print("쿼리 인코딩 테스트:\n")

for query in SAMPLE_QUERIES:
    sparse_vec = encode_query_inference_free(query, idf_dict_boosted)

    print(f"Query: '{query}'")
    print(f"  Tokens: {list(sparse_vec.keys())}")
    print(f"  Sparse vector (non-zero: {len(sparse_vec)}):")

    for token, weight in sorted(sparse_vec.items(), key=lambda x: x[1], reverse=True):
        print(f"    {token:15s}: {weight:.4f}")
    print()

# Step 4: 검색 시뮬레이션
print("\n" + "=" * 60)
print("Step 4: 검색 시뮬레이션")
print("=" * 60)

def encode_document_simple(doc, idf_dict):
    """
    문서를 sparse vector로 변환 (단순화)
    실제로는 BERT 모델 사용
    """
    tokens = simple_tokenize(doc)
    sparse_vec = {}

    for token in tokens:
        if token in idf_dict:
            # 단순화: 실제로는 BERT MLM head의 logits를 사용
            sparse_vec[token] = idf_dict[token] * 0.5  # 가중치 조정

    return sparse_vec

def calculate_similarity(query_vec, doc_vec):
    """Dot product similarity"""
    similarity = 0.0
    for token, weight in query_vec.items():
        if token in doc_vec:
            similarity += weight * doc_vec[token]
    return similarity

# 모든 문서 인코딩 (실제로는 인덱싱 타임에 수행)
print("모든 문서를 sparse vector로 인코딩 중...\n")
doc_vectors = [encode_document_simple(doc, idf_dict_boosted) for doc in SAMPLE_DOCUMENTS]

# 검색 테스트
test_queries = [
    "인공지능 머신러닝",
    "OpenSearch neural sparse 검색",
    "한국어 처리",
]

print("🔍 검색 결과:\n")
print("=" * 60)

for query in test_queries:
    print(f"\n🔎 Query: '{query}'")

    # 쿼리 인코딩 (Inference-Free!)
    query_vec = encode_query_inference_free(query, idf_dict_boosted)

    # 각 문서와의 유사도 계산
    similarities = []
    for i, doc_vec in enumerate(doc_vectors):
        sim = calculate_similarity(query_vec, doc_vec)
        similarities.append((i, sim))

    # 상위 3개 결과
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_results = similarities[:3]

    print("  상위 3개 결과:")
    for rank, (doc_idx, sim_score) in enumerate(top_results, 1):
        doc = SAMPLE_DOCUMENTS[doc_idx]
        print(f"    {rank}. [Score: {sim_score:.4f}] {doc[:60]}...")

# Step 5: idf.json 저장
print("\n" + "=" * 60)
print("Step 5: idf.json 저장 (OpenSearch 형식)")
print("=" * 60)

output_file = "demo_idf.json"

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(idf_dict_boosted, f, ensure_ascii=False, indent=2)

print(f"✓ IDF 가중치 저장: {output_file}")
print(f"  토큰 수: {len(idf_dict_boosted)}")
print(f"  파일 크기: {len(json.dumps(idf_dict_boosted, ensure_ascii=False))} bytes")

# 샘플 idf.json 내용 출력
print("\nidf.json 샘플:")
print("-" * 60)
sample_tokens = list(idf_dict_boosted.items())[:5]
sample_json = {token: weight for token, weight in sample_tokens}
print(json.dumps(sample_json, ensure_ascii=False, indent=2))

# 요약
print("\n" + "=" * 60)
print("✅ 데모 완료!")
print("=" * 60)

print(f"""
핵심 컨셉:

1. 📊 IDF 계산
   - 문서에서 각 토큰의 희귀도를 계산
   - 희귀한 토큰 = 높은 IDF = 중요한 토큰

2. 🔥 트렌드 키워드 부스팅
   - 2024-2025 트렌드 키워드 (LLM, GPT 등)에 가중치 증가
   - 최신 키워드 검색 시 더 높은 점수

3. ⚡ Inference-Free 쿼리 인코딩
   - 쿼리: Tokenizer + IDF lookup만 사용
   - 모델 inference 불필요 → 매우 빠름!
   - BM25와 유사한 지연시간 (1.1x)

4. 🚀 문서 인코딩
   - 문서: BERT 모델로 sparse vector 생성
   - 인덱싱 타임에만 수행 (한 번만)
   - 출력: rank_features 타입

5. 📁 OpenSearch 형식
   - idf.json: 쿼리용 토큰 가중치
   - pytorch_model.bin: 문서 인코더
   - tokenizer files: BERT tokenizer

다음 단계:
  ✓ 전체 스크립트 실행: python3 test_korean_neural_sparse.py
  ✓ Jupyter 노트북 실행: korean_neural_sparse_training.ipynb
  ✓ OpenSearch에 모델 배포
""")

print("\n🎉 데모가 성공적으로 완료되었습니다!")
print(f"📄 생성된 파일: {output_file}")
