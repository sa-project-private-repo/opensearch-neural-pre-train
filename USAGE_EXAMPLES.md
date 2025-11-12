# src 모듈 사용 예제

`src/` 패키지의 모든 함수를 쉽게 import하여 사용할 수 있습니다.

## 📦 간편한 Import

### 방법 1: 패키지에서 직접 import (권장)

```python
from src import (
    # Data loading
    load_korean_news_with_dates,

    # Temporal analysis
    calculate_temporal_idf,
    detect_trending_tokens,

    # Loss functions
    neural_sparse_loss_with_regularization,

    # Cross-lingual
    build_comprehensive_bilingual_dictionary,
)
```

### 방법 2: 모듈별 import

```python
from src.losses import neural_sparse_loss_with_regularization
from src.temporal_analysis import calculate_temporal_idf
from src.cross_lingual_synonyms import build_comprehensive_bilingual_dictionary
```

## 🎯 사용 예제

### 1. 시간 기반 IDF 계산

```python
from transformers import AutoTokenizer
from src import (
    load_korean_news_with_dates,
    calculate_temporal_idf,
    detect_trending_tokens,
    build_trend_boost_dict,
)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# 뉴스 데이터 로드 (날짜 포함)
news_data = load_korean_news_with_dates(max_samples=10000)

# Temporal IDF 계산
idf_token_dict, idf_id_dict = calculate_temporal_idf(
    documents=news_data['documents'],
    dates=news_data['dates'],
    tokenizer=tokenizer,
    decay_factor=0.95,  # 최근 문서에 높은 가중치
)

# 트렌딩 토큰 자동 감지
trending_tokens = detect_trending_tokens(
    documents=news_data['documents'],
    dates=news_data['dates'],
    tokenizer=tokenizer,
    recent_days=30,
    top_k=100,
)

print(f"발견된 트렌딩 토큰: {len(trending_tokens)}")
for token_info in trending_tokens[:10]:
    print(f"  {token_info['token']}: {token_info['trend_score']:.2f}x")
```

### 2. 한영 통합 동의어 사전

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
from src import (
    build_comprehensive_bilingual_dictionary,
    get_default_korean_english_pairs,
    apply_bilingual_synonyms_to_idf,
)

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
model = AutoModelForMaskedLM.from_pretrained("klue/bert-base")

# 샘플 문서
documents = [
    "딥러닝 모델(model)을 학습시킵니다.",
    "검색(search) 시스템을 구축합니다.",
    "BERT 모델은 transformer 아키텍처를 사용합니다.",
]

# 기본 한영 쌍 가져오기
manual_pairs = get_default_korean_english_pairs()
print(f"수동 정의된 쌍: {len(manual_pairs)}")

# 포괄적인 bilingual 사전 구축
bilingual_dict = build_comprehensive_bilingual_dictionary(
    documents=documents,
    token_embeddings=model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy(),
    tokenizer=tokenizer,
    bert_model=model.bert,
    manual_pairs=manual_pairs,
)

print(f"전체 bilingual 사전: {len(bilingual_dict)} 항목")

# IDF에 적용
enhanced_idf = apply_bilingual_synonyms_to_idf(
    idf_dict=idf_token_dict,
    bilingual_dict=bilingual_dict,
    tokenizer=tokenizer,
)

# 이제 '모델'과 'model'이 동일한 IDF 값을 가짐
print(f"'모델' IDF: {enhanced_idf.get('모델', 0):.4f}")
print(f"'model' IDF: {enhanced_idf.get('model', 0):.4f}")
```

### 3. 개선된 Loss Function 사용

```python
import torch
from src import neural_sparse_loss_with_regularization, compute_sparsity_metrics

# 가상의 sparse vectors
doc_sparse = torch.randn(32, 30000).relu()  # batch_size=32, vocab_size=30000
query_sparse = torch.randn(32, 30000).relu()
relevance = torch.ones(32)  # 모두 relevant

# IDF dictionary (token → IDF score)
idf_dict = {i: 2.5 for i in range(30000)}

# Loss 계산 (in-batch negatives 포함)
total_loss, loss_components = neural_sparse_loss_with_regularization(
    doc_sparse=doc_sparse,
    query_sparse=query_sparse,
    relevance=relevance,
    idf_dict=idf_dict,
    lambda_l0=5e-4,
    lambda_idf=1e-2,
    temperature=0.05,
    use_in_batch_negatives=True,  # 핵심 개선!
)

print(f"Total Loss: {total_loss.item():.4f}")
print(f"Contrastive Loss: {loss_components['contrastive_loss']:.4f}")
print(f"L0 Regularization: {loss_components['l0_loss']:.4f}")

# Sparsity 메트릭 확인
sparsity_metrics = compute_sparsity_metrics(doc_sparse)
print(f"Sparsity: {sparsity_metrics['sparsity']:.2%}")
print(f"Non-zero elements: {sparsity_metrics['num_nonzero']:.0f}")
```

### 4. Hard Negative Mining

```python
from src import add_hard_negatives_bm25, add_mixed_negatives

# Query-Document 쌍
qd_pairs = [
    {"query": "딥러닝 모델 학습", "pos_doc": "PyTorch로 모델을 학습합니다", "relevance": 1},
    {"query": "검색 시스템", "pos_doc": "OpenSearch 검색 엔진", "relevance": 1},
]

# 전체 문서 풀
documents = [
    "PyTorch로 모델을 학습합니다",
    "OpenSearch 검색 엔진",
    "Keras를 사용한 딥러닝",
    "Elasticsearch 설정 방법",
    "머신러닝 알고리즘 소개",
]

# BM25 기반 Hard Negatives 추가
augmented_pairs = add_hard_negatives_bm25(
    qd_pairs=qd_pairs,
    documents=documents,
    tokenizer=tokenizer,
    num_hard_negatives=2,  # 각 쿼리당 2개의 hard negative
    top_k=100,
)

print(f"원본 쌍: {len(qd_pairs)}")
print(f"증강된 쌍: {len(augmented_pairs)}")

# 혼합 전략 (random + hard negatives)
mixed_pairs = add_mixed_negatives(
    qd_pairs=qd_pairs,
    documents=documents,
    tokenizer=tokenizer,
    num_random=1,
    num_hard=2,
)
```

### 5. 시간 기반 동의어 발견

```python
from src import (
    discover_synonyms_temporal,
    merge_synonym_dictionaries,
    filter_synonyms_by_frequency,
)

# 토큰 임베딩으로 동의어 발견
synonyms = discover_synonyms_temporal(
    documents=news_data['documents'],
    dates=news_data['dates'],
    token_embeddings=model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy(),
    tokenizer=tokenizer,
    method='kmeans',
    n_clusters=500,
)

print(f"발견된 동의어 그룹: {len(synonyms)}")

# 빈도 기준 필터링
filtered_synonyms = filter_synonyms_by_frequency(
    synonym_dict=synonyms,
    documents=news_data['documents'],
    tokenizer=tokenizer,
    min_frequency=10,  # 최소 10번 출현
)

print(f"필터링 후: {len(filtered_synonyms)} 그룹")

# 예제 출력
for term, synonym_list in list(filtered_synonyms.items())[:5]:
    print(f"\n{term}:")
    for syn in synonym_list[:5]:
        print(f"  - {syn}")
```

## 📚 전체 API 목록

### Loss Functions (5개)
- `in_batch_negatives_loss`
- `margin_ranking_loss`
- `contrastive_loss_with_hard_negatives`
- `neural_sparse_loss_with_regularization`
- `compute_sparsity_metrics`

### Data Loading (4개)
- `load_korean_news_with_dates`
- `load_multiple_korean_datasets`
- `create_time_windows`
- `get_recent_documents`

### Temporal Analysis (6개)
- `calculate_temporal_idf`
- `calculate_windowed_idf`
- `detect_trending_tokens`
- `build_trend_boost_dict`
- `apply_temporal_boost_to_idf`
- `analyze_token_frequency_over_time`

### Negative Sampling (4개)
- `add_hard_negatives_bm25`
- `add_random_negatives`
- `add_mixed_negatives`
- `balance_positive_negative_ratio`

### Temporal Clustering (5개)
- `cluster_tokens_temporal`
- `build_synonym_groups_from_clusters`
- `discover_synonyms_temporal`
- `merge_synonym_dictionaries`
- `filter_synonyms_by_frequency`

### Cross-lingual Synonyms (5개)
- `extract_bilingual_terms`
- `discover_cross_lingual_synonyms_by_embedding`
- `build_comprehensive_bilingual_dictionary`
- `get_default_korean_english_pairs`
- `apply_bilingual_synonyms_to_idf`

## 🎓 더 많은 예제

전체 예제는 다음 파일을 참조하세요:
- [tests/test_korean_neural_sparse.py](tests/test_korean_neural_sparse.py) - Loss function 예제
- [tests/test_temporal_features.py](tests/test_temporal_features.py) - 시간 분석 예제
- [tests/test_bilingual_synonyms.py](tests/test_bilingual_synonyms.py) - 한영 동의어 예제
- [notebooks/korean_neural_sparse_training_v0.3.0.ipynb](notebooks/korean_neural_sparse_training_v0.3.0.ipynb) - 전체 학습 파이프라인
