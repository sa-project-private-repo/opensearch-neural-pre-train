# Korean Neural Sparse 모델 개선 방안

## 현재 문제
- BERT tokenizer의 subword 분절로 인한 영어/기술 용어 검색 성능 저하
- 예: "OpenSearch" → Op, ##en, ##S, ##earch

## 개선 방안

### 1. 즉시 적용 가능 (Easy)
```python
# IDF 계산 시 subword 토큰 필터링
def filter_subword_tokens(idf_dict, tokenizer):
    filtered = {}
    for token_id, idf in idf_dict.items():
        token = tokenizer.decode([token_id])
        # ##로 시작하는 subword 토큰 제외
        if not token.startswith('##'):
            filtered[token_id] = idf
    return filtered
```

### 2. 중기 개선 (Medium)
- **Custom Vocabulary 추가**:
  - 기술 용어 사전 구축 (OpenSearch, ChatGPT, etc.)
  - Tokenizer에 special tokens로 등록
  
```python
# 기술 용어를 special tokens로 추가
technical_terms = ["OpenSearch", "ChatGPT", "LLM", "딥러닝", "프롬프트"]
tokenizer.add_tokens(technical_terms)
model.resize_token_embeddings(len(tokenizer))
```

### 3. 장기 개선 (Hard)
- **다른 Tokenizer 시도**:
  - `klue/roberta-base`: 더 나은 한국어 처리
  - SentencePiece 기반 tokenizer
  
- **Character n-gram 추가**:
  - Elastic의 `ngram_analyzer` 방식
  - Subword와 함께 character trigram 사용

### 4. 하이브리드 접근 (Recommended)
```python
# BM25 + Neural Sparse 조합
{
  "query": {
    "bool": {
      "should": [
        {"neural_sparse": {"field": "neural_vector", "query_text": "..."}},
        {"match": {"content": {"query": "...", "boost": 0.3}}}
      ]
    }
  }
}
```

## 권장 사항
현재 모델은 **한국어 위주 콘텐츠에서 BM25와 하이브리드로 사용** 권장
