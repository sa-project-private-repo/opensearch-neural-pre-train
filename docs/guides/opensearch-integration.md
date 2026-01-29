# OpenSearch Integration Guide

OpenSearch Neural Sparse 모델 통합 가이드

## 1. OpenSearch 요구사항

### 최소 버전
- **OpenSearch 2.9+** (Neural Sparse 지원)
- **ml-commons plugin** 활성화
- **Python 3.8+** (클라이언트)

### JVM Heap 설정

프로덕션 환경 권장 설정:

```bash
# config/jvm.options
-Xms4g
-Xmx4g
```

**메모리 할당 가이드:**
- 소규모 (<100k documents): 4GB
- 중규모 (100k-1M documents): 8GB
- 대규모 (>1M documents): 16GB+

### Plugin 확인

```bash
# ml-commons plugin 설치 확인
curl -XGET "localhost:9200/_cat/plugins?v"
```

출력 예시:
```
name      component     version
node-1    ml-commons    2.9.0.0
```

---

## 2. Index Mapping

### 기본 Mapping

```json
PUT /neural-sparse-index
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "refresh_interval": "30s",
    "index.codec": "best_compression"
  },
  "mappings": {
    "properties": {
      "content": {
        "type": "text",
        "analyzer": "standard"
      },
      "sparse_vector": {
        "type": "rank_features"
      },
      "title": {
        "type": "text"
      },
      "metadata": {
        "type": "object"
      },
      "timestamp": {
        "type": "date"
      }
    }
  }
}
```

### Mapping 설명

| 필드 | 타입 | 용도 |
|------|------|------|
| `content` | text | BM25 검색용 원문 |
| `sparse_vector` | rank_features | Neural Sparse 벡터 ({"token": score}) |
| `title` | text | 제목 (선택) |
| `metadata` | object | 부가 정보 |
| `timestamp` | date | 색인 시간 |

**중요:** `rank_features` 타입은 양수 값만 허용

---

## 3. Document Indexing

### Python Encoder Class

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from opensearchpy import OpenSearch, helpers
from typing import Dict, List

class NeuralSparseEncoder:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.eval()

    def encode(self, text: str) -> Dict[str, float]:
        """텍스트를 Sparse Vector로 변환"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
            # ReLU 활성화 + max pooling
            activations = torch.relu(logits).squeeze(0)
            max_activations = torch.max(activations, dim=0).values

        # 희소 벡터 생성 (0이 아닌 값만)
        sparse_vector = {}
        for idx, score in enumerate(max_activations.tolist()):
            if score > 0:
                token = self.tokenizer.decode([idx])
                sparse_vector[token] = score

        return sparse_vector
```

### Sparse Vector 포맷

```json
{
  "sparse_vector": {
    "검색": 2.34,
    "엔진": 1.89,
    "neural": 3.12,
    "sparse": 2.76
  }
}
```

**특징:**
- Key-value 쌍 (token: score)
- 양수 값만 저장
- Top-k 토큰만 저장 (메모리 절약)

### Bulk API 사용

```python
def index_documents(
    client: OpenSearch,
    encoder: NeuralSparseEncoder,
    documents: List[Dict],
    index_name: str,
    batch_size: int = 100
):
    """문서를 배치로 색인"""
    def generate_actions():
        for doc in documents:
            sparse_vector = encoder.encode(doc["content"])

            yield {
                "_index": index_name,
                "_id": doc.get("id"),
                "_source": {
                    "content": doc["content"],
                    "sparse_vector": sparse_vector,
                    "title": doc.get("title"),
                    "metadata": doc.get("metadata", {}),
                    "timestamp": doc.get("timestamp")
                }
            }

    # Bulk 색인
    success, failed = helpers.bulk(
        client,
        generate_actions(),
        chunk_size=batch_size,
        request_timeout=60,
        raise_on_error=False
    )

    return success, failed
```

### 색인 예제

```python
# OpenSearch 클라이언트
client = OpenSearch(
    hosts=["http://localhost:9200"],
    http_auth=("admin", "admin"),
    use_ssl=False
)

# Encoder 초기화
encoder = NeuralSparseEncoder("./models/v27")

# 문서 색인
documents = [
    {
        "id": "1",
        "content": "OpenSearch는 오픈소스 검색 엔진입니다.",
        "title": "OpenSearch 소개"
    },
    {
        "id": "2",
        "content": "Neural Sparse 모델은 희소 벡터 검색을 지원합니다.",
        "title": "Neural Sparse"
    }
]

success, failed = index_documents(
    client, encoder, documents, "neural-sparse-index"
)
print(f"성공: {success}, 실패: {len(failed)}")
```

---

## 4. Query 구성

### rank_feature Query (2.9+)

```python
def search_rank_feature(
    client: OpenSearch,
    encoder: NeuralSparseEncoder,
    query_text: str,
    index_name: str,
    size: int = 10
):
    """rank_feature query로 검색"""
    query_vector = encoder.encode(query_text)

    query = {
        "size": size,
        "query": {
            "bool": {
                "should": [
                    {
                        "rank_feature": {
                            "field": f"sparse_vector.{token}",
                            "boost": score
                        }
                    }
                    for token, score in query_vector.items()
                ]
            }
        }
    }

    response = client.search(index=index_name, body=query)
    return response["hits"]["hits"]
```

### neural_sparse Query (2.11+)

OpenSearch 2.11+에서는 `neural_sparse` 쿼리 타입 제공:

```json
POST /neural-sparse-index/_search
{
  "query": {
    "neural_sparse": {
      "sparse_vector": {
        "query_text": "오픈소스 검색 엔진",
        "model_id": "your-model-id"
      }
    }
  }
}
```

**장점:**
- 서버 사이드 인코딩
- 클라이언트 부담 감소
- 모델 버전 일관성

---

## 5. Hybrid Search

### BM25 + Neural Sparse

```python
def hybrid_search(
    client: OpenSearch,
    encoder: NeuralSparseEncoder,
    query_text: str,
    index_name: str,
    bm25_weight: float = 0.7,
    neural_weight: float = 0.3,
    size: int = 10
):
    """하이브리드 검색 (BM25 + Neural Sparse)"""
    query_vector = encoder.encode(query_text)

    query = {
        "size": size,
        "query": {
            "hybrid": {
                "queries": [
                    # BM25 query
                    {
                        "match": {
                            "content": {
                                "query": query_text,
                                "boost": bm25_weight
                            }
                        }
                    },
                    # Neural Sparse query
                    {
                        "bool": {
                            "should": [
                                {
                                    "rank_feature": {
                                        "field": f"sparse_vector.{token}",
                                        "boost": score * neural_weight
                                    }
                                }
                                for token, score in query_vector.items()
                            ]
                        }
                    }
                ]
            }
        }
    }

    response = client.search(index=index_name, body=query)
    return response["hits"]["hits"]
```

### linear_combination (2.13+)

```json
POST /neural-sparse-index/_search
{
  "query": {
    "hybrid": {
      "queries": [
        {
          "match": {
            "content": "오픈소스 검색"
          }
        },
        {
          "neural_sparse": {
            "sparse_vector": {
              "query_text": "오픈소스 검색",
              "model_id": "model-id"
            }
          }
        }
      ]
    }
  },
  "search_pipeline": {
    "phase_results_processors": [
      {
        "normalization-processor": {
          "normalization": {
            "technique": "min_max"
          },
          "combination": {
            "technique": "arithmetic_mean",
            "parameters": {
              "weights": [0.7, 0.3]
            }
          }
        }
      }
    ]
  }
}
```

### Weight Tuning 가이드

| Use Case | BM25 Weight | Neural Weight | 설명 |
|----------|-------------|---------------|------|
| 일반 검색 | 0.7 | 0.3 | 키워드 매칭 우선 |
| 의미 검색 | 0.4 | 0.6 | 의미 유사도 우선 |
| 균형 검색 | 0.5 | 0.5 | 둘 다 중요 |
| 키워드 전용 | 1.0 | 0.0 | BM25만 사용 |
| Neural 전용 | 0.0 | 1.0 | Neural만 사용 |

**튜닝 방법:**
1. 평가 셋 준비 (쿼리 + 정답 문서)
2. Weight 0.1 단위로 조정
3. NDCG@10, Recall@100 측정
4. 최적 조합 선택

---

## 6. 성능 최적화

### Index Refreshing

```json
PUT /neural-sparse-index/_settings
{
  "index": {
    "refresh_interval": "30s"
  }
}
```

**가이드:**
- 실시간 검색: `1s` (기본값)
- 배치 색인: `30s` 또는 `-1` (수동)
- 색인 완료 후 수동 refresh:

```python
client.indices.refresh(index="neural-sparse-index")
```

### Shard 수 설정

```python
# 권장 공식: shard_count = ceil(total_size_gb / 30)
total_docs = 1_000_000
avg_doc_size_kb = 5
total_size_gb = (total_docs * avg_doc_size_kb) / (1024 * 1024)
shard_count = max(1, int(total_size_gb / 30) + 1)

print(f"권장 Shard 수: {shard_count}")
```

**권장 사항:**
- Shard 크기: 10-50GB
- Node당 Shard: 20개 이하
- 소규모: 1-3 shards
- 대규모: 5-10 shards

### Query Caching

```json
PUT /neural-sparse-index/_settings
{
  "index.queries.cache.enabled": true
}
```

**캐시 전략:**
- `request_cache`: 전체 응답 캐싱 (동일 쿼리 반복)
- `query_cache`: 필터 결과 캐싱 (부분 재사용)

### Merge Policy

```json
PUT /neural-sparse-index/_settings
{
  "index": {
    "merge.policy.max_merged_segment": "5gb",
    "merge.scheduler.max_thread_count": 1
  }
}
```

### Force Merge (색인 완료 후)

```python
# 배치 색인 완료 후 세그먼트 최적화
client.indices.forcemerge(
    index="neural-sparse-index",
    max_num_segments=1,
    request_timeout=3600
)
```

**주의:** Force merge는 I/O 부하가 크므로 **색인 완료 후** 한 번만 실행

---

## 7. 모니터링

### Index Stats

```python
stats = client.indices.stats(index="neural-sparse-index")
print(f"문서 수: {stats['_all']['primaries']['docs']['count']}")
print(f"크기: {stats['_all']['primaries']['store']['size_in_bytes'] / (1024**3):.2f} GB")
```

### Search Performance

```python
import time

start = time.time()
results = hybrid_search(client, encoder, "검색 쿼리", "neural-sparse-index")
elapsed = time.time() - start

print(f"검색 시간: {elapsed*1000:.2f}ms")
print(f"결과 수: {len(results)}")
```

### Slow Query Log

```json
PUT /neural-sparse-index/_settings
{
  "index.search.slowlog.threshold.query.warn": "10s",
  "index.search.slowlog.threshold.query.info": "5s"
}
```

---

## 8. 트러블슈팅

### 문제: Sparse vector가 색인되지 않음

**원인:** 음수 값 포함

**해결:**
```python
# ReLU 활성화 확인
activations = torch.relu(logits)  # 음수 제거
```

### 문제: 검색 성능 저하

**진단:**
```python
# Profile API로 성능 분석
query["profile"] = True
response = client.search(index="neural-sparse-index", body=query)
print(response["profile"])
```

**해결:**
- Query caching 활성화
- Shard 수 조정
- Force merge 실행

### 문제: Out of Memory

**해결:**
1. JVM heap 증가
2. Sparse vector 차원 축소 (Top-k 토큰만 저장)
3. Bulk batch 크기 감소

---

## 9. 체크리스트

색인 전:
- [ ] OpenSearch 2.9+ 설치
- [ ] ml-commons plugin 활성화
- [ ] JVM heap 설정 (4GB+)
- [ ] Index mapping 생성

색인 중:
- [ ] Bulk API 사용
- [ ] Refresh interval 조정 (`30s` 또는 `-1`)
- [ ] Error handling 구현

색인 후:
- [ ] Force merge 실행
- [ ] Refresh interval 복원 (`1s`)
- [ ] Query caching 활성화

검색 최적화:
- [ ] Hybrid search weight 튜닝
- [ ] Slow query log 모니터링
- [ ] Profile API로 성능 분석

---

## 참고 자료

- [OpenSearch Neural Sparse Documentation](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/)
- [ml-commons Plugin Guide](https://opensearch.org/docs/latest/ml-commons-plugin/index/)
- [Rank Features Field Type](https://opensearch.org/docs/latest/field-types/supported-field-types/rank-features/)
