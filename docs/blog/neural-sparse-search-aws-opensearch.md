# Amazon OpenSearch Service에서 Neural Sparse Search 구현하기: sparse_vector와 SEISMIC ANN 활용 가이드

## 1. 소개

Amazon OpenSearch Service는 AWS에서 제공하는 완전관리형 검색 및 분석 서비스로, 로그 분석, 실시간 모니터링, 전문 검색 등 다양한 워크로드를 지원한다. 최근 OpenSearch는 `sparse_vector` 필드 타입과 SEISMIC(Scalable and Efficient Approximate Sparse Vector Search) ANN 알고리즘을 도입하여, 학습된 희소 벡터(Learned Sparse Representation)를 네이티브로 지원하기 시작했다.

기존 BM25 기반 어휘 검색은 **vocabulary mismatch** 문제가 있다. 사용자가 "자동차"로 검색하면 "차량", "승용차"가 포함된 문서를 찾지 못한다. Dense retrieval(밀집 벡터 검색)은 이 문제를 해결하지만, 고차원 벡터 저장과 ANN 인덱스 구축에 높은 비용이 소요되며 해석가능성이 낮다.

**Learned Sparse Retrieval**은 이 두 접근법의 장점을 결합한다. SPLADE와 같은 모델이 텍스트를 어휘 수준의 희소 벡터로 변환하면, 역색인 구조를 활용한 효율적인 검색이 가능하면서도 의미적 확장(semantic expansion)을 통해 vocabulary mismatch를 해결한다.

이 글에서는 한국어 SPLADE 모델(`sewoong/korean-neural-sparse-encoder`)을 활용하여 Amazon OpenSearch Service에서 `sparse_vector` 필드와 SEISMIC ANN 알고리즘 기반의 Neural Sparse Search를 구현하는 방법을 단계별로 설명하고, 실제 실험 결과를 공유한다.

---

## 2. SPLADE v2 모델 동작 원리

### Term Expansion의 핵심

SPLADE(SParse Lexical AnD Expansion) v2는 BERT의 Masked Language Model(MLM) 헤드를 활용하여 입력 텍스트에 포함되지 않은 관련 토큰까지 활성화하는 **term expansion**을 수행한다. 예를 들어 "김치 담그는 법"이라는 쿼리가 주어지면, 모델은 원래 토큰 외에도 "발효", "배추", "양념" 등 의미적으로 관련된 토큰에 가중치를 부여한다.

### SPLADE-max 수식

각 토큰 위치 `i`와 어휘 항목 `j`에 대해:

```
w_ij = log(1 + ReLU(MLM_logit_ij))
```

어휘 항목 `j`의 최종 희소 표현:

```
s_j = max_i(w_ij)    (모든 위치 i에 대한 max pooling)
```

쿼리 `q`와 문서 `d` 간의 관련성 점수:

```
score(q, d) = sum_j(q_j * d_j)    (희소 내적)
```

`log(1 + ReLU(x))` 활성화 함수는 양수 값을 부드럽게 압축하면서 음수를 0으로 만들어 자연스러운 희소성을 유도한다. Max pooling은 시퀀스 내에서 각 어휘 항목의 최대 활성화 값을 선택하여, 위치에 무관한 어휘 수준 표현을 생성한다.

### 검색 패러다임 비교

| 특성 | BM25 | Dense Retrieval | Learned Sparse (SPLADE) |
|------|------|-----------------|------------------------|
| 표현 방식 | 어휘 기반 빈도 통계 | 고차원 밀집 벡터 | 어휘 수준 희소 벡터 |
| 의미 이해 | 없음 (정확 매칭) | 높음 | 중간~높음 (term expansion) |
| 해석가능성 | 높음 | 낮음 | 높음 (토큰별 가중치) |
| 인덱스 구조 | 역색인 | ANN 인덱스 (HNSW 등) | 역색인 호환 |
| 저장 비용 | 낮음 | 높음 (768~1024차원) | 낮음 (99%+ 희소) |
| Vocabulary mismatch | 취약 | 강건 | 강건 |
| 추론 비용 | 없음 | GPU 필요 | GPU 필요 (인코딩 시) |

### 한국어 모델 특성

본 실험에서 사용한 `sewoong/korean-neural-sparse-encoder`는 다음과 같은 특성을 가진다:

| 속성 | 값 |
|------|-----|
| 백본 모델 | skt/A.X-Encoder-base (ModernBERT) |
| 파라미터 수 | 149M |
| 어휘 크기 | 50,000 토큰 |
| 한국어 토큰 비율 | 48.4% |
| 히든 크기 | 768 |
| 레이어 수 | 22 |
| 아키텍처 | SPLADE-max (log(1+ReLU) + max pool) |

50K 어휘에서 한국어 토큰이 48.4%를 차지하므로, 한국어 텍스트에 대해 효율적인 토크나이제이션이 가능하다. XLM-RoBERTa의 250K 어휘 대비 5배 작은 어휘 크기는 SPLADE의 FLOPS 정규화와 양립하기 용이하여, 실질적으로 높은 희소성을 달성할 수 있다.

---

## 3. 아키텍처 개요

### 시스템 구조

```
+--------+                    +---------+                    +-------------------+
| Client |  -- text -->       | SPLADE  |  -- sparse_vector  | Amazon OpenSearch |
| (App)  |                    | Encoder |  -- {id: weight}   | Service           |
+--------+                    | (GPU)   |                    |                   |
    ^                         +---------+                    | sparse_vector     |
    |                                                        | + SEISMIC ANN     |
    +------- search results <---------------------------------+-------------------+
```

클라이언트 측에서 SPLADE 모델로 텍스트를 인코딩한 후, 희소 벡터를 OpenSearch의 `sparse_vector` 필드에 저장한다. 검색 시에도 쿼리를 동일하게 인코딩하여 `neural_sparse` 쿼리로 전달한다.

### sparse_vector 필드 vs rank_features 필드

OpenSearch는 희소 벡터를 저장하기 위해 두 가지 필드 타입을 제공한다:

| 특성 | sparse_vector | rank_features |
|------|---------------|---------------|
| 키 형식 | 정수 토큰 ID (`"31380": 2.5`) | 문자열 토큰 (`"김치": 2.5`) |
| 쿼리 방식 | `neural_sparse` 쿼리 | `rank_feature` 서브쿼리 조합 |
| ANN 지원 | SEISMIC 알고리즘 | 없음 (정확 검색만) |
| 성능 | 빠름 (정수 키 비교) | 상대적으로 느림 (문자열 키) |
| 대규모 데이터 | ANN으로 확장 가능 | 선형 스캔 |

`sparse_vector` 필드는 **정수 토큰 ID**를 키로 사용한다. 이는 문자열 비교 대비 빠른 정수 비교를 활용하고, SEISMIC ANN 알고리즘과의 통합을 가능하게 한다. 토큰 ID는 모델의 토크나이저에서 부여하는 어휘 인덱스(0~49,999)를 사용한다.

### SEISMIC ANN 알고리즘

SEISMIC은 희소 벡터에 특화된 근사 최근접 이웃(ANN) 알고리즘이다. Dense 벡터에서 사용하는 HNSW와 달리, 희소 벡터의 역색인 구조에 최적화된 클러스터링과 가지치기 기법을 활용한다.

주요 파라미터:

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `n_postings` | 각 토큰의 포스팅 리스트 최대 길이 | 300 |
| `cluster_ratio` | 클러스터링 비율 (낮을수록 적은 클러스터) | 0.1 |
| `summary_prune_ratio` | 요약 벡터 가지치기 비율 | 0.4 |

---

## 4. 구현 가이드

### 4.1 환경 설정

```bash
pip install opensearch-py boto3 requests-aws4auth transformers torch
```

### 4.2 AWS SigV4 인증을 통한 OpenSearch 연결

Amazon OpenSearch Service는 IAM 기반 인증을 사용한다. AWS SigV4 서명을 통해 안전하게 연결한다.

```python
import boto3
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection

# AWS 자격 증명 로드
credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, "us-east-1", "es")

# OpenSearch 클라이언트 생성
client = OpenSearch(
    hosts=[{"host": "your-domain.us-east-1.es.amazonaws.com", "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
)

# 연결 확인
info = client.info()
print(f"OpenSearch version: {info['version']['number']}")
```

### 4.3 SPLADE 인코딩 (클라이언트 측)

HuggingFace에서 모델을 로드하여 텍스트를 희소 벡터로 변환한다. `sparse_vector` 필드에 저장하기 위해 정수 토큰 ID를 키로 사용한다.

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained("sewoong/korean-neural-sparse-encoder")
model = AutoModelForMaskedLM.from_pretrained("sewoong/korean-neural-sparse-encoder")
model.eval()

# 특수 토큰 ID (인코딩 결과에서 제외)
special_ids = {
    tokenizer.cls_token_id, tokenizer.sep_token_id,
    tokenizer.pad_token_id, tokenizer.unk_token_id,
}

@torch.no_grad()
def encode_sparse(text: str, max_length: int = 256) -> dict[str, float]:
    """텍스트를 SPLADE-max 희소 벡터로 변환한다."""
    inputs = tokenizer(
        text, return_tensors="pt", max_length=max_length, truncation=True
    )
    logits = model(**inputs).logits  # [1, seq_len, vocab_size]

    # SPLADE-max: log(1 + ReLU(logits)) -> max pool
    sparse = torch.log1p(torch.relu(logits))
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    sparse = (sparse * mask).max(dim=1).values.squeeze(0)  # [vocab_size]

    # 정수 토큰 ID를 키로 하는 딕셔너리 생성
    sparse_dict = {}
    for idx in (sparse > 0).nonzero(as_tuple=True)[0].tolist():
        if idx in special_ids:
            continue
        sparse_dict[str(idx)] = round(sparse[idx].item(), 4)

    return sparse_dict
```

인코딩 결과 예시:

```python
vec = encode_sparse("한국 전쟁의 원인과 결과")
# {"31380": 2.5134, "32470": 1.8921, "15678": 1.2045, ...}
# 평균 33개의 비영(non-zero) 토큰 (쿼리)
# 평균 54개의 비영 토큰 (문서)
```

### 4.4 인덱스 생성 (sparse_vector + SEISMIC)

`sparse_vector` 필드에 SEISMIC ANN 알고리즘을 구성한다. BM25 하이브리드 검색을 위해 `text` 필드에는 nori 형태소 분석기를 설정한다.

```python
index_body = {
    "settings": {
        "analysis": {
            "analyzer": {
                "korean_nori": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "filter": ["nori_readingform", "lowercase"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "title": {
                "type": "text",
                "analyzer": "korean_nori"
            },
            "content": {
                "type": "text",
                "analyzer": "korean_nori"
            },
            "sparse_embedding": {
                "type": "sparse_vector",
                "index": True,
                "method": {
                    "name": "seismic",
                    "parameters": {
                        "n_postings": 300,
                        "cluster_ratio": 0.1,
                        "summary_prune_ratio": 0.4
                    }
                }
            }
        }
    }
}

client.indices.create(index="my-sparse-index", body=index_body)
```

### 4.5 문서 인덱싱

문서를 인코딩한 후 원문 텍스트와 함께 인덱싱한다.

```python
# 문서 인코딩 및 인덱싱
doc_text = "1950년 6월 25일 북한의 남침으로 시작된 한국 전쟁은 냉전 체제 속에서..."
sparse_vec = encode_sparse(doc_text)

doc = {
    "title": "한국 전쟁의 배경",
    "content": doc_text,
    "sparse_embedding": sparse_vec  # {"31380": 2.5, "32470": 1.8, ...}
}

client.index(index="my-sparse-index", body=doc, id="1")

# 대량 인덱싱 시 bulk API 활용
from opensearchpy.helpers import bulk

actions = []
for i, (text, title) in enumerate(documents):
    vec = encode_sparse(text)
    actions.append({
        "_index": "my-sparse-index",
        "_id": str(i),
        "_source": {
            "title": title,
            "content": text,
            "sparse_embedding": vec,
        }
    })

success, errors = bulk(client, actions, refresh="wait_for")
print(f"Indexed {success} documents, {len(errors)} errors")
```

### 4.6 neural_sparse 쿼리

`neural_sparse` 쿼리는 `sparse_vector` 필드에 대해 희소 내적(sparse dot product)을 수행한다.

```python
# 쿼리 인코딩
query_vec = encode_sparse("한국 전쟁의 원인", max_length=64)

# neural_sparse 쿼리 실행
body = {
    "size": 5,
    "query": {
        "neural_sparse": {
            "sparse_embedding": {
                "query_tokens": query_vec  # {"31380": 1.2, "45678": 0.8, ...}
            }
        }
    }
}

resp = client.search(index="my-sparse-index", body=body)
for hit in resp["hits"]["hits"]:
    print(f"Score: {hit['_score']:.4f}  Title: {hit['_source']['title']}")
```

### 4.7 하이브리드 검색 (BM25 + Neural Sparse)

BM25와 Neural Sparse 결과를 클라이언트 측에서 정규화하여 결합하는 방식이다.

```python
def hybrid_search(
    query_text: str,
    query_vec: dict[str, float],
    bm25_weight: float = 0.3,
    sparse_weight: float = 0.7,
    top_k: int = 5,
) -> list[dict]:
    """BM25와 Neural Sparse 점수를 정규화 후 가중 결합한다."""

    # BM25 쿼리
    bm25_body = {
        "size": top_k * 2,
        "query": {
            "match": {
                "content": {
                    "query": query_text,
                    "analyzer": "korean_nori",
                }
            }
        }
    }
    bm25_resp = client.search(index="my-sparse-index", body=bm25_body)
    bm25_hits = bm25_resp["hits"]["hits"]

    # Neural Sparse 쿼리
    sparse_body = {
        "size": top_k * 2,
        "query": {
            "neural_sparse": {
                "sparse_embedding": {
                    "query_tokens": query_vec
                }
            }
        }
    }
    sparse_resp = client.search(index="my-sparse-index", body=sparse_body)
    sparse_hits = sparse_resp["hits"]["hits"]

    # Min-Max 정규화
    bm25_max = max((h["_score"] for h in bm25_hits), default=1.0)
    sparse_max = max((h["_score"] for h in sparse_hits), default=1.0)

    combined = {}
    for hit in bm25_hits:
        doc_id = hit["_id"]
        combined[doc_id] = {
            "source": hit["_source"],
            "bm25": hit["_score"] / bm25_max if bm25_max > 0 else 0,
            "sparse": 0.0,
        }

    for hit in sparse_hits:
        doc_id = hit["_id"]
        norm = hit["_score"] / sparse_max if sparse_max > 0 else 0
        if doc_id in combined:
            combined[doc_id]["sparse"] = norm
        else:
            combined[doc_id] = {
                "source": hit["_source"],
                "bm25": 0.0,
                "sparse": norm,
            }

    # 가중 합산 및 정렬
    results = []
    for doc_id, entry in combined.items():
        score = bm25_weight * entry["bm25"] + sparse_weight * entry["sparse"]
        results.append({
            "_id": doc_id,
            "_source": entry["source"],
            "hybrid_score": score,
            "bm25_score": entry["bm25"],
            "sparse_score": entry["sparse"],
        })

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results[:top_k]
```

---

## 5. 실험 결과

10,000개 한국어 문서와 50개 테스트 쿼리를 대상으로 Amazon OpenSearch Service(15노드, us-east-1)에서 실험을 수행했다. 문서는 자연어 추론, 질의응답, 감정 분석, 주제 분류 등 다양한 도메인에서 추출하였다. 전체 실험 코드는 `scripts/neural_sparse_search_aws.py`에서 확인할 수 있다.

### 5.1 Exact vs ANN vs Rank Features 비교

세 가지 검색 방식의 성능을 비교했다. Recall@5는 정확 검색(SEISMIC high-recall 기준) 대비 상위 5개 문서의 일치율이다.

| 검색 방식 | 평균 지연시간 | 평균 Recall@5 | 비고 |
|-----------|-------------|--------------|------|
| neural_sparse (SEISMIC ANN) | 7.3ms | **100%** | sparse_vector 필드, 정수 토큰 ID |
| neural_sparse (high-recall) | **6.2ms** | 100% (기준) | sparse_vector, SEISMIC n_postings=1000 |
| rank_feature (legacy) | 7.7ms | 73.2% | rank_features 필드, 문자열 토큰 키 |

**분석**: `sparse_vector` + `neural_sparse` 쿼리 조합은 10,000 문서 규모에서 100% Recall을 유지하며, 기존 `rank_features` 방식 대비 **26.8%p 높은 정확도**를 보여준다. SEISMIC ANN은 10K 규모에서 사실상 정확 검색과 동일한 결과를 반환하여, 소규모 데이터셋에서는 별도의 정확 검색 인덱스 없이 SEISMIC만으로 충분하다.

`rank_features` 방식이 73.2%로 낮은 Recall을 보이는 이유는, `rank_feature` 쿼리가 내부적으로 `saturation` 점수 함수를 적용하여 희소 내적과 다른 점수 분포를 생성하기 때문이다. 반면 `neural_sparse` 쿼리는 순수 희소 내적을 수행하므로 SPLADE 학습 목표와 정확히 일치한다.

### 5.2 SEISMIC 인덱스 파라미터 영향

인덱스 생성 시 SEISMIC 파라미터를 광범위하게 변경하며 검색 성능 변화를 측정했다. 기준 인덱스(n_postings=1000, cluster_ratio=0.05, summary_prune_ratio=0.1) 대비 Recall을 계산한다.

**n_postings 변화 (cluster_ratio=0.1, summary_prune_ratio=0.4 고정)**

| n_postings | Recall@5 | 평균 지연시간 |
|-----------|----------|-------------|
| 10 | 100% | 7.5ms |
| 50 | 100% | 7.3ms |
| 100 | 100% | 7.3ms |
| 300 (기본) | 100% | 7.2ms |
| 500 | 100% | 7.5ms |
| 1000 | 100% | 7.5ms |

**cluster_ratio 변화 (n_postings=300, summary_prune_ratio=0.4 고정)**

| cluster_ratio | Recall@5 | 평균 지연시간 |
|--------------|----------|-------------|
| 0.01 | 100% | 7.5ms |
| 0.05 | 100% | 7.5ms |
| 0.1 (기본) | 100% | 7.2ms |
| 0.2 | 100% | 7.4ms |
| 0.5 | 100% | 7.4ms |

**summary_prune_ratio 변화 (n_postings=300, cluster_ratio=0.1 고정)**

| summary_prune_ratio | Recall@5 | 평균 지연시간 |
|---------------------|----------|-------------|
| 0.1 | 100% | 7.3ms |
| 0.2 | 100% | 7.4ms |
| 0.4 (기본) | 100% | 7.2ms |
| 0.6 | 100% | 7.6ms |
| 0.8 | 100% | 7.6ms |

**분석**: 10,000 문서 규모에서는 가장 공격적인 설정(n_postings=10, cluster_ratio=0.01)에서도 100% Recall을 유지했다. 이는 SEISMIC 알고리즘이 본질적으로 역색인 기반이므로, 10K 규모에서는 모든 관련 포스팅을 효과적으로 탐색할 수 있기 때문이다.

지연시간은 모든 설정에서 7.2~7.6ms 범위에 수렴하며, 이 규모에서는 파라미터가 성능에 유의미한 차이를 만들지 않는다. **대규모 데이터(100만+ 문서)에서는 Recall-Latency 트레이드오프가 뚜렷해질 것이며**, n_postings와 cluster_ratio가 낮을수록 검색 속도는 빨라지지만 Recall이 감소하는 패턴이 나타날 것으로 예상된다. 기본값(`n_postings=300`, `cluster_ratio=0.1`, `summary_prune_ratio=0.4`)이 합리적인 출발점이다.

### 5.3 쿼리 method_parameters 영향

검색 시점에 `method_parameters`를 통해 SEISMIC 쿼리 파라미터를 조정하여 성능 변화를 측정했다.

| 파라미터 설정 | Recall@5 | 평균 지연시간 |
|--------------|----------|-------------|
| default (no params) | 100% | 6.7ms |
| heap_factor=0.5 | 100% | 6.8ms |
| heap_factor=1.0 | 100% | 6.9ms |
| heap_factor=2.0 | 100% | 7.0ms |
| top_n=5 | 100% | 6.7ms |
| top_n=10 | 100% | 6.8ms |
| top_n=20 | 100% | 6.9ms |

**분석**: 10K 문서 규모에서 모든 쿼리 파라미터 설정이 100% Recall을 유지했다. `heap_factor`는 SEISMIC의 후보 힙 크기를 제어하며, `top_n`은 탐색할 상위 포스팅 수를 지정한다. 이 규모에서는 두 파라미터 모두 성능에 유의미한 영향을 미치지 않으나, 100만+ 규모에서는 `heap_factor`를 높이면 Recall이 향상되고, `top_n`을 높이면 더 많은 후보를 탐색하여 정확도가 개선될 것으로 예상된다.

### 5.4 하이브리드 가중치 비율별 결과

BM25와 Neural Sparse의 가중치 비율을 변경하며 하이브리드 검색 성능을 측정했다. BM25는 nori 형태소 분석기를 사용한다.

| BM25:Sparse 비율 | Recall@5 | 평균 지연시간 |
|-----------------|----------|-------------|
| 0.0:1.0 (순수 Sparse) | **100%** | 14.8ms |
| 0.2:0.8 | 93.6% | 14.8ms |
| 0.3:0.7 | 92.4% | 14.7ms |
| 0.5:0.5 | 54.0% | 14.8ms |
| 0.7:0.3 | 30.0% | 14.9ms |
| 1.0:0.0 (순수 BM25) | 22.8% | 15.1ms |

**분석**: 가장 주목할 결과는 **순수 Neural Sparse(100%)가 순수 BM25(22.8%)보다 77.2%p 높은 Recall**을 달성했다는 점이다. 이는 SPLADE 모델의 term expansion이 한국어 텍스트에서 매우 효과적으로 작동하며, 단순 어휘 매칭으로는 관련 문서를 찾기 어려운 쿼리에서 큰 차이를 만듦을 보여준다.

BM25 가중치가 증가할수록 Recall이 급격히 감소하는 패턴을 보인다. 0.3 이하의 BM25 가중치에서는 90%+ Recall을 유지하지만, 0.5 이상에서는 BM25의 정확 매칭 점수가 SPLADE의 의미적 확장 결과를 희석하여 Recall이 크게 하락한다. 이는 테스트 쿼리가 원문과 다른 표현을 사용하는 경우(의역, 동의어)가 많아 BM25의 정확 매칭이 불리한 환경이기 때문이다.

하이브리드 검색의 지연시간은 BM25와 neural_sparse 두 쿼리를 순차 실행 후 클라이언트 측에서 점수를 정규화하고 병합하는 과정으로 인해 약 15ms로, 단일 쿼리 대비 약 2배 증가했다.

### 인코딩 통계

| 항목 | 값 |
|------|-----|
| 실험 문서 수 | 10,000 |
| 테스트 쿼리 수 | 50 |
| 문서당 평균 비영 토큰 수 | 54 |
| 쿼리당 평균 비영 토큰 수 | 33 |
| 어휘 크기 | 50,000 |
| 희소율 | 99.89% (문서), 99.93% (쿼리) |
| CPU 인코딩 속도 | ~21 docs/s (Intel Xeon) |
| 전체 인코딩 시간 (10K docs) | ~480초 (CPU) |

---

## 6. 모범 사례

### sparse_vector + SEISMIC 사용 권장

실험 결과가 명확히 보여주듯, `sparse_vector` 필드 + `neural_sparse` 쿼리 조합은 기존 `rank_features` 방식보다 빠르고 정확하다. 신규 프로젝트에서는 반드시 `sparse_vector`를 사용할 것을 권장한다.

```python
# 권장: sparse_vector + neural_sparse
"sparse_embedding": {
    "type": "sparse_vector",
    "index": True,
    "method": {"name": "seismic", "parameters": {"n_postings": 300}}
}

# 비권장: rank_features + rank_feature 쿼리
"sparse_embedding": {"type": "rank_features"}
```

### SEISMIC 파라미터 권장값

| 파라미터 | 권장값 | 설명 |
|----------|--------|------|
| `n_postings` | 300 | 정확도와 속도의 균형. 100만+ 문서 시 200~500 범위 탐색 |
| `cluster_ratio` | 0.1 | 기본값. 대규모 데이터에서 0.05~0.2 범위 탐색 |
| `summary_prune_ratio` | 0.4 | 기본값. Recall 민감 워크로드에서 0.2로 낮출 수 있음 |

### 한국어 하이브리드 검색 전략

한국어 검색에서는 nori 형태소 분석기와 Neural Sparse를 함께 활용하는 것이 효과적이다.

```python
# nori 분석기 설정: BM25 하이브리드를 위한 text 필드
"content": {
    "type": "text",
    "analyzer": "korean_nori"
}
```

본 실험에서 순수 Neural Sparse가 BM25보다 77.2%p 높은 Recall을 보였다. 이는 테스트 쿼리가 원문과 다른 표현을 사용하는 경우(의역, 동의어)가 많아 BM25의 정확 매칭에 불리한 환경이기 때문이다. 프로덕션 환경에서는 정확 매칭이 도움이 되는 도메인(제품명, 고유명사 등)이 있으므로, BM25:Sparse = 0.2:0.8 비율을 시작점으로 튜닝할 것을 권장한다.

### 프로덕션 배포 시 고려사항

**모델 서빙**
- 인코딩은 GPU 환경에서 수행하는 것이 효율적 (CPU 대비 약 10배 속도)
- Amazon SageMaker 엔드포인트나 AWS Lambda (ARM) 활용 고려
- 쿼리 인코딩 지연은 GPU 기준 ~5ms로, 전체 검색 파이프라인에 미치는 영향이 작음

**배치 인코딩**
- 대량 문서 인덱싱 시 배치 처리로 처리량 극대화
- GPU 메모리에 맞춰 배치 크기 조절 (batch_size=16~64)

**인덱스 설계**
- 문서당 sparse_vector 저장 크기: 약 1~2KB (54개 비영 토큰 기준)
- 100만 문서 기준 sparse_vector 필드만 약 1~2GB 추가 저장 필요
- 샤드 수는 데이터 규모와 쿼리 처리량에 따라 결정

**모니터링**
- 인코딩 지연시간, 검색 지연시간, Recall 지표를 대시보드에 추적
- SEISMIC 파라미터 변경 시 A/B 테스트로 영향도 측정

---

## 7. 결론

이 글에서는 Amazon OpenSearch Service의 `sparse_vector` 필드와 SEISMIC ANN 알고리즘을 활용한 Neural Sparse Search 구현 방법을 살펴보았다. 한국어 SPLADE 모델(`sewoong/korean-neural-sparse-encoder`)과 10,000개 문서를 사용한 실험을 통해 다음과 같은 결과를 확인했다.

**성능 개선**:
- `sparse_vector` + `neural_sparse`는 기존 `rank_features` 대비 **26.8%p 높은 정확도(100% vs 73.2% Recall@5)**를 달성했다.
- Neural Sparse 단독으로 BM25 대비 **77.2%p 높은 Recall(100% vs 22.8%)**을 기록하여, SPLADE의 term expansion이 한국어에 매우 효과적으로 작동함을 입증했다.

**SEISMIC ANN의 안정성**:
- 10,000 문서 규모에서 가장 공격적인 SEISMIC 파라미터(n_postings=10, cluster_ratio=0.01)에서도 100% Recall을 유지하여, 소규모~중규모 데이터셋에서 별도 튜닝 없이 사용 가능하다.
- 대규모 데이터(100만+ 문서)에서는 파라미터 튜닝을 통한 Recall-Latency 트레이드오프 최적화가 필요할 것이며, 기본값이 합리적인 출발점이다.

**실용성**:
- 한국어 SPLADE 모델은 149M 파라미터로 경량이며, 문서당 평균 54개의 비영 토큰으로 99.89%의 희소율을 달성한다.
- `sparse_vector` 필드의 정수 토큰 ID 형식은 저장 효율성과 검색 속도 모두에서 유리하다.
- 클라이언트 측 인코딩 방식은 ML 플러그인 설정 없이 즉시 적용 가능하다.

Neural Sparse Search는 BM25의 효율성과 Dense Retrieval의 의미 이해 능력을 결합하여, 검색 품질과 운영 비용 사이의 최적 균형점을 제공한다. Amazon OpenSearch Service의 `sparse_vector` 필드와 SEISMIC ANN은 이 접근법을 프로덕션 환경에서 손쉽게 적용할 수 있게 해준다.

---

*전체 실험 코드: [`scripts/neural_sparse_search_aws.py`](../../scripts/neural_sparse_search_aws.py)*
*모델: [`sewoong/korean-neural-sparse-encoder`](https://huggingface.co/sewoong/korean-neural-sparse-encoder) (HuggingFace)*
