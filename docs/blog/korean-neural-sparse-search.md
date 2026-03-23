# 한국어 Neural Sparse 검색 모델 개발과 OpenSearch 하이브리드 검색 벤치마크

## 개요

이 글에서는 한국어에 특화된 Neural Sparse 검색 모델을 개발하고, Amazon OpenSearch Service에서 BM25, Dense Vector, Neural Sparse의 단독 성능과 하이브리드 조합 성능을 종합적으로 비교한 결과를 소개합니다.

검색 시스템에서 "의미적으로 관련 있는 문서를 정확하게 찾는 것"은 핵심 과제입니다. 전통적인 BM25 기반의 키워드 매칭은 동의어나 의미적 유사성을 처리하지 못하는 한계가 있고, Dense Vector 기반의 시맨틱 검색은 정확한 키워드 매칭에 약점이 있습니다. Neural Sparse 검색은 이 두 가지 접근법의 장점을 결합하여, 키워드 매칭의 정확성과 의미적 확장을 동시에 제공하는 방법입니다.

## SPLADE: Neural Sparse 검색의 핵심 기술

### Dense Vector와 Sparse Vector의 차이

검색 시스템에서 문서와 쿼리를 벡터로 표현하는 방식은 크게 두 가지로 나뉩니다.

**Dense Vector (밀집 벡터)** 방식은 BERT, BGE-M3, Titan Embedding과 같은 인코더 모델을 사용하여 텍스트를 고정 차원(예: 768, 1024)의 실수 벡터로 변환합니다. 이 벡터의 모든 차원에 값이 존재하며, 각 차원은 명시적인 의미를 갖지 않습니다. 유사도 계산은 코사인 유사도나 내적(dot product)을 사용하며, 검색 시 근사 최근접 이웃(ANN) 알고리즘(HNSW, IVF 등)을 활용합니다.

```
Dense Vector 예시 (1024차원):
[0.023, -0.041, 0.118, 0.005, -0.072, ..., 0.031]  # 모든 차원에 값 존재
```

**Sparse Vector (희소 벡터)** 방식은 텍스트를 어휘(vocabulary) 크기의 벡터로 변환하되, 대부분의 차원이 0이고 소수의 차원에만 값이 존재합니다. 각 차원은 특정 토큰(단어)에 대응하므로 해석이 가능하다는 특징이 있습니다. 유사도 계산은 희소 벡터 간의 내적(dot product)으로 수행합니다.

```
Sparse Vector 예시 (32,000차원 중 ~40개만 비영):
{"서울": 2.66, "맛집": 2.31, "추천": 1.95, "음식점": 1.42, "레스토랑": 1.18, ...}
# 나머지 ~31,960개 차원은 0
```

이 두 방식의 핵심적인 차이는 **해석 가능성**과 **검색 방식**에 있습니다. Dense Vector는 의미를 압축적으로 표현하지만 각 차원이 무엇을 의미하는지 알 수 없습니다. 반면 Sparse Vector는 각 비영 차원이 특정 토큰에 대응하므로, 모델이 어떤 단어를 중요하게 판단했는지 직접 확인할 수 있습니다.

### SPLADE의 작동 원리

SPLADE(SParse Lexical AnD Expansion)는 사전 학습된 Masked Language Model(MLM)의 토큰 예측 능력을 활용하여 희소 벡터를 생성하는 기법입니다. 기존의 BM25가 문서에 실제로 등장하는 단어만을 사용하는 반면, SPLADE는 MLM의 언어 이해 능력을 통해 문서에 직접 등장하지 않는 관련 단어까지 활성화하는 "어휘 확장(lexical expansion)" 기능을 제공합니다.

SPLADE의 핵심 연산 과정은 다음과 같습니다.

**1단계: MLM 로짓 생성**

입력 텍스트를 토크나이저로 토큰화한 후, MLM 모델에 통과시켜 각 입력 토큰 위치에서 전체 어휘에 대한 로짓(logit) 값을 얻습니다. 이 로짓은 해당 위치에 각 어휘 토큰이 등장할 확률을 나타냅니다.

```python
# 입력: "서울 맛집 추천해주세요"
# 토큰화: ["서울", "맛", "##집", "추천", "##해", "##주", "##세요"]
# 각 토큰 위치에서 32,000개 어휘에 대한 로짓 생성
logits = mlm_model(input_ids)  # shape: [seq_len, vocab_size]
```

**2단계: log(1 + ReLU) 활성화**

로짓에 ReLU를 적용하여 음수 값을 제거한 후, log(1 + x) 변환을 적용합니다. 이 변환은 큰 값의 영향을 억제하면서도 양수 값을 보존하는 효과가 있습니다. 이것이 SPLADE-v2 논문에서 제안된 핵심 활성화 함수입니다.

```python
sparse_scores = torch.log1p(torch.relu(logits))  # log(1 + ReLU(x))
```

**3단계: Max Pooling**

시퀀스 차원에 대해 max pooling을 수행하여, 각 어휘 토큰에 대해 모든 입력 위치 중 가장 높은 활성화 값을 선택합니다. 이렇게 하면 입력 길이와 무관하게 어휘 크기의 고정된 희소 벡터가 생성됩니다.

```python
# attention mask를 적용하여 패딩 토큰 제외
masked_scores = sparse_scores * attention_mask.unsqueeze(-1)
sparse_vector = masked_scores.max(dim=1).values  # shape: [vocab_size]
```

**4단계: 어휘 확장 효과**

결과적으로, 입력 텍스트에 직접 등장하지 않는 단어도 활성화될 수 있습니다. 예를 들어 "서울 맛집"이라는 쿼리에 대해 모델은 다음과 같이 관련 토큰을 확장합니다.

```
입력: "서울 맛집"
활성화된 토큰들:
  서울: 2.66    (입력에 존재)
  맛집: 2.31    (입력에 존재)
  음식점: 1.42  (어휘 확장 - 동의어)
  레스토랑: 1.18 (어휘 확장 - 관련어)
  강남: 0.95    (어휘 확장 - 연관 지역)
  ...
```

이 어휘 확장 기능이 SPLADE가 BM25를 능가하는 핵심 이유입니다. BM25는 "맛집"이라는 단어가 문서에 정확히 등장해야만 매칭하지만, SPLADE는 "음식점", "레스토랑" 등 동의어가 포함된 문서도 높은 점수를 부여할 수 있습니다.

### Dense Vector 대비 Neural Sparse의 장단점

| 특성 | Dense Vector | Neural Sparse |
|------|-------------|---------------|
| 표현 차원 | 고정 (768~1024) | 어휘 크기 (32K~50K), 대부분 0 |
| 해석 가능성 | 불가 | 가능 (토큰 단위 가중치) |
| 키워드 매칭 | 약함 | 강함 (BM25 수준) |
| 의미적 유사성 | 강함 | 중간 (어휘 확장 범위 내) |
| 인덱스 크기 | 고정 | 문서 복잡도에 비례 |
| 검색 방식 | ANN (HNSW) | Inverted Index / SEISMIC |

## 한국어 SPLADE 모델 학습

### 베이스 모델 선택

SPLADE의 성능은 베이스 MLM 모델의 품질에 크게 의존합니다. 한국어 SPLADE 모델 개발 과정에서 여러 베이스 모델을 실험한 결과, 다음과 같은 결론을 얻었습니다.

| 베이스 모델 | 파라미터 | 어휘 크기 | 결과 |
|------------|---------|----------|------|
| XLM-RoBERTa | 560M | 250,000 | 실패 - FLOPS 정규화로 희소성 제어 불가 |
| skt/A.X-Encoder-base (ModernBERT) | 149M | 50,000 | 성공 - V33 모델 |
| klue/roberta-large | 337M | 32,000 | 성공 - 최종 모델, V33 대비 성능 향상 |

핵심 발견은 **어휘 크기가 SPLADE FLOPS 정규화와의 호환성을 결정한다**는 것입니다. XLM-RoBERTa의 250K 어휘는 FLOPS 정규화가 충분한 희소성 압력을 가하기 어려워 희소 벡터를 생성할 수 없었습니다. 32K~50K 범위의 어휘가 SPLADE에 적합합니다.

또한 MLM 헤드의 사전 학습 여부가 중요합니다. SPLADE는 MLM 로짓에 의존하므로, 랜덤 초기화된 MLM 헤드로는 의미 있는 토큰 가중치를 생성할 수 없습니다. klue/roberta-large는 사전 학습된 MLM 헤드를 보유하고 있어 별도의 MLM 사전 학습 단계 없이 바로 SPLADE 학습이 가능했습니다.

### 학습 데이터

학습 데이터는 14개 한국어 데이터셋에서 추출한 4,840,000개의 트리플렛(query, positive, negative)으로 구성됩니다.

| 데이터 소스 | 규모 | 유형 |
|-----------|------|------|
| AIHub 뉴스 QA | 1,330K | 뉴스 질의응답 |
| OPUS-100 번역 | 732K | 병렬 코퍼스 |
| ko-triplet | 682K | 검색 트리플렛 |
| mC4-ko | 475K | 웹 텍스트 |
| Wikipedia-ko | 329K | 백과사전 QA |
| 기타 9개 데이터셋 | 1,292K | NLI, 감성분석, 대화 등 |

각 트리플렛은 하나의 쿼리, 하나의 관련 문서(positive), 하나의 비관련 문서(negative)로 구성되며, negative는 BGE-M3 모델을 이용한 hard negative mining으로 생성했습니다.

### 손실 함수와 정규화

SPLADE 학습의 손실 함수는 두 가지 요소로 구성됩니다.

**InfoNCE 손실 (대조 학습)**

쿼리와 관련 문서의 내적을 최대화하고, 비관련 문서와의 내적을 최소화하는 대조 학습 손실입니다. temperature는 1.0을 사용합니다. SPLADE는 희소 벡터 간의 내적(dot product)으로 유사도를 계산하므로, cosine similarity에 사용하는 낮은 temperature(0.07 등)가 아닌 1.0이 적합합니다.

```
L_infonce = -log(exp(q . d+ / tau) / sum(exp(q . di / tau)))
```

**FLOPS 정규화**

모델이 너무 많은 토큰을 활성화하면 희소성이 사라지고 검색 효율이 떨어집니다. FLOPS 정규화는 활성화된 토큰 수를 제어하여 적절한 희소성을 유지합니다.

```
L_flops = sum_j (mean_i (w_j^i))^2
```

여기서 w_j^i는 배치 내 i번째 문서(또는 쿼리)의 j번째 토큰 가중치입니다. 이 정규화는 각 토큰의 평균 활성화를 줄여, 자주 활성화되는 일반적인 토큰(예: 조사, 접속사)의 가중치를 억제합니다.

FLOPS 정규화의 강도는 lambda 파라미터로 조절하며, 쿼리(lambda_q=0.01)와 문서(lambda_d=0.003)에 서로 다른 값을 적용합니다. 학습 초기에는 lambda 값을 목표의 10%에서 시작하여 20,000 스텝에 걸쳐 이차함수(quadratic) 형태로 워밍업합니다.

### 학습 설정

| 하이퍼파라미터 | 값 |
|--------------|-----|
| 베이스 모델 | klue/roberta-large (337M params) |
| 배치 크기 | 128/GPU x 2 (gradient accumulation) x 8 GPUs = 2,048 |
| 학습률 | 5e-5 (cosine decay, 6% warmup) |
| 에폭 | 25 |
| 쿼리 최대 길이 | 64 토큰 |
| 문서 최대 길이 | 256 토큰 |
| FLOPS lambda_q / lambda_d | 0.01 / 0.003 |
| 하드웨어 | 8x NVIDIA B200 GPU (183GB VRAM) |
| 학습 시간 | 약 13.4시간 |
| 정밀도 | bfloat16 mixed precision |
| 분산 학습 | PyTorch DDP (DistributedDataParallel) |

최종 모델의 평균 활성 토큰 수는 쿼리당 약 36개, 문서당 약 58개입니다. 이는 32,000개 어휘 중 0.1~0.2%만 활성화되는 매우 높은 희소성을 보여줍니다.

## 벤치마크 설계

### 평가 데이터셋

3개의 한국어 검색 벤치마크 데이터셋을 사용했습니다.

| 데이터셋 | 쿼리 수 | 문서 수 | 특성 |
|---------|--------|--------|------|
| Ko-StrategyQA | 592 | 9,251 | 다단계 추론이 필요한 전략적 질문, 한국어 위키피디아 기반 |
| MIRACL-ko | 213 | 10,000 | 다국어 정보 검색 벤치마크의 한국어 분할, 위키피디아 기반 |
| Mr.TyDi-ko | 421 | 10,000 | 다양한 언어의 정보 검색 벤치마크, 한국어 위키피디아 기반 |

### 비교 모델

| 모델 | 유형 | 파라미터 | 차원 |
|------|------|---------|------|
| BM25 (Nori tokenizer) | 키워드 검색 | - | - |
| Amazon Titan Embedding v2 | Dense Vector | - | 1,024 |
| opensearch-neural-sparse-encoding-multilingual-v1 | Sparse Vector | 149M | 250K vocab |
| korean-neural-sparse-encoder-base-klue-large | Sparse Vector | 337M | 32K vocab |

### 검색 방법

단독 검색 4가지와 하이브리드 검색 7가지, 총 11가지 방법을 비교했습니다.

단독 검색은 각 모델이 독립적으로 문서를 검색하는 방식입니다. 하이브리드 검색은 Reciprocal Rank Fusion(RRF)을 사용한 후기 결합(late fusion) 방식으로, 각 검색 모델의 순위를 결합합니다. RRF 공식은 다음과 같습니다.

```
RRF_score(d) = sum_r (1 / (k + rank_r(d)))
```

여기서 k는 순위 상수(기본값 60)이며, rank_r(d)는 r번째 검색 시스템에서 문서 d의 순위입니다. RRF는 점수 정규화가 불필요하고, 서로 다른 척도의 검색 결과를 안정적으로 결합할 수 있다는 장점이 있습니다.

### 평가 지표

- **Recall@K**: 상위 K개 결과에 관련 문서가 포함된 쿼리의 비율
- **MRR@10**: 관련 문서의 역순위 평균 (상위 10개 내)
- **Latency P50**: 쿼리 응답 시간의 중앙값 (밀리초)

### 인프라 구성

- OpenSearch 클러스터: Amazon OpenSearch Service (ltr-vector.awsbuddy.com)
- 인덱스 구성: BM25용 Nori 분석기, Dense용 HNSW(FAISS), Sparse용 rank_features
- 샤드: 6개, 레플리카: 2개

## 벤치마크 결과

### 단독 모델 성능 (Recall@1)

| 모델 | Ko-StrategyQA | MIRACL-ko | Mr.TyDi-ko | 평균 |
|------|:---:|:---:|:---:|:---:|
| BM25 | 53.7% | 44.1% | 55.6% | 51.1% |
| OS-sparse-multilingual | 39.5% | 50.7% | 66.0% | 52.1% |
| klue-large sparse (본 연구) | 60.0% | 63.4% | 73.6% | 65.7% |
| Titan Embedding v2 | 68.6% | 60.1% | 74.3% | 67.7% |

### 단독 모델 전체 지표

| 모델 | 데이터셋 | R@1 | R@5 | R@10 | MRR@10 | P50 (ms) |
|------|---------|:---:|:---:|:---:|:---:|:---:|
| BM25 | Ko-StrategyQA | 53.7% | 75.3% | 81.9% | 0.630 | 10 |
| BM25 | MIRACL-ko | 44.1% | 80.8% | 90.6% | 0.594 | 10 |
| BM25 | Mr.TyDi-ko | 55.6% | 79.1% | 85.7% | 0.660 | 10 |
| Titan v2 | Ko-StrategyQA | 68.6% | 83.3% | 86.1% | 0.752 | 33 |
| Titan v2 | MIRACL-ko | 60.1% | 88.7% | 92.5% | 0.722 | 33 |
| Titan v2 | Mr.TyDi-ko | 74.3% | 88.8% | 91.4% | 0.808 | 33 |
| OS-sparse-multi | Ko-StrategyQA | 39.5% | 63.2% | 68.4% | 0.501 | 28 |
| OS-sparse-multi | MIRACL-ko | 50.7% | 83.6% | 91.1% | 0.655 | 25 |
| OS-sparse-multi | Mr.TyDi-ko | 66.0% | 86.9% | 90.7% | 0.757 | 27 |
| klue-sparse | Ko-StrategyQA | 60.0% | 79.2% | 83.6% | 0.684 | 17 |
| klue-sparse | MIRACL-ko | 63.4% | 90.6% | 94.4% | 0.738 | 18 |
| klue-sparse | Mr.TyDi-ko | 73.6% | 91.0% | 94.8% | 0.810 | 18 |

### 하이브리드 검색 성능 (Recall@1)

| 방법 | Ko-StrategyQA | MIRACL-ko | Mr.TyDi-ko | 평균 |
|------|:---:|:---:|:---:|:---:|
| BM25 + Titan v2 | 65.7% | 63.8% | 73.4% | 67.6% |
| BM25 + OS-sparse | 55.9% | 56.8% | 75.3% | 62.7% |
| BM25 + klue-sparse | 62.3% | 66.7% | 75.8% | 68.3% |
| Titan v2 + OS-sparse | 58.6% | 60.1% | 76.5% | 65.1% |
| Titan v2 + klue-sparse | 66.4% | 65.7% | 77.2% | 69.8% |
| BM25 + Titan + OS-sparse | 64.7% | 64.8% | 79.3% | 69.6% |
| BM25 + Titan + klue-sparse | 68.2% | 65.7% | 81.7% | 71.9% |

### 하이브리드 검색 전체 지표

| 방법 | 데이터셋 | R@1 | R@5 | R@10 | MRR@10 | P50 (ms) |
|------|---------|:---:|:---:|:---:|:---:|:---:|
| BM25 + Titan | Ko-StrategyQA | 65.7% | 83.8% | 87.7% | 0.738 | 107 |
| BM25 + Titan | MIRACL-ko | 63.8% | 88.3% | 94.4% | 0.741 | 105 |
| BM25 + Titan | Mr.TyDi-ko | 73.4% | 91.2% | 94.1% | 0.812 | 106 |
| BM25 + klue-sparse | Ko-StrategyQA | 62.3% | 82.8% | 87.7% | 0.715 | 35 |
| BM25 + klue-sparse | MIRACL-ko | 66.7% | 90.1% | 95.3% | 0.770 | 36 |
| BM25 + klue-sparse | Mr.TyDi-ko | 75.8% | 91.0% | 96.4% | 0.828 | 36 |
| Titan + klue-sparse | Ko-StrategyQA | 66.4% | 85.6% | 87.8% | 0.750 | 122 |
| Titan + klue-sparse | MIRACL-ko | 65.7% | 91.5% | 96.2% | 0.759 | 122 |
| Titan + klue-sparse | Mr.TyDi-ko | 77.2% | 93.6% | 96.7% | 0.845 | 123 |
| BM25 + Titan + klue-sparse | Ko-StrategyQA | 68.2% | 86.0% | 88.2% | 0.759 | 132 |
| BM25 + Titan + klue-sparse | MIRACL-ko | 65.7% | 91.5% | 96.2% | 0.765 | 133 |
| BM25 + Titan + klue-sparse | Mr.TyDi-ko | 81.7% | 94.8% | 97.4% | 0.873 | 133 |

## 결과 분석

### 1. 한국어 특화 모델의 우위

본 연구에서 개발한 klue-large sparse 모델은 OpenSearch 공식 다국어 sparse 모델(opensearch-neural-sparse-encoding-multilingual-v1) 대비 모든 벤치마크에서 대폭 우위를 보였습니다.

| 데이터셋 | OS-sparse-multi | klue-sparse | 차이 |
|---------|:---:|:---:|:---:|
| Ko-StrategyQA | 39.5% | 60.0% | +20.5pp |
| MIRACL-ko | 50.7% | 63.4% | +12.7pp |
| Mr.TyDi-ko | 66.0% | 73.6% | +7.6pp |

이 차이는 베이스 모델의 한국어 이해 능력, 어휘 크기(250K vs 32K), 그리고 한국어 특화 학습 데이터에 기인합니다. 특히 OS-sparse-multilingual의 250K 어휘는 FLOPS 정규화와의 호환성이 낮아, 한국어에서 효과적인 희소 벡터 생성이 어렵습니다.

### 2. Sparse vs Dense 성능 비교

klue-large sparse 모델은 Titan Embedding v2 dense 모델과 대등한 성능을 보였습니다.

- Ko-StrategyQA: Titan v2가 +8.6pp 우위 (68.6% vs 60.0%)
- MIRACL-ko: klue-sparse가 +3.3pp 우위 (63.4% vs 60.1%)
- Mr.TyDi-ko: 거의 동등 (74.3% vs 73.6%)

특히 MIRACL-ko에서 sparse 모델이 dense 모델을 능가한 것은 주목할 만합니다. 이는 위키피디아 기반의 사실적 검색에서 키워드 매칭과 어휘 확장의 조합이 순수 시맨틱 유사도보다 효과적일 수 있음을 보여줍니다.

### 3. 하이브리드 검색의 효과

단독 모델보다 하이브리드 검색이 일관되게 높은 성능을 보였으며, 특히 3-way 하이브리드(BM25 + Titan + klue-sparse)가 전체 최고 성능을 기록했습니다.

Mr.TyDi-ko에서의 성능 변화를 보면:
- BM25 단독: 55.6%
- Titan v2 단독: 74.3%
- klue-sparse 단독: 73.6%
- BM25 + Titan + klue-sparse: **81.7%** (+7.4pp 개선, Titan 단독 대비)

이는 BM25의 정확한 키워드 매칭, Dense의 시맨틱 이해, Sparse의 어휘 확장이 상호 보완적으로 작용한다는 것을 의미합니다.

### 4. 응답 시간 비교

| 방법 | P50 응답 시간 |
|------|:---:|
| BM25 | 10ms |
| klue-sparse (rank_features) | 17-18ms |
| OS-sparse-multi | 25-28ms |
| Titan v2 (kNN) | 33ms |
| BM25 + klue-sparse (2-way) | 35-36ms |
| BM25 + Titan (2-way) | 105-107ms |
| BM25 + Titan + klue-sparse (3-way) | 132-133ms |

klue-sparse 모델은 BM25 대비 약 2배의 응답 시간으로 상당한 성능 향상을 제공합니다. BM25 + klue-sparse 하이브리드는 36ms로, Dense 모델 포함 하이브리드(100ms 이상) 대비 훨씬 빠르면서도 높은 검색 품질을 제공합니다.

## 결론

이 연구를 통해 다음을 확인했습니다.

1. **한국어 특화 SPLADE 모델은 다국어 범용 모델 대비 평균 13.6pp R@1 우위**를 보이며, 언어별 특화 학습의 중요성을 입증합니다.

2. **Neural Sparse 검색은 Dense Vector 검색과 대등한 성능**을 달성할 수 있으며, 특정 벤치마크(MIRACL-ko)에서는 Dense를 능가합니다.

3. **BM25 + Dense + Sparse 3-way 하이브리드가 최적의 검색 품질**을 제공하며, Mr.TyDi-ko에서 81.7% R@1을 달성했습니다.

4. **Sparse 검색은 Dense 대비 낮은 응답 시간**으로 실용적인 운영이 가능합니다. BM25 + Sparse 하이브리드(36ms)는 BM25 + Dense(107ms)보다 3배 빠르면서 동등하거나 더 나은 성능을 제공합니다.

개발된 모델은 HuggingFace에서 `sewoong/korean-neural-sparse-encoder-base-klue-large`로 공개되어 있으며, OpenSearch의 rank_features 또는 sparse_vector 타입과 함께 사용할 수 있습니다.

## 참고 자료

- SPLADE v2: Formal et al., "SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval" (2021)
- klue/roberta-large: Park et al., "KLUE: Korean Language Understanding Evaluation" (2021)
- opensearch-neural-sparse-encoding-multilingual-v1: OpenSearch Project (2024)
- Amazon Titan Text Embeddings v2: AWS Documentation
- Reciprocal Rank Fusion: Cormack et al., SIGIR 2009
