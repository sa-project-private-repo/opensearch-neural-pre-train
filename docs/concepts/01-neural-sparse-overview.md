# Neural Sparse 검색 개념

Neural Sparse Search는 키워드의 중요도를 학습하는 머신러닝 기반 검색 방식입니다. 기존의 키워드 검색과 의미 기반 검색의 장점을 모두 갖춘 하이브리드 접근법입니다.

## 1. Neural Sparse 검색의 정의 및 목적

### Sparse Retrieval이란?

Sparse Retrieval은 문서와 쿼리를 **높은 차원의 벡터로 표현**하되, **대부분의 값이 0인** 벡터(sparse vector)를 사용하는 검색 방식입니다.

**핵심 특성:**
- **Sparse**: 어휘 크기(vocab size)만큼의 차원을 가지나, 의미 있는 값은 극히 드물어 희소(sparse)
- **Interpretable**: 0이 아닌 각 차원이 특정 토큰을 대표하므로 해석 가능
- **Lexical + Semantic**: 키워드 기반 검색 + 의미 확장 (synonym expansion)

### 작동 원리

```
Input: "당뇨병 치료 방법"
       ↓
[Tokenization]
당뇨 / 병 / 치료 / 방법 / 의약품 / ...

       ↓
[Transformer Encoder - XLM-RoBERTa]
Hidden States 학습

       ↓
[Neural Sparse Head - 학습된 투영층]
각 토큰의 중요도 점수 계산

       ↓
Output Sparse Representation:
{
  "당뇨병": 2.847,
  "치료": 2.156,
  "방법": 1.892,
  "의료": 1.423,
  "당뇨": 1.289,
  ...
  (나머지는 0.0)
}
```

### 단어 중요도의 원리 (IDF 가중치)

Neural Sparse 모델은 **Inverse Document Frequency (IDF)** 개념을 활용하여 중요한 토큰에 높은 점수를 부여합니다:

- **고 IDF 토큰** (희귀): "당뇨병", "서울", "맛있는" → 높은 활성화 점수
- **저 IDF 토큰** (공통): "은", "는", "의" → 낮거나 0에 가까운 점수

**학습 메커니즘:**
```
L_flops = λ_flops × Σ(activation × idf_weight)
         + λ_stopword × Σ(stopword_activation)
```

모델은 다음을 동시에 학습합니다:
1. 의미 유사 쿼리와 문서의 유사도를 높임
2. 불용어(stopword)의 활성화를 억제
3. 드물고 의미 있는 토큰의 활성화를 강조

---

## 2. 검색 방법 비교 테이블

| 항목 | BM25 | Dense | Neural Sparse |
|------|------|-------|---------------|
| **표현 방식** | 어휘 기반 가중치 (TF-IDF) | 조밀 벡터 (Dense Vector) | 희소 벡터 (Sparse Vector) |
| **어휘 크기** | 전체 어휘 (고정) | 128-384 차원 | 어휘 크기 (250K) |
| **차원 수** | 고정 | 작음 (128-384) | 매우 큼 (250K) |
| **0이 아닌 값** | 매우 적음 | 모든 차원 | 매우 적음 |
| **검색 속도** | 매우 빠름 | 느림 (k-NN 필요) | 빠름 (BM25 수준) |
| **메모리** | 매우 효율적 | 효율적 | 효율적 |
| **의미 이해** | 미흡 | 우수 | 우수 |
| **키워드 매칭** | 정확 | 부정확 | 정확 + 의미 확장 |
| **해석 가능성** | 높음 | 낮음 | 높음 |
| **동의어 처리** | 불가 (별도 사전 필요) | 우수 | 우수 (학습됨) |
| **조사/어미 처리** | 미흡 | 우수 | 우수 |
| **한국어 특화** | 좋지 않음 | 좋음 | 매우 좋음 |

### 검색 시나리오별 추천

| 시나리오 | 추천 방식 | 이유 |
|---------|---------|------|
| 정확한 키워드 매칭 | BM25 | 완벽한 키워드 일치 필요 |
| 의미 유사성 | Dense | 순수 의미 기반 |
| **키워드 + 의미** | **Neural Sparse** | **양쪽 장점 모두** |
| 한국 도메인 (의료, 법률, IT) | Neural Sparse | 전문 용어 정확도 |
| 다언어 검색 | Neural Sparse | XLM-RoBERTa 250K vocab |

---

## 3. V26 벤치마크 결과

### V26 성능 지표

V26은 OpenSearch Korean Neural Sparse Model의 최신 버전입니다.

#### Recall 성능

| 메트릭 | V26 Neural Sparse | BM25 | BGE-M3 Dense | 개선율 |
|--------|-------------------|------|--------------|--------|
| **Recall@1** | **40.7%** | 30.0% | 37.1% | +35.7% vs BM25 |
| **Recall@5** | **51.4%** | 45.2% | 48.3% | +13.7% vs BM25 |
| **Recall@10** | 76.9% | 70.1% | 74.2% | +9.8% vs BM25 |

#### 순위 성능

| 메트릭 | V26 | BM25 | BGE-M3 |
|--------|-----|------|--------|
| **MRR (Mean Reciprocal Rank)** | **0.4555** | 0.3612 | 0.4189 |
| **nDCG@10** | **0.5234** | 0.4721 | 0.5018 |

### V26 vs V25 개선 사항

| 메트릭 | V25 | V26 | 개선도 |
|--------|-----|-----|--------|
| Recall@1 | 28.2% | 40.7% | **+44.3%** ⬆️ |
| Recall@5 | 38.1% | 51.4% | **+34.9%** ⬆️ |
| Semantic Ratio* | 73.2% | 95.8% | **+30.9%** ⬆️ |
| Stopword Activation | 1.2 | 0.18 | **-85%** ⬇️ |

*Semantic Ratio: 상위 10개 토큰 중 실제 의미 있는 토큰의 비율

### BM25 대비 우위 시나리오

**V26이 BM25를 이기는 경우:**

1. **동의어/유사어 검색**
   - 쿼리: "당뇨 관리"
   - BM25: 정확 일치만 찾음 (당뇨, 관리)
   - V26: 당뇨병, 혈당, 인슐린, 수치 등 의미 관련 토큰 활성화

2. **도메인 용어 인식**
   - 쿼리: "심근경색"
   - BM25: "심근", "경색" 분리 검색
   - V26: 의료 용어 패턴 인식, 유사 질환(뇌졸중, 협심증) 동시 검색

3. **문맥 이해**
   - 쿼리: "비행기 탑승수속"
   - BM25: "비행기", "탑승", "수속" 모두 포함된 문서만
   - V26: 항공사, 탑승권, 체크인, 라운지 등 관련 개념 포함

4. **오타/표기 변이**
   - 쿼리: "인터넷뱅킹" vs 문서: "인터넷 뱅킹"
   - BM25: 매칭 실패
   - V26: 의미 관계로 매칭 성공

**BM25가 여전히 우수한 경우:**

- 정확한 모델/상품명 검색 (e.g., "Galaxy S24")
- 숫자/코드 검색 (e.g., "ISO-9001")
- 극도로 특수한 용어

---

## 4. 적합한 사용 시나리오

### 1. 키워드 중심 검색 (Keyword-Centric Search)

```
시나리오: 의료 검색 플랫폼
쿼리: "폐암 초기 증상"

V26 활성 토큰:
- "폐암" (IDF: 4.2)
- "증상" (IDF: 2.8)
- "암" (IDF: 3.1)
- "초기" (IDF: 2.3)
- "진단" (IDF: 3.5) ← 의미 확장
- "검진" (IDF: 3.2) ← 의미 확장
- "호흡" (IDF: 3.1) ← 의미 확장

결과: BM25보다 완전한 관련 문서 검색
```

### 2. 도메인 용어 활용 (Domain Terminology)

**법률 문서 검색:**
```
쿼리: "부당 해고 손해배상"
V26 자동 인식:
- 부당해고, 부당징계
- 손해배상, 위자료
- 근로자, 고용주
- 소송, 중재
```

**의료 전문 문서:**
```
쿼리: "당뇨병 관리"
V26 활성화:
- 당뇨, 혈당
- 인슐린, 메트포민
- HbA1c (당화혈색소)
- 식이 요법, 운동 요법
```

### 3. 해석 가능성이 필요한 경우 (Explainability Required)

**규제 및 감시:**
```
질문: "왜 이 문서가 검색되었나?"
V26 답변: "당뇨병(2.85), 치료(2.16), 약물(1.89) 토큰이 일치"
→ 쿼리와의 명확한 매칭 기준 제시

Dense 모델: "유사도 0.76" (이유 불명)
→ 설명 불가능
```

**사용자 신뢰:**
```
V26: 검색 결과의 각 단어가 왜 매칭되었는지 설명 가능
→ 투명성 증대, 신뢰도 향상
```

### 4. 한국어 텍스트 검색 (Korean-Specific)

```
쿼리: "서울 맛있는 한식당"

조사/어미 처리 (BM25 실패):
- "서울" vs "서울의" (조사)
- "맛있는" vs "맛있다" (어미)

V26 동의어 인식:
- 맛있는 → 맛이 좋은, 맛나는, 풍미
- 한식당 → 한식 레스토랑, 정식당
- 서울 → 서울시, 서울권
```

---

## 5. 한국어 검색에서의 이점

### 5.1 조사(Particles) 및 어미(Endings) 처리

한국어는 문법적 기능을 조사와 어미로 표현합니다:

```
같은 의미의 다양한 형태:
- 당뇨병 (기본형)
- 당뇨병을 (목적격 조사)
- 당뇨병이 (주격 조사)
- 당뇨병은 (화제 조사)
- 당뇨병이 있는 (관형사형 + 조사)
- 당뇨병이 걸렸다 (서술형)
```

**BM25의 문제:**
```
쿼리: "당뇨병 치료"
찾은 문서: "당뇨병의 치료", "당뇨병을 치료", "당뇨병 치료법"
→ 모두 동일한 의미이지만 표면형 다름
→ BM25는 정확 일치만 강조
```

**V26의 해결:**
```
X-LM RoBERTa + Neural Sparse Head
→ 조사/어미 제거한 의미 단위로 토큰화
→ "당뇨병" 개념을 모든 변형에 동일하게 가중
→ 표기 변이에 강건함
```

### 5.2 XLM-RoBERTa의 250K 어휘 활용

**어휘 크기 비교:**

| 모델 | 어휘 크기 | 한국어 커버리지 | 특징 |
|------|---------|---------------|------|
| BERT-base (EN) | 30K | 극도로 낮음 | 영어 최적 |
| KoBERT | 32K | 중간 | 한국어 최적 |
| **XLM-RoBERTa** | **250K** | 높음 | 다국어 균형 |

**V26의 장점:**
```
XLM-RoBERTa 250K 어휘:
- 한국어: ~50K (부분어휘, 복합어, 용어)
- 영문: ~80K (domain terms, technical vocabulary)
- 기타 언어: 나머지

→ 한국 의료/법률/IT 용어 충분히 포함
→ 영문 용어(약물명, 질병명)도 동시 처리
→ 다국어 문서 검색 가능
```

### 5.3 IDF 가중치를 통한 Stopword 제어

**한국어 불용어 문제:**

```
불용어 (Stopword):
- 조사: 은, 는, 이, 가, 을, 를, 에, 에서, 와, 으로, ...
- 어미: 다, 하다, 되다, 있다, ...
- 관사/대명사: 이, 그, 저, 무엇, 누가, ...
- 수사: 한, 두, 세, 많은, 적은, ...
```

**문제점:**

```
쿼리: "당뇨병 치료"
단순 TF-IDF (BM25):
- "당뇨병" (TF: 3, IDF: 3.5)
- "치료" (TF: 3, IDF: 2.8)
- "은" (TF: 10, IDF: 0.1)  ← 매우 흔한 조사

스코어 계산:
- 당뇨병: 3 × 3.5 = 10.5
- 치료: 3 × 2.8 = 8.4
- "은": 10 × 0.1 = 1.0 (낮음)

→ "은"의 영향은 낮지만 여전히 노이즈
```

**V26의 해결책:**

```
학습된 Neural Sparse Head:
1. IDF 기반 페널티:
   - 고 IDF (희귀): 높은 활성화 가능
   - 저 IDF (공통): 활성화 억제

2. Stopword 명시적 페널티:
   - 177개 한국어 불용어 목록 유지
   - 불용어 활성화 × 15.0 페널티 (V26)
   - 강제로 0에 가깝게 억제

3. 특수 토큰 제외:
   - [CLS], [SEP], <s>, </s> 등은
   - IDF 정규화에서 제외
   - 고정 페널티 100.0 적용

결과: 의미 있는 토큰만 활성화
```

**실제 예시:**

```
문서: "당뇨병은 혈당 관리가 중요한 만성질환이다"

V26 활성 토큰:
- "당뇨병" (IDF: 4.2) → score: 2.85
- "혈당" (IDF: 3.8) → score: 2.15
- "관리" (IDF: 2.9) → score: 1.89
- "만성질환" (IDF: 4.1) → score: 2.34

V26 억제된 토큰:
- "은" (IDF: 0.5) → score: 0.0 ✓
- "가" (IDF: 0.4) → score: 0.0 ✓
- "인" (IDF: 0.6) → score: 0.0 ✓
- "다" (IDF: 0.3) → score: 0.0 ✓

Sparse Representation: {당뇨병: 2.85, 혈당: 2.15, 관리: 1.89, 만성질환: 2.34, ...}
```

### 5.4 한국 도메인 특화

**의료 도메인:**
```
질병명: 당뇨병, 고혈압, 뇌졸중, 심근경색
약물명: 메트포민, 리우글리제, 글리메피리드
증상: 다뇨, 다음, 피로, 체중 감소
검사: HbA1c, 공복혈당, 인슐린
```

**법률 도메인:**
```
법령: 근로기준법, 민법, 형법, 특별법
판례: 판시사항, 판단이유, 다수의견
용어: 청구, 항소, 상고, 소송, 중재, 조정
```

**IT 도메인:**
```
기술: API, REST, GraphQL, PostgreSQL
개념: 인증, 인가, 암호화, 해싱
도구: Docker, Kubernetes, Git, CI/CD
```

V26은 이러한 도메인 용어를 XLM-RoBERTa의 광범위한 어휘와 Neural Sparse Head의 학습을 통해 정확히 인식합니다.

---

## 6. 기술 사양 (V26)

### 모델 구성

| 항목 | 값 |
|------|-----|
| Base Model | `xlm-roberta-base` |
| Parameters | 278M |
| Hidden Size | 768 |
| Vocabulary Size | 250,002 |
| Max Sequence Length | 192 |
| Teacher Model | BAAI/bge-m3 |

### 손실 함수 (Loss Function)

```
L_total = λ_infonce × L_infonce      # InfoNCE Contrastive Loss
        + λ_self × L_self            # Self-Reconstruction Loss
        + λ_positive × L_positive    # Positive Alignment Loss
        + λ_flops × L_idf_flops      # IDF-Weighted Sparsity
        + λ_min_act × L_min_act      # Minimum Activation
        + λ_kd × L_kd                # Knowledge Distillation
```

### V26 하이퍼파라미터

| 파라미터 | V25 | V26 | 목적 |
|---------|-----|-----|------|
| `lambda_flops` | 0.002 | **0.010** | Sparse 강화 (5배) |
| `stopword_penalty` | 5.0 | **15.0** | 불용어 억제 (3배) |
| `idf_alpha` | 2.5 | **4.0** | IDF 곡선 가파르게 |
| `special_token_penalty` | - | **100.0** | 특수 토큰 제외 |
| `stopword_list_size` | 98 | **177** | 한국어 불용어 확충 |

---

## 참고 자료

### 관련 논문

- [SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) - 원본 SPLADE 아키텍처
- [SPLADE v2: Sparse Lexical and Expansion Model v2](https://arxiv.org/abs/2109.10086) - 개선된 버전
- [Learned Sparse Retrievers](https://arxiv.org/abs/2411.04403v2) - 최근 연구

### OpenSearch 문서

- [OpenSearch Neural Sparse Search](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/)
- [OpenSearch Neural Sparse Encoding Models](https://huggingface.co/opensearch-project)

### 관련 모델

- [XLM-RoBERTa Base](https://huggingface.co/xlm-roberta-base) - 기본 트랜스포머
- [BAAI BGE-M3](https://huggingface.co/BAAI/bge-m3) - Dense 교사 모델

---

## 다음 단계

- [02-training-workflow.md](./02-training-workflow.md) - 모델 학습 과정
- [03-inference-guide.md](./03-inference-guide.md) - 추론 및 배포
- [../guides/model-loading-guide.md](../guides/model-loading-guide.md) - 모델 로딩 가이드
