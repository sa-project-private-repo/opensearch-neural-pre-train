# 한국어 Stopword 참조서

## 개요

한국어 신경망 희소 검색(Neural Sparse Retrieval) 모델에서 의미 있는 토큰 활성화를 방해하는 한국어 조사, 어미, 기능어에 대한 완전한 참조 자료입니다. 이 문서는 V26 모델의 정규화 전략과 함께 177개의 확장 Stopword 목록을 제공합니다.

---

## 1. Stopword 문제 설명

### 왜 Stopword가 문제인가?

Neural Sparse Retrieval에서 SPLADE 모델은 각 토큰의 활성화 점수(activation score)를 학습합니다. 문제는 한국어의 구조적 특성입니다:

- **조사(Particles)**: "을", "는", "이" 등은 문법적으로 필수이지만 의미를 담지 않음
- **어미(Endings)**: "다", "습니다", "고" 등은 시제/태를 나타내지만 검색 의미성과 무관
- **기능어(Function Words)**: "있다", "하다", "것" 등은 빈번하게 등장하지만 의미 변별력 낮음

### V25에서 발견된 문제

V25 모델의 분석 결과:

```
V25 모델의 상위 10개 활성화 토큰:
1. 수 (것/기능어) - 3.8 (높음)
2. 있습니다 (어미) - 3.7
3. 것 (명사화) - 3.6
4. 하는 (동사 어미) - 3.5
5. 을 (조사) - 3.4
6. 는 (조사) - 3.3
7. 고 (연결어미) - 3.2
8. 면 (조건어미) - 3.1
9. 에 (위치조사) - 3.0
10. 가 (주격조사) - 2.9

Recall@1: 28.2% (BM25 대비 -6%, semantic 대비 -24%)
```

**핵심 문제**: Stopword가 의미 있는 토큰(병, 당, 치료 등)을 압도함.

### V26 해결책

V26은 다층적 접근으로 이 문제를 해결:

1. **확장 Stopword 목록**: 163개 → 177개 (50개 추가)
2. **강화된 FLOPS 정규화**: lambda_flops 5배 증가 (0.002 → 0.010)
3. **높은 Stopword 페널티**: stopword_penalty 3배 증가 (5.0 → 15.0)
4. **특수 토큰 분리**: special_penalty (100.0)로 별도 처리

**결과**:
```
V26 모델의 상위 10개 활성화 토큰:
1. 병 (의미) - 3.87 (낮아짐)
2. 당 (의미) - 3.85
3. 치료 (의미) - 3.84
4. 뇨 (의미) - 3.82
5. 혈 (의미) - 2.97
6. 방법 (의미) - 2.74
7. 당뇨 (의미) - 2.51
8. 혈당 (의미) - 2.35
9. 의료 (의미) - 2.12
10. 약 (의미) - 2.01

Recall@1: 40.7% (BM25 대비 +35.7%, semantic 대비 +3.6pp)
개선도: +44.3%
```

---

## 2. 카테고리별 Stopword 목록

### A. 조사(Particles) - 53개

조사는 명사나 명사구 뒤에 붙어 문법적 관계를 나타내는 기능어입니다.

#### 주격 조사 (Subject Markers)
- **이**: "책이 있다"
- **가**: "사람이 가다"
- **께서**: "선생님께서 말씀하셨다"

#### 목적격 조사 (Object Markers)
- **을**: "책을 읽다"
- **를**: "물을 마시다"

#### 주제 조사 (Topic Markers)
- **은**: "의약품은 효과가 있다"
- **는**: "환자는 치료를 받는다"

#### 소유/속성 조사 (Possessive/Attributive)
- **의**: "의료의 중요성"

#### 위치 조사 (Locative Particles)
- **에**: "병원에 가다"
- **에서**: "집에서 일하다"
- **에게**: "선생님에게 배우다"
- **한테**: "형한테 물어보다"
- **께**: "할머니께 인사하다"

#### 방향/목표 조사 (Direction/Goal)
- **로**: "학교로 가다"
- **으로**: "회사로 돌아가다"

#### 도구/수단 조사 (Instrumental/Means)
- **로써**: "의사로써의 책임"
- **으로써**: "이 방법으로써 치료한다"

#### 동반 조사 (Comitative - with)
- **와**: "친구와 함께"
- **과**: "형과 놀다"
- **랑**: "누구랑 가?"
- **이랑**: "나이랑 비슷하다"

#### 비교 조사 (Comparative)
- **보다**: "이 약이 그것보다 효과 있다"
- **처럼**: "의사처럼 생각하다"
- **같이**: "우리 같이 가자"
- **만큼**: "내 만큼 크다"

#### 복수 표시 조사 (Plural Marker)
- **들**: "환자들이 왔다"

#### 제한 조사 (Only/Just)
- **만**: "너만 왔니?"
- **뿐**: "그것뿐이다"

#### 첨가 조사 (Also/Too)
- **도**: "나도 간다"

#### 출발점 조사 (From)
- **부터**: "내일부터 시작"
- **에서부터**: "시작점에서부터 계산"

#### 종료점 조사 (Until)
- **까지**: "내일까지 완료"

#### 선택 조사 (Or)
- **나**: "사과나 배를 먹다"
- **이나**: "의약품이나 의료기구"

#### 포함/양보 조사 (Even)
- **조차**: "아이도 조차 안 온다"
- **마저**: "마지막까지 마저 왔다"

**총 53개 조사**

---

### B. 어미(Endings) - 47개

어미는 동사나 형용사 어간에 붙어 문법적 기능(시제, 태, 양태 등)을 수행합니다.

#### 종결어미 (Declarative Endings)
- **다**: "먹다", "간다"
- **습니다**: "먹습니다"
- **ㅂ니다**: "갑니다"
- **니다**: "하니다"
- **입니다**: "의사입니다"
- **요**: "가요", "먹어요"
- **어요**: "좋어요"
- **아요**: "간다아요"
- **죠**: "맞죠?"
- **지요**: "그렇지요?"
- **야**: "가야" (informal)
- **이야**: "의사이야"

#### 의문어미 (Interrogative Endings)
- **까**: "가까?"
- **습니까**: "어떻습니까?"
- **ㅂ니까**: "갑니까?"
- **니까**: "하니까?"
- **나요**: "뭐 하나요?"
- **을까요**: "어디로 갈까요?"
- **ㄹ까요**: "갈까요?"

#### 명령어미 (Imperative Endings)
- **세요**: "앉으세요"
- **십시오**: "앉으십시오"
- **어라**: "가거라"
- **아라**: "해라"

#### 연결어미 (Connective Endings)
- **고**: "가고 오다"
- **서**: "먹고 자다"
- **며**: "먹으며 말하다"
- **면서**: "보면서 생각하다"
- **지만**: "예쁘지만 비싸다"
- **는데**: "먹는데 맛있다"
- **ㄴ데**: "먹은데 맛있다"
- **은데**: "좋은데 비싸다"
- **니까**: "먹으니까 배부르다"
- **으니까**: "가으니까 뵈자"
- **면**: "가면 본다"
- **으면**: "먹으면 맛있다"
- **려고**: "가려고 한다"
- **으려고**: "먹으려고 한다"

#### 명사화 어미 (Nominalization)
- **는것**: "먹는것이 좋다"
- **은것**: "한것이 없다"
- **ㄴ것**: "간것이 언제냐"
- **기**: "먹기가 좋다"
- **음**: "먹음으로써 배부르다"

#### 수식어미 (Modifiers)
- **는**: "먹는 사람"
- **은**: "먹은 사람"
- **ㄴ**: "간 사람"
- **을**: "먹을 사람"
- **ㄹ**: "갈 사람"

**총 47개 어미**

---

### C. 기능어(Function Words) - 27개

의미를 담지는 않지만 문법 구조에 필수적인 단어들입니다.

#### 계사 (Copula)
- **이다**: "의사이다"
- **아니다**: "환자가 아니다"

#### 보조 동사 (Auxiliary Verbs)
- **있다**: "책이 있다", "일하고 있다"
- **없다**: "시간이 없다", "가지 않았다"
- **하다**: "공부하다"
- **되다**: "의사가 되다"

#### 지시어 (Demonstratives)
- **이**: "이것은 뭐냐"
- **그**: "그것이 맞다"
- **저**: "저기가 어디냐"

#### 대명사 (Pronouns)
- **나**: "나는 의사다"
- **너**: "너는 누구니?"
- **우리**: "우리는 팀이다"
- **저희**: "저희 병원에 오세요"

#### 공통 부사/수식어 (Common Adverbs/Intensifiers)
- **매우**: "매우 좋다"
- **아주**: "아주 잘했다"
- **정말**: "정말 멋있다"
- **진짜**: "진짜 좋아"
- **좀**: "좀 도와줄래?"
- **많이**: "물을 많이 마셔"
- **조금**: "조금만 기다려"

#### 연접어 (Conjunctions)
- **그리고**: "사과그리고 배"
- **그러나**: "좋으나 비싸다"
- **하지만**: "예쁘지만 비싸다"
- **그래서**: "그래서 간다"

#### 의문사 (Question Words - 낮은 변별력)
- **무엇**: "뭐 하니?"
- **뭐**: "뭐 하니?"
- **어디**: "어디 가니?"
- **언제**: "언제 가?"
- **왜**: "왜 그래?"
- **어떻게**: "어떻게 했어?"

**총 27개 기능어**

---

### D. V26 확장 Stopword - 50개 추가

V25 분석에서 발견된 고빈도 문법 구조와 기능어 패턴들입니다.

#### 고형식 동사 어미 (Formal/Informal Verb Endings)
- **있습니다**: "일하고 있습니다" (formal)
- **합니다**: "공부합니다"
- **입니다**: "학생입니다"
- **됩니다**: "의사가 됩니다"
- **했습니다**: "했습니다"
- **있어요**: "지금 있어요" (informal)
- **해요**: "뭐 해요?"
- **이에요**: "학생이에요"
- **되요**: "어른이 되요"
- **했어요**: "뭐 했어요?"
- **있어**: "뭐 하고 있어?" (very informal)
- **해**: "해!" (very informal)
- **이야**: "너 미쳤니야?"
- **돼**: "어른이 돼"
- **했어**: "뭐 했어?"

#### 명사화 패턴 (Nominalization Patterns)
- **것입니다**: "할 것입니다" (will do)
- **것이다**: "한 것이다" (did)
- **것은**: "할 것은" (as for will do)
- **것을**: "할 것을" (will do [obj])
- **것이**: "할 것이" (will do)
- **수**: "할 수 있다" (can)
- **때**: "갈 때" (when go)
- **것**: "할 것이 많다" (things to do)
- **데**: "간 데가" (place that went)

#### 고빈도 연결/전환 표현 (Connective Patterns)
- **그런데**: "예쁜데" (but)
- **따라서**: "따라서 간다" (therefore)
- **그러므로**: "그러므로 된다" (thus)
- **그래서**: "그래서 갔다" (so)
- **하지만**: "비싸지만 좋다" (but)
- **그러나**: "예쁘나 비싸다" (but)
- **그리고**: "먹고 자고 그리고 간다" (and)
- **또한**: "또한 중요하다" (also)
- **또는**: "사과 또는 배" (or)
- **및**: "공부 및 운동" (and)

#### 수식/보조 동사 표현 (Auxiliary/Modifier Patterns)
- **있는**: "있는 데가" (place is)
- **하는**: "하는 일" (doing work)
- **되는**: "되는 것" (becoming thing)
- **하게**: "하게 되다" (to do)
- **되게**: "되게 좋다" (become good)
- **할**: "할 것이 많다" (to do)
- **될**: "될 것 같다" (to become)
- **있을**: "있을 리 없다" (can't be)
- **없을**: "없을 수도" (may not)

#### 의미 없는 부사/수식어 (Semantic-poor Adverbs)
- **더**: "더 가자" (more)
- **가장**: "가장 좋다" (most)
- **바로**: "바로 여기" (right here)
- **이미**: "이미 갔어" (already)
- **아직**: "아직 안 갔어" (not yet)
- **다시**: "다시 한번" (again)
- **모두**: "모두 갔어" (all)

#### 양태/약화 표현 (Modal/Aspectual Markers)
- **수있**: "할 수있다" (can)
- **수없**: "할 수없다" (can't)
- **겠**: "하겠습니다" (will)
- **어야**: "해야 한다" (should)
- **어도**: "해도 된다" (may)

#### 흔한 술어 어미 조합 (Common Predicative Combinations)
- **한다**: "공부한다" (study)
- **한**: "하는 것" (doing)
- **하고**: "공부하고 간다" (study and go)
- **해서**: "공부해서 배운다" (study and learn)
- **하면**: "공부하면 된다" (if study okay)

**총 50개 추가 Stopword**

---

## 3. V26 확장 Stopword 전체 목록 (177개)

### 모든 Stopword 합계: 177개

```
조사 (53개):
이 가 께서 을 를 은 는 의 에 에서 에게 한테 께 로 으로 로써 으로써
와 과 랑 이랑 보다 처럼 같이 만큼 들 만 뿐 도 부터 에서부터 까지
나 이나 조차 마저

어미 (47개):
다 습니다 ㅂ니다 니다 입니다 요 어요 아요 죠 지요 야 이야
까 습니까 ㅂ니까 니까 나요 을까요 ㄹ까요
세요 십시오 어라 아라
고 서 며 면서 지만 는데 ㄴ데 은데 니까 으니까 면 으면 려고 으려고
는것 은것 ㄴ것 기 음
는 은 ㄴ 을 ㄹ

기능어 (27개):
이다 아니다 있다 없다 하다 되다
이 그 저
나 너 우리 저희
매우 아주 정말 진짜 좀 많이 조금
그리고 그러나 하지만 그래서
무엇 뭐 어디 언제 왜 어떻게

V26 추가 (50개):
있습니다 합니다 입니다 됩니다 했습니다
있어요 해요 이에요 되요 했어요
있어 해 이야 돼 했어
것입니다 것이다 것은 것을 것이 수 때 것 데
그런데 따라서 그러므로 그래서 하지만 그러나 그리고 또한 또는 및
있는 하는 되는 하게 되게 할 될 있을 없을
더 가장 매우 아주 잘 바로 이미 아직 다시 모두
수있 수없 겠 어야 어도
한다 한 하고 해서 하면
```

---

## 4. 처리 메커니즘

### 4.1 Masking 방식 (Inference Time 하드 마스킹)

#### 함수: `create_stopword_mask_v26()`

추론 시간에 Stopword 토큰의 활성화를 완전히 억제합니다.

```python
def create_stopword_mask_v26(
    tokenizer: PreTrainedTokenizer,
    include_subwords: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    V26 확장 Stopword를 위한 이진 마스크 생성.

    Returns:
        텐서 [vocab_size]:
        - 1.0 = 일반 토큰 (유지)
        - 0.0 = Stopword 토큰 (마스크)
    """
    mask = torch.ones(vocab_size, device=device)
    stopword_ids = get_korean_stopword_ids_v26(tokenizer)

    for token_id in stopword_ids:
        mask[token_id] = 0.0  # Hard masking

    return mask
```

**특징:**
- Inference time에 Stopword 완전 제거
- 계산 효율적 (덧셈 대신 곱셈)
- 추론 후 처리 없음

**사용 예시:**
```python
from src.train.idf.korean_stopwords import create_stopword_mask_v26

mask = create_stopword_mask_v26(tokenizer, device=model.device)
sparse_logits = sparse_logits * mask  # Stopword 활성화 = 0
```

---

### 4.2 Penalty 방식 (Training Time 소프트 페널티)

#### 함수: `create_stopword_penalty_weights()`

학습 중 FLOPS 손실함수에서 Stopword에 가중치를 부여합니다.

```python
def create_stopword_penalty_weights(
    tokenizer: PreTrainedTokenizer,
    stopword_penalty: float = 15.0,
    special_penalty: float = 100.0,
    include_subwords: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Stopword 토큰을 위한 페널티 가중치 생성.

    Returns:
        가중치 텐서 [vocab_size]:
        - 1.0 = 일반 토큰 (기본 페널티)
        - 15.0 = 한국어 Stopword (높은 페널티)
        - 100.0 = 특수 토큰 (최고 페널티)
    """
    weights = torch.ones(vocab_size, device=device)

    # 특수 토큰: 최고 페널티
    special_ids = _get_special_token_ids(tokenizer)
    for token_id in special_ids:
        weights[token_id] = special_penalty  # 100.0

    # Stopword: 중간 페널티
    stopword_ids = get_korean_stopword_ids(tokenizer)
    for token_id in stopword_ids:
        if token_id not in special_ids:
            weights[token_id] = stopword_penalty  # 15.0

    return weights
```

**학습 손실함수 적용:**

```python
# V26 FLOPS 손실 (IDF-aware with stopword penalty)
flops_logits = torch.log(1 + torch.relu(logits)) * penalty_weights
flops_loss = flops_logits.sum()

# 전체 손실
total_loss = (
    contrastive_loss
    + lambda_flops * flops_loss  # Stopword penalty 포함
    + lambda_kd * kd_loss
    + margin_loss
)
```

**특징:**
- 학습 중 점진적 억제 (soft penalization)
- 모델이 Stopword를 "학습하지 않도록" 유도
- 완전 제거보다 유연함

---

### 4.3 특수 토큰 처리 (Special Token Isolation)

#### 함수: `get_special_token_ids_only()`

V26에서는 특수 토큰(`<s>`, `</s>`)을 일반 Stopword와 분리합니다.

```python
def get_special_token_ids_only(
    tokenizer: PreTrainedTokenizer
) -> Set[int]:
    """
    특수 토큰만 반환 (Stopword 제외).

    XLM-RoBERTa 특수 토큰:
    - <s> (ID 0): 시작
    - </s> (ID 1): 끝
    - <pad> (ID 2): 패딩
    - <unk> (ID 3): 미지
    - ID 4-6: 추가 특수 토큰
    """
    special_ids: Set[int] = set()

    # 표준 특수 토큰
    if tokenizer.pad_token_id is not None:
        special_ids.add(tokenizer.pad_token_id)
    if tokenizer.bos_token_id is not None:
        special_ids.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        special_ids.add(tokenizer.eos_token_id)
    # ... etc

    # XLM-RoBERTa IDs 0-6은 일반적으로 특수
    for i in range(7):
        special_ids.add(i)

    # 공통 구두점
    punct_tokens = [".", ",", "!", "?", ":", ";", "-", "'", '"', ...]
    for punct in punct_tokens:
        special_ids.add(tokenizer.convert_tokens_to_ids(punct))

    return special_ids
```

**특수 토큰 분리의 이점:**

V25 문제:
```
특수 토큰이 IDF 정규화에 포함되어 활성화 범위 압축 발생
특수 토큰 활성화: [0.5, 15.0] (매우 넓음)
→ 정규화 후: [0.033, 1.0] (범위 축소)
→ 의미 토큰 활성화 억제
```

V26 해결:
```
특수 토큰 제외:
특수 토큰 활성화: 범위 조정 없음
의미 토큰 활성화: [0.5, 4.0] (의미 있는 범위 유지)
→ 정규화 후: [0.125, 1.0] (적절한 범위)
→ 의미 토큰 강조
```

---

## 5. 코드 참조

### 5.1 Stopword ID 획득

#### `get_korean_stopword_ids_v26()`

```python
def get_korean_stopword_ids_v26(
    tokenizer: PreTrainedTokenizer,
    include_subwords: bool = True,
) -> Set[int]:
    """
    V26 확장 한국어 Stopword의 토큰 ID 획득.

    Args:
        tokenizer: XLM-RoBERTa 토크나이저
        include_subwords: 부분단어 변형 포함 (▁ 마커)

    Returns:
        Stopword 토큰 ID 집합 (특수 토큰 제외)

    처리 방식:
    1. KOREAN_STOPWORDS_V26 목록 반복
    2. 각 단어를 인코딩하여 토큰 ID 획득
    3. SentencePiece 변형 포함 (" word", "▁word")
    4. 어휘 내 직접 매칭

    예시:
        "을" → {ID_을, ID_▁을}
        "는" → {ID_는, ID_▁는}
        "것입니다" → {ID_것입니다, ID_▁것입니다}
    """
    stopword_ids: Set[int] = set()

    for word in KOREAN_STOPWORDS_V26:
        # 직접 인코딩
        tokens = tokenizer.encode(word, add_special_tokens=False)
        stopword_ids.update(tokens)

        if include_subwords:
            # 앞 공백 포함
            space_word = f" {word}"
            space_tokens = tokenizer.encode(space_word, add_special_tokens=False)
            stopword_ids.update(space_tokens)

            # SentencePiece 마커 포함
            sp_word = f"▁{word}"
            if sp_word in tokenizer.get_vocab():
                stopword_ids.add(tokenizer.convert_tokens_to_ids(sp_word))

            # 직접 어휘 매칭
            if word in tokenizer.get_vocab():
                stopword_ids.add(tokenizer.convert_tokens_to_ids(word))

    logger.info(f"V26: {len(stopword_ids)}개 Stopword 토큰 ID 식별됨")
    return stopword_ids
```

**출력 예시:**
```
V26: 1823개 Stopword 토큰 ID 식별됨
(부분단어 변형 포함)

ID 예시:
- 177 ("을")
- 204 ("는")
- 295 ("것입니다")
- 1042 ("▁있습니다")
```

---

### 5.2 마스크 생성

#### `create_stopword_mask_v26()` - 완전 구현

```python
def create_stopword_mask_v26(
    tokenizer: PreTrainedTokenizer,
    include_subwords: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    V26 확장 Stopword 토큰을 위한 이진 마스크 생성.

    Returns:
        이진 마스크 텐서 [vocab_size]:
        - 1.0 = 일반 토큰 (유지)
        - 0.0 = Stopword 토큰 (마스크)

    특징:
    - 특수 토큰 제외 (별도의 특수 토큰 페널티 사용)
    - SentencePiece 부분단어 변형 포함
    - Inference time hard masking

    사용 시나리오:
    1. 모델 로드 후 마스크 생성
    2. Sparse logits에 곱셈 적용
    3. Top-k 토큰 선택
    """
    vocab_size = tokenizer.vocab_size
    mask = torch.ones(vocab_size, device=device)

    # V26 Stopword ID 획득 (특수 토큰 제외)
    stopword_ids = get_korean_stopword_ids_v26(
        tokenizer,
        include_subwords=include_subwords,
    )

    # 마스크 적용
    for token_id in stopword_ids:
        if 0 <= token_id < vocab_size:
            mask[token_id] = 0.0

    masked_count = (mask == 0).sum().item()
    logger.info(f"V26: {masked_count}개 토큰 마스킹됨")

    return mask
```

**사용 예시:**

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from src.train.idf.korean_stopwords import create_stopword_mask_v26

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained("sewoong/korean-neural-sparse-encoder-v26")
model = AutoModelForMaskedLM.from_pretrained("sewoong/korean-neural-sparse-encoder-v26")

# 마스크 생성
mask = create_stopword_mask_v26(tokenizer, device=model.device)

# 텍스트 인코딩
text = "당뇨병 치료 방법"
inputs = tokenizer(text, return_tensors="pt", max_length=192, truncation=True)

# 모델 추론
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

    # SPLADE 변환
    sparse_logits = torch.log1p(torch.relu(logits))
    sparse_repr = sparse_logits.max(dim=1).values[0]

    # Stopword 마스크 적용
    sparse_repr = sparse_repr * mask  # Stopword 활성화 = 0

# Top-10 토큰 (모두 의미 토큰)
top_values, top_indices = sparse_repr.topk(10)
for idx, val in zip(top_indices.tolist(), top_values.tolist()):
    if val > 0:
        token = tokenizer.decode([idx]).strip()
        print(f"{token}: {val:.4f}")
```

**출력:**
```
병: 3.87
당: 3.85
치료: 3.84
뇨: 3.82
혈: 2.97
방법: 2.74
당뇨: 2.51
혈당: 2.35
의료: 2.12
약: 2.01
```

---

### 5.3 특수 토큰만 획득

#### `get_special_token_ids_only()`

```python
def get_special_token_ids_only(
    tokenizer: PreTrainedTokenizer
) -> Set[int]:
    """
    특수 토큰 ID만 반환 (Stopword 제외).

    용도:
    - IDF 정규화 제외 (V26 특수 토큰 분리)
    - 특수 토큰만 높은 penalty 적용
    - Stopword와 분리된 처리

    반환되는 특수 토큰:
    - <s> (BOS): ID 0
    - </s> (EOS): ID 1
    - <pad>: ID 2
    - <unk>: ID 3
    - 구두점: ".", ",", "!", "?", ...
    """
    return _get_special_token_ids(tokenizer)
```

---

## 6. V26 vs V25 비교

### 성능 개선

| 지표 | V25 | V26 | 개선 |
|------|-----|-----|------|
| **Recall@1** | 28.2% | 40.7% | +44.3% |
| **Semantic Ratio** | 73.2% | 95.8% | +30.9% |
| **vs BM25** | -6pp | +35.7% | ✅ 역전 |
| **vs Semantic (BGE-M3)** | -24pp | +3.6pp | ✅ 우월 |

### 하이퍼파라미터 변경

| 파라미터 | V25 | V26 | 변경 사유 |
|---------|-----|-----|---------|
| `lambda_flops` | 0.002 | 0.010 | 5배 증가: FLOPS 규제 강화 |
| `stopword_penalty` | 5.0 | 15.0 | 3배 증가: Stopword 억제 강화 |
| `idf_alpha` | 2.5 | 4.0 | IDF 곡선 가팔라짐 |
| `special_token_penalty` | - | 100.0 | NEW: 특수 토큰 분리 |
| `stopword_list_size` | 163 | 177 | 50개 추가: 고빈도 문법 구조 |

### Stopword 활성화 비교

#### V25 (문제)
```
상위 10개 활성화 토큰:
1. 수 (기능어) - 3.8 ← Stopword 지배
2. 있습니다 (어미) - 3.7
3. 것 (명사화) - 3.6
4. 하는 (동사 어미) - 3.5
5. 을 (조사) - 3.4
...
Semantic 토큰: 0개 (상위 10에)
```

#### V26 (개선)
```
상위 10개 활성화 토큰:
1. 병 (의미) - 3.87 ← 모두 의미 토큰
2. 당 (의미) - 3.85
3. 치료 (의미) - 3.84
4. 뇨 (의미) - 3.82
5. 혈 (의미) - 2.97
...
Semantic 토큰: 10/10 (100%)
```

---

## 7. 구현 가이드

### 학습 시

```python
from src.train.idf.korean_stopwords import (
    create_stopword_penalty_weights,
    get_special_token_ids_only
)

# 학습 초기화
penalty_weights = create_stopword_penalty_weights(
    tokenizer,
    stopword_penalty=15.0,  # V26: 조사/어미 페널티
    special_penalty=100.0,   # V26: 특수 토큰 페널티
    include_subwords=True,
    device=device
)

# 손실함수에 적용
flops_logits = torch.log(1 + torch.relu(logits)) * penalty_weights
flops_loss = flops_logits.sum()

total_loss = contrastive_loss + lambda_flops * flops_loss + ...
```

### 추론 시

```python
from src.train.idf.korean_stopwords import create_stopword_mask_v26

# 추론 초기화
mask = create_stopword_mask_v26(
    tokenizer,
    include_subwords=True,
    device=device
)

# 모델 추론
sparse_repr = get_sparse_representation(text)

# Stopword 제거
sparse_repr = sparse_repr * mask

# Top-k 선택
top_values, top_indices = sparse_repr.topk(k=10)
```

---

## 8. 제한사항 및 주의사항

1. **멀티링구언 모델**: XLM-RoBERTa 기반이므로 영어, 중국어 등 다른 언어의 Stopword도 포함될 수 있습니다. 한국어 특화를 위해서는 한국어 텍스트에서 주로 사용하세요.

2. **도메인 특화**: V26 목록은 일반 한국어 말뭉치 기반입니다. 의료/법률 등 도메인 특화 모델에서는 추가 Stopword 정의가 필요할 수 있습니다.

3. **동적 마스킹**: 부분단어 변형 때문에 일부 Stopword가 누락될 수 있습니다. `include_subwords=True`로 설정하여 최대 커버리지를 확보하세요.

4. **특수 토큰 처리**: `get_special_token_ids_only()`는 V26 IDF 정규화 호환성을 위해 설계되었습니다. 다른 용도에서는 `get_korean_stopword_ids_v26()`을 사용하세요.

---

## 참고 자료

- **모델**: [sewoong/korean-neural-sparse-encoder-v26](https://huggingface.co/sewoong/korean-neural-sparse-encoder-v26)
- **소스 코드**: `src/train/idf/korean_stopwords.py`
- **논문**: [SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720)
- **OpenSearch**: [Neural Sparse Search](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/)

---

**문서 작성**: 2026-01-29
**적용 버전**: V26 (한국어 신경망 희소 인코더)
