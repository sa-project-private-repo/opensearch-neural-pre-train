# SPLADE-doc 아키텍처

XLM-RoBERTa 기반 한국어 Neural Sparse 모델의 핵심 아키텍처를 설명한다.

## SPLADE-doc 핵심 아키텍처

### Base Model: XLM-RoBERTa

XLM-RoBERTa를 선택한 이유:

| 특성 | XLM-RoBERTa | mBERT/KoBERT |
|------|-------------|--------------|
| Vocabulary Size | 250,002 | ~50,000 |
| Sparse Vector 표현력 | 5x 풍부 | 제한적 |
| 다국어 정렬 | 우수 | 보통 |
| SPLADE 연구 검증 | 다수 | 제한적 |

### MLM Head를 통한 Vocabulary Expansion

SPLADE-doc의 핵심은 MLM(Masked Language Modeling) Head를 활용한 vocabulary expansion이다.

```
입력 문서: "서울 맛집 추천"
    ↓
XLM-RoBERTa Encoder
    ↓
MLM Head (vocab 전체에 대한 logits 생성)
    ↓
확장된 토큰: "서울", "맛집", "추천", "음식", "레스토랑", "식당", ...
```

입력에 없는 관련 토큰도 활성화되어 semantic matching이 가능해진다.

## 컴포넌트 상세

### 1. XLMRobertaForMaskedLM

```python
from transformers import XLMRobertaForMaskedLM

# MLM head 포함 로드
self.model = XLMRobertaForMaskedLM.from_pretrained(model_name)
self.transformer = self.model.roberta  # encoder 부분

# Forward: [batch, seq_len] -> [batch, seq_len, vocab_size]
outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
logits = outputs.logits  # MLM logits
```

### 2. Sparsity Induction: log1p(ReLU(logits))

SPLADE의 핵심 수식:

```
sparse_repr = log(1 + ReLU(logits))
```

```python
# ReLU: 음수 제거 -> 희소성 유도
# log1p: 큰 값 압축 + 0에서 미분 가능
self.relu = nn.ReLU()
sparse_scores = torch.log1p(self.relu(logits))
```

효과:
- **ReLU**: 음수 logits 제거로 희소성 확보
- **log(1+x)**:
  - 큰 값 압축 (수치 안정성)
  - x=0에서 기울기 1 유지 (학습 용이)
  - 희소 벡터에 적합한 스케일링

### 3. Max Pooling: sparse_scores.max(dim=1)

시퀀스 전체에서 각 vocabulary 토큰의 최대 활성화 값 추출:

```python
# sparse_scores: [batch, seq_len, vocab_size]
# attention mask 적용
mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
sparse_scores = sparse_scores * mask

# Max pooling over sequence positions
# sparse_repr: [batch, vocab_size]
sparse_repr, _ = sparse_scores.max(dim=1)
```

각 토큰이 문서 내 어느 위치에서든 높게 활성화되면 최종 표현에 반영된다.

### 4. Token Weights: Per-Position Importance

분석 및 시각화를 위한 위치별 중요도:

```python
# 각 위치에서 가장 높은 vocabulary 활성화 값
token_weights = sparse_scores.max(dim=-1).values  # [batch, seq_len]
```

## Model 변형

### SPLADEDocXLMR (기본 모델)

```python
class SPLADEDocXLMR(nn.Module):
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        dropout: float = 0.1,
        use_mlm_head: bool = True,
    ):
        super().__init__()

        if use_mlm_head:
            # MLM head로 vocabulary expansion
            self.model = XLMRobertaForMaskedLM.from_pretrained(model_name)
        else:
            # 입력 토큰만 활성화 (경량 버전)
            self.transformer = XLMRobertaModel.from_pretrained(model_name)
            self.token_importance = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
```

`use_mlm_head=False` 옵션:
- 입력 토큰만 활성화 (vocabulary expansion 없음)
- 메모리/계산 효율적
- 단순 키워드 매칭에 적합

### SPLADEDocXLMRWithIDF (IDF 가중치 적용)

불용어 억제 및 희소 토큰 보존을 위한 IDF 가중치:

```python
class SPLADEDocXLMRWithIDF(SPLADEDocXLMR):
    def __init__(
        self,
        idf_weights: Optional[torch.Tensor] = None,
        idf_alpha: float = 2.0,  # 패널티 강도
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.idf_alpha = idf_alpha
        if idf_weights is not None:
            self.register_buffer('idf_weights', idf_weights)

    def forward(self, input_ids, attention_mask, apply_idf=True):
        sparse_repr, token_weights = super().forward(input_ids, attention_mask)

        if apply_idf and self.idf_weights is not None:
            sparse_repr = sparse_repr * self.idf_weights.unsqueeze(0)

        return sparse_repr, token_weights
```

IDF 패널티 계산:

```python
def compute_idf_penalty_weights(self, idf_values: torch.Tensor) -> torch.Tensor:
    # IDF 정규화: [0, 1]
    normalized_idf = (idf_values - idf_values.min()) / (
        idf_values.max() - idf_values.min() + 1e-8
    )

    # 패널티: w_j = exp(-alpha * normalized_idf_j)
    # High IDF -> 낮은 패널티 (희소 토큰 보존)
    # Low IDF -> 높은 패널티 (불용어 억제)
    penalty_weights = torch.exp(-self.idf_alpha * normalized_idf)

    return penalty_weights
```

### create_splade_xlmr (팩토리 함수)

```python
def create_splade_xlmr(
    model_name: str = "xlm-roberta-base",
    use_idf: bool = True,
    idf_weights: Optional[torch.Tensor] = None,
    dropout: float = 0.1,
    use_mlm_head: bool = True,
) -> Union[SPLADEDocXLMR, SPLADEDocXLMRWithIDF]:
    """
    Args:
        model_name: "xlm-roberta-base" 또는 "xlm-roberta-large"
        use_idf: IDF 가중치 사용 여부
        idf_weights: 사전 계산된 IDF 가중치 [vocab_size]
        dropout: Dropout 비율
        use_mlm_head: MLM head 사용 (vocabulary expansion)

    Returns:
        SPLADEDocXLMR 또는 SPLADEDocXLMRWithIDF 인스턴스
    """
    if use_idf:
        return SPLADEDocXLMRWithIDF(...)
    else:
        return SPLADEDocXLMR(...)
```

## V26 모델 사양

| Parameter | Value |
|-----------|-------|
| Base Model | xlm-roberta-base |
| Vocab Size | 250,002 |
| Hidden Dim | 768 |
| Num Layers | 12 |
| Attention Heads | 12 |
| Total Params | ~278M |
| Dropout | 0.1 |
| MLM Head | True |
| IDF Weighting | Optional |

## 코드 참조

### forward() 메서드 흐름

```python
def forward(
    self,
    input_ids: torch.Tensor,       # [batch, seq_len]
    attention_mask: torch.Tensor,  # [batch, seq_len]
) -> Tuple[torch.Tensor, torch.Tensor]:

    # 1. MLM 순전파
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [batch, seq_len, vocab_size]

    # 2. Sparsity Induction
    sparse_scores = torch.log1p(self.relu(logits))  # [batch, seq_len, vocab_size]

    # 3. Padding 마스킹
    mask = attention_mask.unsqueeze(-1).float()
    sparse_scores = sparse_scores * mask

    # 4. Max Pooling
    sparse_repr, _ = sparse_scores.max(dim=1)  # [batch, vocab_size]

    # 5. Token Weights (분석용)
    token_weights = sparse_scores.max(dim=-1).values  # [batch, seq_len]

    return sparse_repr, token_weights
```

### encode_documents() 사용법

배치 문서 인코딩:

```python
from transformers import AutoTokenizer
from src.model.splade_xlmr import create_splade_xlmr

# 모델 초기화
model = create_splade_xlmr(use_idf=False)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# 문서 토크나이징
docs = ["서울 맛집 추천", "부산 여행 코스"]
inputs = tokenizer(docs, padding=True, truncation=True, return_tensors="pt")

# 인코딩
with torch.no_grad():
    sparse_repr = model.encode_documents(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    # sparse_repr: [2, 250002]
```

### get_top_k_tokens() 분석 도구

Sparse 표현의 상위 활성화 토큰 분석:

```python
# 단일 문서 인코딩
doc = "서울 강남 맛집 추천해주세요"
inputs = tokenizer(doc, return_tensors="pt")

with torch.no_grad():
    sparse_repr, _ = model.forward(
        inputs["input_ids"],
        inputs["attention_mask"]
    )

# 상위 20개 토큰 추출
top_tokens = model.get_top_k_tokens(
    sparse_repr=sparse_repr[0],  # 첫 번째 문서
    tokenizer=tokenizer,
    k=20,
)

# 출력 예시
# {
#     "서울": 3.45,
#     "강남": 3.21,
#     "맛집": 3.89,
#     "추천": 2.76,
#     "음식": 1.92,      # vocabulary expansion으로 추가됨
#     "레스토랑": 1.54,  # vocabulary expansion으로 추가됨
#     ...
# }
```

## 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Document                            │
│                    "서울 맛집 추천해주세요"                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    XLM-RoBERTa Tokenizer                         │
│                   [CLS] 서울 맛집 추천 해 주세요 [SEP]            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 XLM-RoBERTa Encoder (12 layers)                  │
│                      Hidden: [batch, seq, 768]                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MLM Head                                  │
│                  Logits: [batch, seq, 250002]                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│               Sparsity Induction: log(1 + ReLU(x))              │
│                Sparse Scores: [batch, seq, 250002]              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Max Pooling (dim=1)                          │
│              Sparse Representation: [batch, 250002]             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Output Sparse Vector                         │
│     {서울: 3.45, 맛집: 3.89, 추천: 2.76, 음식: 1.92, ...}        │
│                    (대부분 0, 수십~수백 개만 활성화)               │
└─────────────────────────────────────────────────────────────────┘
```

## 참고

- 코드: `src/model/splade_xlmr.py`
- 원본 SPLADE 논문: [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)
- XLM-RoBERTa: [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)
