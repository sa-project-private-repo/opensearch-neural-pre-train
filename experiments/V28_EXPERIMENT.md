# V28 Neural Sparse Model Experiment

## 실험 목표

V26 모델의 두 가지 핵심 문제를 해결:

1. **다국어 토큰 누출**: 한국어 쿼리에서 비한국어 토큰이 91% 활성화
2. **정적 희소 표현**: 동일 키워드가 문맥과 무관하게 항상 같은 토큰 활성화

## 실험 구성

### V28a: Korean Language Filtering

**가설**: 비한국어 토큰에 높은 페널티를 부과하면 한국어 토큰 비율이 증가하고 검색 성능이 향상될 것이다.

**방법**:
- XLM-RoBERTa vocab에서 한국어/영어 토큰 분류
- 비한국어 토큰에 100.0 penalty weight 적용
- `lambda_language = 0.5`로 언어 필터링 손실 추가

**구현 파일**:
| 파일 | 설명 |
|------|------|
| `src/train/idf/korean_tokens.py` | 한국어 토큰 ID 빌더 |
| `src/model/losses.py` | `SPLADELossV28` 클래스 |
| `src/train/config/v28.py` | V28 설정 |

**성공 기준**:
| 지표 | V26 현재 | V28a 목표 |
|------|----------|----------|
| 한국어 토큰 비율 | ~10% | >85% |
| 다국어 노이즈 | 91% | <5% |
| Ko-StrategyQA Recall@1 | 30.4% | >35% |

---

### V28b: Context-Gated Sparse Expansion (CGSE)

**가설**: 문서 전체 컨텍스트를 기반으로 토큰별 가중치를 조절하면 문맥 의존적 희소 표현이 가능해진다.

**문제 예시**:
```
현재 (V26): "점심 메뉴" → 항상 동일한 토큰 활성화

목표 (V28b):
- "출근했는데 점심 메뉴" → 회사, 직장인, 비빔밥
- "학교를 갔는데 점심 메뉴" → 학생, 급식, 도시락
```

**아키텍처**:
```
Document → Transformer → Hidden States
                ↓
         Context Pooling (Multi-Head Attention)
                ↓
         Context Vector [batch, 768]
                ↓
         Gate Projection (768 → 256 → 250002)
                ↓
         Context Gate [batch, vocab_size] (Sigmoid)
                ↓
         MLM Logits × Context Gate → Gated Logits
                ↓
         ReLU + log(1+x) → Max Pooling → Sparse Vector
```

**구현 파일**:
| 파일 | 설명 |
|------|------|
| `src/model/splade_xlmr.py` | `ContextGate`, `SPLADEDocContextGated` |

**하이퍼파라미터**:
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `context_gate_hidden` | 256 | Gate MLP 히든 크기 |
| `context_attention_heads` | 4 | Attention heads 수 |
| `context_gate_dropout` | 0.1 | Dropout rate |

**성공 기준**:
| 지표 | V28a | V28b 목표 |
|------|------|----------|
| 컨텍스트 구분율 | 0% | >60% |
| Ko-StrategyQA Recall@1 | >35% | >40% |
| 동일 키워드 토큰 오버랩 | 100% | <50% |

---

## 학습 설정

### configs/train_v28.yaml 주요 설정

```yaml
model:
  model_class: "SPLADEDocContextGated"
  use_context_gate: true
  context_gate_hidden: 256
  context_attention_heads: 4

loss:
  # V26 기반
  lambda_flops: 0.010
  special_token_penalty: 100.0
  stopword_penalty: 15.0

  # V28a: 언어 필터링
  enable_language_filtering: true
  non_korean_penalty: 100.0
  lambda_language: 0.5

  # V28b: 컨텍스트 게이트
  use_context_gate: true
  use_context_aware_kd: true
  context_kd_weight: 1.0

training:
  num_epochs: 25
  batch_size: 24
  learning_rate: 3.0e-5
  gradient_accumulation_steps: 4
```

### Curriculum Learning Phases

| Phase | Epochs | Temperature | lambda_kd | 설명 |
|-------|--------|-------------|-----------|------|
| 1 | 1-8 | 0.08 | 2.5 | Foundation with BGE-M3 teacher |
| 2 | 9-17 | 0.05 | 1.5 | Balanced training with hard negatives |
| 3 | 18-25 | 0.04 | 0.8 | Hard negative refinement with context gate |

---

## 검증 방법

### 1. 한국어 토큰 비율 검증

```bash
make eval-v28-language
```

```python
from benchmark.encoders import NeuralSparseEncoder
encoder = NeuralSparseEncoder('outputs/train_v28/best_model')
result = encoder.encode(['토니 베넷의 중간 이름은?'])[0]
korean_ratio = sum(1 for t in result if is_korean(t)) / len(result)
print(f'Korean token ratio: {korean_ratio:.2%}')
```

### 2. 컨텍스트 구분율 검증

```bash
make eval-v28-context
```

```python
queries = [
    "출근했는데 점심 메뉴 추천해줘",
    "학교를 갔는데 점심 메뉴 추천해줘"
]
results = encoder.encode(queries)
overlap = set(results[0].keys()) & set(results[1].keys())
total = set(results[0].keys()) | set(results[1].keys())
overlap_ratio = len(overlap) / len(total)
print(f'Token overlap: {overlap_ratio:.2%}')  # 목표: <50%
```

### 3. Ko-StrategyQA 벤치마크

```bash
make benchmark-ko-strategyqa-v28
```

---

## 실험 실행

### V27 완료 후 자동 실행

```bash
nohup ./scripts/run_v28_after_v27.sh > outputs/v28_auto.log 2>&1 &
```

### 수동 실행

```bash
# V28 학습 시작
make train-v28

# 백그라운드 실행
make train-v28-bg

# 체크포인트에서 재개
make train-v28-resume
```

---

## 버전 히스토리

| 버전 | 핵심 변경 | 상태 |
|------|----------|------|
| V22 | KoBERT backbone, curriculum learning | 완료 |
| V24 | XLM-RoBERTa + BGE-M3 teacher | 완료 |
| V25 | IDF-aware FLOPS | 완료 |
| V26 | Enhanced IDF + Special Token Fix | 완료 |
| V27 | Travel Domain Data | 진행중 |
| **V28a** | **Korean Language Filtering** | **코드 완료, 학습 대기** |
| **V28b** | **Context-Gated Expansion** | **코드 완료, 학습 대기** |

---

## 참고 자료

- [SPLADE Paper](https://arxiv.org/abs/2107.05720)
- [OpenSearch Neural Sparse](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1)
- [BGE-M3 Teacher Model](https://huggingface.co/BAAI/bge-m3)
