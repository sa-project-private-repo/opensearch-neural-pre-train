# Loss Functions

Neural Sparse 모델 학습에 사용되는 손실 함수들의 상세 설명.

## 개요

SPLADE 기반 Neural Sparse 모델은 다중 목표 손실 함수를 사용하여 학습한다. 각 손실 함수는 모델의 특정 능력을 최적화하며, 가중치를 통해 균형을 조절한다.

총 손실:

$$
\mathcal{L}_{\text{total}} = \sum_{i} \lambda_i \mathcal{L}_i
$$

---

## 1. InfoNCE Loss (Contrastive Learning)

### 정의

Noise Contrastive Estimation 기반 대조 학습 손실. In-batch negatives를 활용하여 배치 내 다른 샘플을 자동으로 negative로 사용한다.

### 수식

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(q, p^+) / \tau)}{\sum_{i=1}^{N} \exp(\text{sim}(q, p_i) / \tau)}
$$

여기서:
- $q$: 앵커(쿼리) 표현
- $p^+$: 정답 문서 표현
- $p_i$: 배치 내 모든 문서 (in-batch negatives)
- $\tau$: temperature 파라미터
- $\text{sim}$: 유사도 함수 (cosine 또는 dot product)

### 구현 세부사항

```python
# Cosine similarity 정규화
anchor_repr = F.normalize(anchor_repr, p=2, dim=-1)
positive_repr = F.normalize(positive_repr, p=2, dim=-1)

# 유사도 행렬 계산
sim_matrix = torch.mm(anchor_repr, positive_repr.t()) / temperature

# Cross-entropy loss (대각선이 positive)
labels = torch.arange(batch_size, device=anchor_repr.device)
loss = F.cross_entropy(sim_matrix, labels)
```

### 역할

- 의미적으로 유사한 쿼리-문서 쌍을 가깝게 학습
- 배치 크기가 클수록 더 많은 negative 샘플 제공
- Temperature가 낮을수록 분포가 sharp해짐 (hard negative에 집중)

### V26 설정

| 파라미터 | 값 | 설명 |
|---------|---|------|
| `temperature` | 0.07 | InfoNCE softmax temperature |
| `similarity` | cosine | 유사도 측정 방식 |

---

## 2. Self-Reconstruction Loss

### 정의

입력 토큰에 대한 활성화를 유도하는 손실. 모델이 실제 입력에 존재하는 토큰을 sparse representation에서 높게 활성화하도록 학습한다.

### 수식

$$
\mathcal{L}_{\text{self}} = \text{BCE}(\mathbf{r}, \mathbf{t})
$$

여기서:
- $\mathbf{r} \in \mathbb{R}^{|V|}$: sparse representation (vocab 크기)
- $\mathbf{t} \in \{0, 1\}^{|V|}$: 입력 토큰 위치를 나타내는 타겟 벡터

타겟 벡터 생성:
$$
t_j = \begin{cases} 1 & \text{if token } j \text{ appears in input} \\ 0 & \text{otherwise} \end{cases}
$$

### 구현 세부사항

```python
# 타겟 생성: 입력 토큰 위치에 1
target = torch.zeros_like(sparse_repr)
target.scatter_add_(1, input_ids, attention_mask.float())
target = target.clamp(max=1.0)

# Binary cross-entropy
loss = F.binary_cross_entropy_with_logits(sparse_repr, target)
```

### 역할

- Inference-free sparse retrieval의 핵심 (입력 토큰만 활성화)
- 토큰 재현율(token recall) 보장
- Vocabulary expansion과의 균형 필요

---

## 3. Positive Activation Loss

### 정의

앵커가 positive 문서의 토큰을 활성화하도록 유도. Cross-document term alignment를 통해 유의어 및 관련 용어 학습을 촉진한다.

### 수식

$$
\mathcal{L}_{\text{positive}} = -\frac{1}{|T_p|} \sum_{j \in T_p} r_j^{(a)}
$$

여기서:
- $T_p$: positive 문서의 토큰 집합
- $r_j^{(a)}$: 앵커의 j번째 토큰 활성화 값

### 구현 세부사항

```python
# Positive 문서 토큰 마스크 생성
positive_mask = torch.zeros_like(anchor_repr)
positive_mask.scatter_add_(1, positive_input_ids, positive_attention_mask.float())
positive_mask = (positive_mask > 0).float()

# Positive 토큰 위치의 평균 활성화 (높을수록 좋음)
positive_activations = anchor_repr * positive_mask
mean_positive_activation = positive_activations.sum(dim=1) / num_positive_tokens

# 손실: -activation (최대화를 위해 음수)
loss = -mean_positive_activation.mean()
```

### 역할

- 유의어 학습: "서울" 쿼리가 "수도" 토큰도 활성화
- Term expansion 유도
- 쿼리-문서 간 어휘 갭 해소

---

## 4. Triplet Margin Loss

### 정의

앵커-positive 유사도가 앵커-negative 유사도보다 margin 이상 크도록 학습.

### 수식

$$
\mathcal{L}_{\text{margin}} = \max(0, m - \text{sim}(a, p) + \text{sim}(a, n))
$$

여기서:
- $a$: 앵커 표현
- $p$: positive 표현
- $n$: negative 표현
- $m$: margin 값

### 구현 세부사항

```python
# Cosine similarity
anchor_repr = F.normalize(anchor_repr, p=2, dim=-1)
positive_repr = F.normalize(positive_repr, p=2, dim=-1)
negative_repr = F.normalize(negative_repr, p=2, dim=-1)

pos_sim = (anchor_repr * positive_repr).sum(dim=-1)
neg_sim = (anchor_repr * negative_repr).sum(dim=-1)

# Triplet loss
loss = F.relu(margin - pos_sim + neg_sim).mean()
```

### V26 설정

V26에서는 **비활성화** (`lambda_margin=0.0`). InfoNCE가 충분한 대조 학습 신호를 제공하므로 중복 회피.

| 파라미터 | 값 | 설명 |
|---------|---|------|
| `margin` | 0.3 | Cosine similarity margin |
| `lambda_margin` | 0.0 | 비활성화 |

---

## 5. IDF-Aware FLOPS Loss

### 정의

희소성(sparsity) 규제를 위한 핵심 손실. 토큰별 활성화를 억제하되, IDF 가중치를 적용하여 정보량에 따라 차등 페널티를 부여한다.

### 수식

$$
\mathcal{L}_{\text{FLOPS}} = \sum_{j=1}^{|V|} w_j \cdot |\bar{a}_j| + \beta \sum_{j=1}^{|V|} w_j \cdot \bar{a}_j^2
$$

여기서:
- $\bar{a}_j$: j번째 토큰의 배치 평균 활성화
- $w_j$: IDF 기반 페널티 가중치
- $\beta$: L2 페널티 비율

### IDF 가중치 계산

$$
w_j = \exp(-\alpha \cdot \text{idf}_{\text{norm}, j})
$$

IDF 정규화:
$$
\text{idf}_{\text{norm}, j} = \frac{\text{idf}_j - \text{idf}_{\min}}{\text{idf}_{\max} - \text{idf}_{\min}}
$$

| IDF 값 | 토큰 유형 | 페널티 |
|--------|---------|--------|
| 높음 (희귀) | 의미 토큰 (서울, 맛있는) | 낮음 |
| 낮음 (빈번) | 불용어 (은/는/이/가) | 높음 |

### 구현 세부사항

```python
# 배치 평균 활성화
mean_activation = sparse_repr.mean(dim=0)  # [vocab_size]

# 가중 L1 + L2 페널티
weighted_l1 = (penalty_weights * mean_activation.abs()).sum()
weighted_l2 = (penalty_weights * (mean_activation ** 2)).sum()

loss = weighted_l1 + beta * weighted_l2
```

### V26 특수 토큰 처리

V25의 근본 문제: 특수 토큰(`<s>`, `</s>`)의 df=0으로 인해 IDF=max, 정규화 후 penalty=min이 되어 정규화 범위가 압축됨.

**V26 해결책:**

1. **특수 토큰 제외**: IDF min/max 계산에서 특수 토큰 제외
2. **고정 페널티**: 특수 토큰에 `special_penalty=100.0` 고정 적용
3. **더 급격한 IDF 곡선**: `alpha=4.0` (V25: 2.5)

```python
# 특수 토큰 제외하고 정규화 범위 계산
real_mask = torch.ones(len(idf_weights), dtype=torch.bool)
for tid in special_token_ids:
    real_mask[tid] = False

real_idf = idf_weights[real_mask]
idf_min = real_idf.min()
idf_max = real_idf.max()

# 특수 토큰에 고정 페널티 적용
for tid in special_token_ids:
    penalty_weights[tid] = special_penalty  # 100.0
```

### V26 설정

| 파라미터 | V25 | V26 | 변경 |
|---------|-----|-----|------|
| `lambda_flops` | 0.002 | 0.010 | 5x 증가 |
| `idf_alpha` | 2.5 | 4.0 | 더 급격한 곡선 |
| `special_penalty` | - | 100.0 | 신규 |
| `stopword_penalty` | 5.0 | 15.0 | 3x 증가 |
| `beta` | 0.3 | 0.3 | 동일 |

---

## 6. Minimum Activation Loss

### 정의

Top-k 활성화가 임계값 이상을 유지하도록 강제. 모델이 near-zero 출력(garbage output)을 생성하는 것을 방지한다.

### 수식

$$
\mathcal{L}_{\text{min}} = \text{ReLU}\left(\theta - \frac{1}{k} \sum_{i=1}^{k} a_{(i)}\right)
$$

여기서:
- $a_{(i)}$: i번째로 높은 활성화 값
- $k$: top-k 개수
- $\theta$: 최소 활성화 임계값

### 구현 세부사항

```python
# Top-k 활성화 추출
top_k_values, _ = torch.topk(sparse_repr, k=top_k, dim=-1)

# Top-k 평균
mean_top_k = top_k_values.mean(dim=-1)

# 임계값 미달 시 페널티
loss = F.relu(min_activation - mean_top_k).mean()
```

### 역할

- FLOPS 규제로 인한 과도한 희소화 방지
- 최소한의 의미 있는 활성화 보장
- 검색 성능 하한선 유지

### V26 설정

| 파라미터 | 값 | 설명 |
|---------|---|------|
| `top_k` | 5 | 상위 5개 토큰 |
| `min_activation` | 0.5 | 최소 활성화 임계값 |

---

## 7. Knowledge Distillation (BGE-M3 Teacher)

### 정의

Dense teacher 모델(BGE-M3)의 유사도 분포를 sparse student 모델이 모방하도록 학습.

### 수식

$$
\mathcal{L}_{\text{KD}} = T^2 \cdot \text{KL}\left(\text{softmax}\left(\frac{z_t}{T}\right) \| \text{softmax}\left(\frac{z_s}{T}\right)\right) + \alpha_{\text{MSE}} \cdot \text{MSE}(\tilde{z}_t, \tilde{z}_s)
$$

여기서:
- $z_t$: teacher 유사도 점수
- $z_s$: student 유사도 점수
- $T$: distillation temperature
- $\tilde{z}$: z-score 정규화된 점수
- $T^2$: gradient 크기 보정 계수

### 구현 세부사항

```python
# Temperature-scaled softmax
student_log_probs = F.log_softmax(student_scores / temperature, dim=-1)
teacher_probs = F.softmax(teacher_scores / temperature, dim=-1)

# KL divergence with T² scaling
kl_loss = (temperature ** 2) * F.kl_div(
    student_log_probs, teacher_probs, reduction="batchmean"
)

# Z-score normalized MSE (scale-invariant)
student_norm = (student_scores - student_scores.mean()) / (student_scores.std() + 1e-8)
teacher_norm = (teacher_scores - teacher_scores.mean()) / (teacher_scores.std() + 1e-8)
mse_loss = F.mse_loss(student_norm, teacher_norm)

# Combined
loss = alpha_kl * kl_loss + alpha_mse * mse_loss
```

### 역할

- Dense 모델의 semantic understanding 전이
- Sparse 모델의 초기 학습 안정화
- KL: 순위 분포 학습
- MSE: 절대적 유사도 스케일 학습

### V26 설정

| 파라미터 | 값 | 설명 |
|---------|---|------|
| `lambda_kd` | 2.0 | KD 손실 가중치 |
| `kd_temperature` | 3.0 | Softmax temperature |
| `alpha_kl` | 0.7 | KL 비율 |
| `alpha_mse` | 0.3 | MSE 비율 |
| `teacher_model` | BAAI/bge-m3 | Teacher 모델 |

---

## 8. Loss Weight 참조표

### 버전별 손실 가중치 비교

| Loss | V22 | V23 | V25 | V26 | 목적 |
|------|-----|-----|-----|-----|------|
| `lambda_infonce` | 2.0 | 2.5 | 3.0 | 3.0 | Contrastive learning |
| `lambda_self` | 4.0 | 1.0 | 0.5 | 0.5 | Input reconstruction |
| `lambda_positive` | 10.0 | 3.0 | 2.0 | 2.0 | Cross-document alignment |
| `lambda_margin` | 2.5 | 0.0 | 0.0 | 0.0 | Triplet margin (비활성화) |
| `lambda_flops` | 0.005 | 0.003 | 0.002 | **0.010** | Sparsity regulation |
| `lambda_min_act` | 1.0 | 1.0 | 1.0 | 1.0 | Activation floor |
| `lambda_kd` | - | 1.0 | 2.0 | 2.0 | Knowledge distillation |

### V26 핵심 변경사항

1. **FLOPS 가중치 5x 증가**: 0.002 -> 0.010
2. **IDF alpha 증가**: 2.5 -> 4.0 (더 급격한 페널티 곡선)
3. **특수 토큰 고정 페널티**: 100.0 (신규)
4. **Stopword 페널티 3x 증가**: 5.0 -> 15.0

---

## 9. Curriculum Learning 연계

손실 가중치는 curriculum phase에 따라 동적으로 조정된다.

### Phase별 가중치 변화 (V26)

| Phase | Epochs | `temperature` | `lambda_infonce` | `lambda_kd` | 목적 |
|-------|--------|---------------|------------------|-------------|------|
| 1 | 1-8 | 0.08 | 2.5 | 2.5 | Teacher guidance 강조 |
| 2 | 9-17 | 0.05 | 3.0 | 1.5 | 균형 학습 |
| 3 | 18-25 | 0.04 | 3.0 | 0.8 | Student 독립성 강화 |

---

## 10. 손실 함수 선택 가이드

### 문제별 권장 조정

| 문제 | 증상 | 권장 조정 |
|------|------|----------|
| 불용어 과다 활성화 | top-10에 조사/접속사 | `lambda_flops` 증가, `stopword_penalty` 증가 |
| 의미 토큰 억제 | 핵심 명사 활성화 낮음 | `idf_alpha` 감소, `lambda_min_act` 증가 |
| 토큰 재현율 저하 | 입력 토큰 누락 | `lambda_self` 증가 |
| 유의어 미학습 | 관련 토큰 비활성화 | `lambda_positive` 증가 |
| 학습 불안정 | loss 진동 | `lambda_kd` 증가, `kd_temperature` 증가 |
| 과적합 | val loss 발산 | `lambda_flops` 증가, `dropout` 증가 |

---

## 참고 자료

- SPLADE 논문: https://arxiv.org/abs/2109.10086
- InfoNCE 원문: https://arxiv.org/abs/1807.03748
- Knowledge Distillation: https://arxiv.org/abs/1503.02531
- 구현 코드: `src/model/losses.py`
