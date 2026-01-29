# V26 하이퍼파라미터 참고서

V26 모델은 XLM-RoBERTa 기반 SPLADE 희소 검색 모델로, IDF 기반 FLOPS 정규화와 향상된 불용어 처리를 특징으로 합니다.

## V26 기본 설정 개요

V26은 V25의 개선된 버전으로, 더 강력한 희소성 제약과 향상된 불용어 억제를 목표로 설계되었습니다.

| 카테고리 | 설정 항목 | 값 |
|---------|---------|-----|
| 모델 | 베이스 모델 | xlm-roberta-base |
| 모델 | Dropout | 0.1 |
| 모델 | 확장 모드 | mlm |
| 데이터 | 배치 크기 | 24 |
| 데이터 | 최대 길이 | 192 |
| 데이터 | 워커 수 | 4 |
| 학습 | 에폭 수 | 25 |
| 학습 | 학습률 | 3e-5 |
| 학습 | 혼합 정밀도 | bf16 |
| 손실 | FLOPS 가중치 | 0.010 |
| 손실 | 불용어 페널티 | 15.0 |
| 손실 | IDF 알파 | 4.0 |

---

## 1. 모델 설정

모델 구조와 인코더 관련 설정입니다.

### V24ModelConfig 기본값

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `name` | xlm-roberta-base | 사전학습된 변환기 모델 이름 |
| `dropout` | 0.1 | 정규화를 위한 드롭아웃 비율 (0.0~1.0) |
| `use_expansion` | True | MLM 기반 어휘 확장 사용 여부 |
| `expansion_mode` | mlm | 확장 모드: 'mlm' (MLM 헤드) 또는 'projection' (학습된 프로젝션) |
| `freeze_encoder_layers` | 0 | 동결할 인코더 레이어 수 (0 = 모두 학습) |

### 모델 선택 가이드

**xlm-roberta-base (권장)**
- 다국어 지원 (111개 언어)
- 한국어 포함
- 2억 5천만 파라미터
- V26의 표준 설정

**튜닝 권장사항**
- 한국어 전용: xlm-roberta-base 유지
- 성능 우선: xlm-roberta-large (3배 느림)
- 속도 우선: xlm-roberta-small (정확도 감소)

---

## 2. 학습 설정

학습 프로세스 관련 모든 하이퍼파라미터입니다.

| 파라미터 | 값 | 범위 | 설명 |
|---------|-----|------|------|
| `num_epochs` | 25 | 1~100 | 총 에폭 수 |
| `learning_rate` | 3e-5 | 1e-6~1e-4 | 초기 학습률 (AdamW) |
| `weight_decay` | 0.01 | 0.0~0.1 | L2 정규화 계수 |
| `warmup_ratio` | 0.1 | 0.0~0.5 | 워밍업 비율 (전체 스텝의 fraction) |
| `gradient_clip` | 1.0 | 0.5~2.0 | 최대 그래디언트 노름 (clipping용) |
| `gradient_accumulation_steps` | 4 | 1~16 | 그래디언트 누적 스텝 수 |
| `mixed_precision` | bf16 | fp32/fp16/bf16 | 혼합 정밀도 모드 |
| `save_every_n_epochs` | 5 | 1~10 | 체크포인트 저장 간격 (에폭) |
| `keep_last_n_checkpoints` | 3 | 1~10 | 유지할 최근 체크포인트 수 |

### 배치 크기 계산

**효과적인 배치 크기** = `batch_size × gradient_accumulation_steps × num_gpus`

V26 기본값:
```
Effective batch size = 24 × 4 × 1 = 96
```

다중 GPU 시:
```
Effective batch size = 24 × 4 × 8 = 768 (8개 GPU 기준)
```

### 학습률 스케줄

V26은 선형 워밍업 + 선형 감소 스케줄 사용:

- **워밍업**: 전체 스텝의 10% 동안 0 → 3e-5로 증가
- **감소**: 이후 3e-5 → 0으로 선형 감소
- **커리큘럼 학습 중**: 각 단계별 LR 승수 적용

### 혼합 정밀도 선택

| 옵션 | 메모리 절감 | 속도 | 정확도 | 권장 |
|-----|----------|------|-------|------|
| fp32 | 기준선 | 기준선 | 최고 | 작은 모델 |
| fp16 | 50% | 2배 빠름 | 약간 감소 | RTX 시리즈 |
| bf16 | 50% | 2배 빠름 | 동일 | A100, H100 (권장) |

---

## 3. 손실 함수 설정

손실 가중치 및 하이퍼파라미터입니다.

### 손실 가중치

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `lambda_infonce` | 3.0 | InfoNCE 대조 손실 가중치 |
| `lambda_self` | 0.5 | 자기재구성 손실 가중치 |
| `lambda_positive` | 2.0 | 양성 활성화 손실 가중치 |
| `lambda_margin` | 0.0 | 삼중항 마진 손실 가중치 (비활성화) |
| `lambda_flops` | 0.010 | IDF 기반 FLOPS 정규화 가중치 |
| `lambda_min_act` | 1.0 | 최소 활성화 손실 가중치 |
| `lambda_kd` | 2.0 | 지식 증류 손실 가중치 (커리큘럼 기반 동적) |

### 손실 하이퍼파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| `temperature` | 0.07 | 0.01~0.2 | InfoNCE 온도 (대조 손실 scaling) |
| `margin` | 0.3 | 0.0~1.0 | 코사인 유사도 마진 |
| `top_k` | 5 | 1~10 | 최소 활성화 손실용 top-k |
| `min_activation` | 0.5 | 0.0~1.0 | 최소 활성화 임계값 |
| `kd_temperature` | 3.0 | 1.0~5.0 | 지식 증류 온도 (소프트 레이블) |

---

## 4. V26 손실 설정 (V25와의 비교)

### V25 vs V26 주요 변경사항

| 파라미터 | V25 | V26 | 변경 비율 | 변경 사유 |
|---------|-----|-----|---------|---------|
| `lambda_flops` | 0.002 | 0.010 | 5배 증가 | 더 강력한 희소성 제약 |
| `stopword_penalty` | 5.0 | 15.0 | 3배 증가 | 불용어 억제 개선 |
| `idf_alpha` | 2.5 | 4.0 | 1.6배 증가 | IDF 곡선 상승화 |
| `special_token_penalty` | 미지원 | 100.0 | 신규 | 특수 토큰 명시적 억제 |
| `use_extended_stopwords` | 미지원 | True | 신규 | 확장 한국어 불용어 목록 |

### V26 IDF 설정

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `use_idf_weighting` | True | IDF 기반 페널티 활성화 (V26에서 필수) |
| `idf_alpha` | 4.0 | IDF 지수 감쇠 인수 (높을수록 곡선 상승화) |
| `idf_weights_path` | None | 사전계산된 IDF 가중치 경로. None이면 corpus에서 계산 |
| `recompute_idf` | False | 캐시 존재 시에도 IDF 재계산 강제 |
| `idf_smoothing` | bm25 | IDF 평활 방법: 'bm25' 또는 'standard' |

### V26 특수 토큰 설정

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `special_token_penalty` | 100.0 | 특수 토큰(<s>, </s>, [PAD], [UNK>)의 고정 페널티 |

V26에서 처음 도입됨. 특수 토큰은 IDF 정규화 범위에서 제외되고 명시적 페널티 적용.

### V26 불용어 설정

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `use_stopword_mask` | True | 추론 시 불용어 마스킹 활성화 |
| `stopword_penalty` | 15.0 | FLOPS 손실에서 불용어의 추가 페널티 승수 |
| `use_extended_stopwords` | True | 확장된 한국어 불용어 목록(KOREAN_STOPWORDS_V26) 사용 |

---

## 5. 커리큘럼 학습 스케줄

V26은 3단계 커리큘럼 학습을 사용합니다. 각 단계에서는 학습 목표와 데이터 구성이 다릅니다.

### 전체 스케줄

| 단계 | 에폭 | 온도 | lambda_infonce | LR 승수 | lambda_kd | 목표 |
|------|------|------|---------------|---------|-----------|------|
| 1 | 1-8 | 0.08 | 2.5 | 1.0 | 2.5 | BGE-M3 교사 안내로 기초 구축 |
| 2 | 9-17 | 0.05 | 3.0 | 0.5 | 1.5 | 어려운 음성 샘플로 균형 학습 |
| 3 | 18-25 | 0.04 | 3.0 | 0.25 | 0.8 | 어려운 음성 정제 |

### 단계별 세부 설정

#### 단계 1: 기초 구축 (Epoch 1-8)

**목표**: 전반적 표현 학습 + 교사 신호 활용

```
온도: 0.08 (완화된 대조)
lambda_infonce: 2.5 (낮은 대조 가중치)
lr_multiplier: 1.0 (최대 학습률 유지)
lambda_kd: 2.5 (높은 지식 증류)
```

**데이터 가중치**:
- single_term (단일 용어): 40%
- multi_term (다중 용어): 35%
- original (원본 데이터): 25%

**전략**:
- BGE-M3 교사 모델의 강한 지도
- 단순 쿼리부터 학습 시작
- 불용어 및 희소성 패턴 인식

#### 단계 2: 균형 학습 (Epoch 9-17)

**목표**: 어려운 음성 샘플 포함 + 교사 의존도 감소

```
온도: 0.05 (강화된 대조)
lambda_infonce: 3.0 (일반적 대조 가중치)
lr_multiplier: 0.5 (학습률 50% 감소)
lambda_kd: 1.5 (중간 지식 증류)
```

**데이터 가중치**:
- single_term: 30%
- multi_term: 35%
- hard_neg (어려운 음성): 35%

**전략**:
- 어려운 음성 샘플 도입
- 학습률 감소로 미세 조정
- 교사 신호의 점진적 감소
- 불용어 마스킹 강화

#### 단계 3: 정제 (Epoch 18-25)

**목표**: 어려운 음성에 대한 강력한 성능 달성

```
온도: 0.04 (가장 강한 대조)
lambda_infonce: 3.0 (일반적 대조 가중치)
lr_multiplier: 0.25 (학습률 75% 감소)
lambda_kd: 0.8 (약한 지식 증류)
```

**데이터 가중치**:
- hard_neg: 50%
- multi_term: 30%
- single_term: 20%

**전략**:
- 어려운 음성 중심 학습 (50%)
- 매우 낮은 학습률로 수렴
- 교사 신호 최소화
- 희소 표현 최종 정제

### 온도의 의미

- **높은 온도 (0.08)**: 대조 손실 분포 완화 (부드러운 확률)
- **낮은 온도 (0.04)**: 대조 손실 분포 강화 (날카로운 확률)

온도 감소 → 모델이 샘플 간 미묘한 차이에 더 집중

---

## 6. 데이터 설정

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `train_files` | data/v24.0/train_*.jsonl | 학습 데이터 파일 패턴 |
| `val_files` | data/v24.0/val.jsonl | 검증 데이터 파일 |
| `batch_size` | 24 | GPU 당 배치 크기 |
| `max_length` | 192 | 토큰화 최대 길이 |
| `num_workers` | 4 | 데이터 로딩 워커 수 |
| `prefetch_factor` | 2 | 워커당 프리페치 배치 수 |
| `pin_memory` | True | GPU 전송을 위한 메모리 고정 |

### 배치 크기 최적화

| GPU 메모리 | 권장 배치 크기 | 누적 스텝 | 효과적 배치 |
|-----------|-------------|---------|----------|
| 16GB | 12 | 4 | 48 |
| 24GB | 24 | 4 | 96 (V26 기본) |
| 40GB | 48 | 2 | 96 |
| 80GB | 96 | 1 | 96 |

### max_length 선택

- **128**: 빠른 학습, 긴 쿼리/문서 자르기 (성능 감소)
- **192**: 균형 (V26 기본, 권장)
- **256**: 정확도 향상, 메모리 40% 증가
- **384**: 최대 정확도, 메모리 2배 필요

---

## 7. 지식 증류 설정 (KDConfig)

다른 강력한 모델(예: BGE-M3)에서 지식을 전달받습니다.

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `enable_kd` | True | 지식 증류 활성화 |
| `teacher_model_name` | bge-m3 | 교사 모델 이름 |
| `kd_weight` | 동적 | 손실 가중치 (커리큘럼 단계별 조정) |
| `temperature` | 3.0 | 소프트 레이블 온도 |

**커리큘럼별 동적 KD 가중치**:
- Phase 1: 2.5 (높은 교사 의존도)
- Phase 2: 1.5 (중간 교사 의존도)
- Phase 3: 0.8 (낮은 교사 의존도)

---

## 8. 어려운 음성 마이닝 (HardNegativeConfig)

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `enable_hnm` | True | 어려운 음성 마이닝 활성화 |
| `hnm_strategy` | curriculum | 마이닝 전략: 'curriculum', 'all_gather', 'in_batch' |
| `top_k` | 16 | 마이닝할 상위 K개 음성 선택 |
| `use_score_debiasing` | True | 스코어 편향 제거 (중복 제거용) |

---

## 9. 튜닝 가이드

### 증상별 문제 해결

#### 과소 학습 (Underfitting)

**증상**:
- 검증 손실이 계속 감소
- 최종 성능이 낮음
- 에폭 종료 후에도 학습 가능성 있음

**해결 방법**:

| 조치 | 변경 사항 | 효과 |
|-----|---------|------|
| 에폭 증가 | 25 → 30 | 더 많은 학습 기회 |
| 학습률 증가 | 3e-5 → 5e-5 | 더 큰 업데이트 |
| 드롭아웃 감소 | 0.1 → 0.05 | 정규화 약화 |
| 배치 크기 감소 | 24 → 12 | 더 자주 업데이트 |
| gradient_accumulation 감소 | 4 → 2 | 더 자주 가중치 업데이트 |
| warmup_ratio 감소 | 0.1 → 0.05 | 더 빨리 최대 LR 도달 |

#### 과도 학습 (Overfitting)

**증상**:
- 훈련 손실 << 검증 손실
- 에폭 중반 이후 검증 성능 악화
- 최종 모델 성능 저하

**해결 방법**:

| 조치 | 변경 사항 | 효과 |
|-----|---------|------|
| 학습률 감소 | 3e-5 → 1e-5 | 더 느린 학습 (정제) |
| 드롭아웃 증가 | 0.1 → 0.2 | 더 강한 정규화 |
| weight_decay 증가 | 0.01 → 0.05 | L2 정규화 강화 |
| 배치 크기 증가 | 24 → 48 | 일반화 개선 |
| gradient_accumulation 증가 | 4 → 8 | 더 큰 효과적 배치 |
| Early stopping 추가 | 에폭 25 → 20 | 검증 성능 기준 조기 중단 |
| 에폭 감소 | 25 → 20 | 덜 학습 |

#### 불용어 누수 (Stopword Leakage)

**증상**:
- 결과에 "이다", "등", "그리고" 등 불용어 과다 포함
- SPLADE 점수에서 불용어가 중요 토큰으로 나타남

**해결 방법**:

| 조치 | 변경 사항 | 효과 |
|-----|---------|------|
| stopword_penalty 증가 | 15.0 → 25.0 | 불용어에 대한 페널티 강화 |
| special_token_penalty 증가 | 100.0 → 150.0 | 특수 토큰 억제 강화 |
| lambda_flops 증가 | 0.010 → 0.015 | 희소성 제약 강화 |
| idf_alpha 증가 | 4.0 → 5.0 | IDF 곡선 더 상승화 |
| use_stopword_mask 확인 | True | 추론 시 불용어 마스킹 활성 |
| 확장 불용어 확인 | use_extended_stopwords=True | 한국어 불용어 목록 최신화 |

#### 약한 검색 성능 (Weak Retrieval)

**증상**:
- nDCG@10 < 0.4
- 관련 문서를 상위에 배치하지 못함
- 검색 정확도 낮음

**해결 방법**:

| 조치 | 변경 사항 | 효과 |
|-----|---------|------|
| lambda_infonce 증가 | 3.0 → 4.0 | 대조 학습 강화 |
| temperature 감소 | 0.07 → 0.05 | 대조 신호 강화 |
| lambda_kd 증가 | Phase-specific → +0.5 | 교사 신호 강화 |
| 지식 증류 활성화 | enable_kd=True | 교사 모델 활용 |
| 어려운 음성 마이닝 | enable_hnm=True | 어려운 음성 학습 |
| 에폭 증가 | 25 → 30 | 더 많은 학습 |
| KD 온도 조정 | 3.0 → 2.0 | 더 날카로운 소프트 레이블 |

#### 높은 계산 비용 (High Computational Cost)

**증상**:
- 학습 시간 너무 길음 (> 24시간)
- GPU 메모리 부족

**해결 방법**:

| 조치 | 변경 사항 | 효과 |
|-----|---------|------|
| 배치 크기 감소 | 24 → 12 | 메모리 50% 절감 |
| max_length 감소 | 192 → 128 | 메모리 30% 절감 |
| gradient_accumulation 증가 | 4 → 8 | 배치 크기 유지, 메모리 절감 |
| num_workers 감소 | 4 → 2 | CPU 메모리 절감 |
| 에폭 감소 | 25 → 20 | 학습 시간 20% 단축 |
| 혼합 정밀도 유지 | bf16 | 이미 최적화됨 |
| 체크포인트 간격 증가 | 5 → 10 | I/O 오버헤드 감소 |

---

## 10. 고급 튜닝 전략

### 전략 1: IDF 가중치 최적화

**목표**: 어휘 빈도 특성에 맞춘 희소 표현

```
1. IDF 가중치 사전계산:
   - 학습 데이터에서 IDF 계산
   - 저장: outputs/train_v26/idf_weights/xlmr_v26_idf

2. 파라미터 조정:
   - idf_alpha: 3.0 (약) ~ 5.0 (강)
   - idf_smoothing: 'bm25' (한국어) 권장

3. 검증:
   - 일반적 불용어 점수 < 0.1
   - 중요 토큰 점수 > 0.3
```

### 전략 2: 점진적 학습률 감소

V26의 3단계 커리큘럼을 활용한 학습률 스케줄:

```python
# Phase 1-8: 1.0x → Learning rate = 3e-5
# Phase 9-17: 0.5x → Learning rate = 1.5e-5
# Phase 18-25: 0.25x → Learning rate = 7.5e-6
```

이는 초기에 큰 업데이트로 빠른 수렴, 후기에 작은 업데이트로 정제를 실현합니다.

### 전략 3: 데이터 가중치 커스터마이징

기본 커리큘럼 대신 커스텀 데이터 가중치:

```python
# 집중: 다중 용어 쿼리가 많은 도메인
curriculum_phases[0].data_weights = {
    "multi_term": 0.6,  # 높음
    "single_term": 0.2,
    "original": 0.2,
}

# 보편: 다양한 쿼리 길이
curriculum_phases[0].data_weights = {
    "single_term": 0.4,   # 기본값
    "multi_term": 0.35,
    "original": 0.25,
}
```

### 전략 4: 하이브리드 정규화

FLOPS + 불용어 + 특수 토큰 페널티 조합:

```
Total Penalty =
  lambda_flops × (IDF_penalty + stopword_penalty + special_token_penalty)
```

**강력한 희소성** (권장):
- lambda_flops: 0.015 (높음)
- stopword_penalty: 20.0 (높음)
- special_token_penalty: 120.0 (높음)

**균형 (기본, V26)**:
- lambda_flops: 0.010
- stopword_penalty: 15.0
- special_token_penalty: 100.0

**약한 희소성**:
- lambda_flops: 0.005 (낮음)
- stopword_penalty: 10.0 (낮음)
- special_token_penalty: 80.0 (낮음)

---

## 11. 파라미터 영향도 매트릭스

각 파라미터가 주요 지표에 미치는 영향 (상대 중요도):

| 파라미터 | nDCG@10 | 희소성 | 속도 | 안정성 |
|---------|---------|-------|------|-------|
| learning_rate | ★★★★★ | ★★ | ★ | ★★★★★ |
| num_epochs | ★★★★ | ★★★ | ★★★★ | ★★ |
| lambda_infonce | ★★★★★ | ★★ | ★ | ★★★ |
| lambda_flops | ★★★ | ★★★★★ | ★ | ★★★ |
| stopword_penalty | ★★★ | ★★★★ | ★ | ★★ |
| batch_size | ★★★★ | ★ | ★★★ | ★★★★ |
| temperature | ★★★ | ★ | ★ | ★★★ |
| dropout | ★★ | ★ | ★ | ★★★★ |
| weight_decay | ★★ | ★ | ★ | ★★★★ |
| idf_alpha | ★★★ | ★★★ | ★★ | ★★ |

**범례**:
- ★★★★★: 매우 높은 영향도
- ★★★★: 높은 영향도
- ★★★: 중간 영향도
- ★★: 낮은 영향도
- ★: 매우 낮은 영향도

---

## 12. 체크리스트: 처음 사용자

V26으로 처음 학습하기 전에 확인사항:

- [ ] CUDA 환경 설정 완료
- [ ] 학습 데이터 준비 완료 (data/v24.0/)
- [ ] output_dir 쓰기 권한 확인
- [ ] GPU 메모리 >= 24GB (배치 크기 24 기준)
- [ ] IDF 가중치 경로 설정 또는 자동 계산 활성화
- [ ] 불용어 목록 확인 (한국어 특화)
- [ ] 혼합 정밀도 지원 확인 (bf16 권장)
- [ ] 커리큘럼 학습 활성화 확인 (enable_curriculum=True)
- [ ] 체크포인트 저장 디렉터리 준비
- [ ] 모니터링 도구(TensorBoard) 준비

---

## 13. 참고: 설정 로드 예시

### Python 코드

```python
from src.train.config.v26 import create_default_v26_config

# 기본 V26 설정 생성
config = create_default_v26_config(
    train_files=["data/v24.0/train_*.jsonl"],
    val_files=["data/v24.0/val.jsonl"],
    output_dir="outputs/train_v26",
    batch_size=24,
    num_epochs=25,
)

# 커스텀 변경
config.loss.lambda_flops = 0.015  # 희소성 강화
config.loss.stopword_penalty = 20.0  # 불용어 억제 강화
config.training.learning_rate = 5e-5  # 학습률 증가

# 설정 검증
config.validate()
```

### YAML 설정 파일

```yaml
model:
  name: xlm-roberta-base
  dropout: 0.1
  use_expansion: true
  expansion_mode: mlm

data:
  train_files:
    - data/v24.0/train_*.jsonl
  val_files:
    - data/v24.0/val.jsonl
  batch_size: 24
  max_length: 192
  num_workers: 4

training:
  num_epochs: 25
  learning_rate: 3e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  gradient_accumulation_steps: 4
  mixed_precision: bf16
  output_dir: outputs/train_v26

loss:
  lambda_infonce: 3.0
  lambda_self: 0.5
  lambda_positive: 2.0
  lambda_flops: 0.010
  lambda_kd: 2.0

  # IDF 설정 (V26 특화)
  idf_alpha: 4.0
  idf_smoothing: bm25

  # 불용어 설정 (V26 강화)
  stopword_penalty: 15.0
  use_extended_stopwords: true

  # 특수 토큰 설정 (V26 신규)
  special_token_penalty: 100.0
```

---

## 요약: V26 vs V25 핵심 개선점

| 측면 | V25 | V26 | 개선 이유 |
|-----|-----|-----|---------|
| **희소성 강도** | 약함 | 강함 | 더 적은 활성화로 효율성 증가 |
| **불용어 처리** | 기본 | 강화됨 | 한국어 특화 불용어 더 효과적 억제 |
| **IDF 곡선** | 완만함 | 상승화됨 | 고빈도 용어와 저빈도 용어의 차별화 강화 |
| **특수 토큰** | 미지원 | 명시적 억제 | 문법적 의미 없는 토큰 제거 |
| **확장 불용어** | 미지원 | 지원됨 | 한국어 도메인 특화 향상 |

---

## 참고 자료

- **설정 코드**: `src/train/config/v26.py`
- **모델 코드**: `src/train/` (모델 구현)
- **학습 스크립트**: `src/train/__main__.py`
- **이전 버전**: V25 (`src/train/config/v25.py`)

---

**문서 버전**: V26 설정 기반
**마지막 업데이트**: 2025-01
**대상 사용자**: V26 신규 사용자, 하이퍼파라미터 튜닝 사용자
