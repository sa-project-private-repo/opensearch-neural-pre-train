# OpenSearch Neural Sparse 한국어 특화 사전 학습 계획

## 📋 프로젝트 개요

**목표:** 한영 혼용 환경에 최적화된 OpenSearch Neural Sparse Retrieval 모델 사전 학습

**기대 효과:**
- 한국어 검색 성능 향상
- 한영 동의어 자동 매칭 (모델 → model, 검색 → search)
- Cross-lingual retrieval 지원
- OpenSearch 플러그인 직접 배포 가능

---

## 🎯 Phase 1: 아키텍처 및 데이터 준비 (완료)

### ✅ 1.1 데이터 수집 완료
- [x] Korean Wikipedia (100K documents)
- [x] KLUE 데이터셋
- [x] KorQuAD 데이터셋
- [x] Korean News with dates
- [x] LLM 합성 Query-Document pairs
- [x] 한영 동의어 사전 (bilingual synonyms)

**데이터 위치:**
- `dataset/base_model/documents.json` (0.7 MB)
- `dataset/base_model/qd_pairs_base.pkl` (54.9 MB)
- `dataset/llm_generated/synthetic_qd_pairs.pkl` (2.6 KB)
- `dataset/llm_generated/enhanced_synonyms.json` (1.5 KB)

### ✅ 1.2 인프라 준비 완료
- [x] Ollama LLM 통합 (한국어 데이터 생성)
- [x] Wikipedia loader with caching
- [x] DatasetManager for data management
- [x] 테스트 환경 구축

---

## 🚀 Phase 2: Neural Sparse 모델 아키텍처 설계 (진행 중)

### 🔧 2.1 Neural Sparse Encoding 이해

**Neural Sparse Retrieval 핵심 개념:**
```
Query/Document → BERT Encoder → Sparse Vector (Vocab Size)
                                    ↓
                              Top-K Non-zero Terms
                                    ↓
                         Inverted Index (OpenSearch)
```

**특징:**
- BERT 기반 contextualized term weighting
- Sparse vector (대부분의 값이 0)
- Inverted index와 호환 (기존 검색 인프라 활용)
- Dense retrieval보다 해석 가능성 높음

### 📐 2.2 모델 아키텍처

```
Input: "한국어 검색 모델"
   ↓
BERT Encoder (klue/bert-base 또는 multilingual)
   ↓
Token Embeddings [CLS] 한국어 검색 모델 [SEP]
   ↓
MLM Head (Masked Language Model style)
   ↓
Sparse Weights: {
    한국어: 0.85,
    검색: 0.92,
    모델: 0.88,
    search: 0.45,  ← 한영 동의어 학습됨
    model: 0.42,
    ...
}
```

**핵심 컴포넌트:**
1. **Base Encoder**: `klue/bert-base` 또는 `xlm-roberta-base`
2. **Projection Head**: Token-level MLM-style output
3. **Loss Function**: FLOPS regularization + Ranking loss

### 🎯 2.3 학습 목표 정의

**Multi-task Learning:**

1. **Query-Document Matching (Primary)**
   - Positive pairs: (query, relevant_doc) → high similarity
   - Hard negatives: (query, irrelevant_doc) → low similarity
   - Loss: Contrastive loss or margin ranking loss

2. **Cross-lingual Term Alignment (Secondary)**
   - 한영 동의어 쌍의 activation 유사도 최대화
   - Ex: "모델" vs "model" → similar sparse patterns
   - Loss: Cosine similarity loss

3. **Sparsity Regularization (Constraint)**
   - FLOPS (FLoating point OPerations) 제약
   - 너무 많은 term이 activate되면 성능 저하
   - Loss: L1 regularization on activations

**종합 Loss:**
```python
total_loss = (
    α * ranking_loss +           # Query-doc matching
    β * cross_lingual_loss +     # 한영 동의어
    γ * sparsity_loss            # Sparsity constraint
)
```

---

## 📊 Phase 3: 학습 데이터 파이프라인 구축

### 🔄 3.1 데이터 증강 전략

**현재 데이터:**
- Base QD pairs: ~100K pairs
- Synthetic QD pairs: ~10 pairs (테스트)
- Bilingual synonyms: 32 entries

**증강 계획:**

#### Step 1: LLM 합성 데이터 대량 생성
```bash
# Notebook 2 재실행 (1000 documents)
# 예상 출력: 3000 synthetic pairs (1000 docs × 3 queries)
```

#### Step 2: Hard Negative Mining
- 각 query에 대해 BM25로 상위 100개 문서 검색
- Positive 제외한 상위 10개를 hard negatives로 사용
- 예상 출력: 100K × 10 = 1M negative pairs

#### Step 3: Cross-lingual Augmentation
- 한영 동의어 사전 활용
- Query의 term을 영어로 치환
- Ex: "검색 모델" → "search 모델", "검색 model", "search model"
- 예상 출력: 기존 데이터 × 2-3배 증강

### 📁 3.2 최종 학습 데이터 구조

```
TrainingDataset/
  ├─ positive_pairs/
  │   ├─ original_qd_pairs.pkl      # 100K pairs
  │   ├─ synthetic_qd_pairs.pkl     # 3K pairs
  │   └─ augmented_qd_pairs.pkl     # 300K pairs (cross-lingual)
  │
  ├─ negative_pairs/
  │   ├─ hard_negatives.pkl         # 1M pairs (BM25 mining)
  │   └─ random_negatives.pkl       # 100K pairs (sampling)
  │
  └─ bilingual_synonyms/
      └─ synonym_pairs.json          # 32+ entries
```

**예상 총 학습 데이터:**
- Positive: ~400K pairs
- Negative: ~1.1M pairs
- Synonym pairs: 100+ pairs

---

## 🏗️ Phase 4: 모델 학습 구현

### 🔨 4.1 구현 체크리스트

#### ☐ Step 4.1.1: Base Model 선택
```python
# Option 1: 한국어 특화 (권장)
base_model = "klue/bert-base"

# Option 2: 다국어 지원
base_model = "xlm-roberta-base"

# Option 3: 경량화
base_model = "klue/roberta-small"
```

**선택 기준:**
- 한국어 성능: klue/bert-base > xlm-roberta-base
- 다국어 지원: xlm-roberta-base
- 속도: klue/roberta-small

#### ☐ Step 4.1.2: Neural Sparse Encoder 구현
```python
# src/models/neural_sparse_encoder.py

class NeuralSparseEncoder(nn.Module):
    def __init__(self, base_model: str, vocab_size: int):
        self.bert = AutoModel.from_pretrained(base_model)
        self.projection = nn.Linear(768, vocab_size)  # BERT hidden → vocab
        self.activation = nn.ReLU()  # Non-negative weights

    def forward(self, input_ids, attention_mask):
        # BERT encoding
        outputs = self.bert(input_ids, attention_mask)
        token_embeddings = outputs.last_hidden_state

        # Sparse projection
        sparse_logits = self.projection(token_embeddings)
        sparse_weights = self.activation(sparse_logits)

        # Max pooling over tokens (query/doc representation)
        sparse_vec, _ = torch.max(sparse_weights, dim=1)

        return sparse_vec  # [batch, vocab_size]
```

#### ☐ Step 4.1.3: Loss Functions 구현
```python
# src/training/losses.py

def ranking_loss(query_vec, pos_doc_vec, neg_doc_vecs):
    """Margin ranking loss for query-document matching."""
    pos_score = torch.sum(query_vec * pos_doc_vec, dim=-1)
    neg_scores = torch.sum(query_vec * neg_doc_vecs, dim=-1)

    margin = 0.1
    loss = torch.relu(margin - pos_score + neg_scores).mean()
    return loss

def cross_lingual_loss(korean_vec, english_vec):
    """Cosine similarity loss for bilingual terms."""
    cos_sim = F.cosine_similarity(korean_vec, english_vec)
    loss = 1 - cos_sim.mean()
    return loss

def flops_loss(sparse_vec, lambda_flops=0.001):
    """FLOPS regularization for sparsity."""
    l1_norm = torch.sum(torch.abs(sparse_vec), dim=-1)
    loss = lambda_flops * l1_norm.mean()
    return loss
```

#### ☐ Step 4.1.4: Training Loop 구현
```python
# src/training/trainer.py

class NeuralSparseTrainer:
    def train_epoch(self):
        for batch in self.train_loader:
            # Forward pass
            query_vec = self.model(batch['query_ids'])
            pos_doc_vec = self.model(batch['pos_doc_ids'])
            neg_doc_vecs = self.model(batch['neg_doc_ids'])

            # Compute losses
            rank_loss = ranking_loss(query_vec, pos_doc_vec, neg_doc_vecs)
            sparse_loss = flops_loss(query_vec) + flops_loss(pos_doc_vec)

            # Optional: cross-lingual loss
            if batch.has('synonym_pairs'):
                kor_vec = self.model(batch['korean_term_ids'])
                eng_vec = self.model(batch['english_term_ids'])
                cl_loss = cross_lingual_loss(kor_vec, eng_vec)
            else:
                cl_loss = 0

            # Total loss
            loss = rank_loss + 0.1 * cl_loss + sparse_loss

            # Backward
            loss.backward()
            self.optimizer.step()
```

### ⚙️ 4.2 학습 하이퍼파라미터

```yaml
# config/training_config.yaml

model:
  base: "klue/bert-base"
  hidden_size: 768
  vocab_size: 30000  # BERT vocab size

training:
  epochs: 10
  batch_size: 32
  learning_rate: 2e-5
  warmup_steps: 1000
  max_grad_norm: 1.0

  # Loss weights
  alpha_ranking: 1.0      # Query-doc matching
  beta_cross_lingual: 0.1 # 한영 동의어
  gamma_sparsity: 0.001   # FLOPS regularization

  # Negative sampling
  num_hard_negatives: 10
  num_random_negatives: 5

data:
  max_seq_length: 256
  query_max_length: 64
  doc_max_length: 256

evaluation:
  eval_steps: 1000
  save_steps: 2000
  metric: "ndcg@10"
```

### 🖥️ 4.3 GPU 메모리 최적화

**예상 메모리 사용량:**
- Base model (klue/bert-base): ~500MB
- Batch (32 samples): ~2GB
- Optimizer states: ~1GB
- **Total: ~3.5GB**

**최적화 전략:**
```python
# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Gradient accumulation:**
```python
# Effective batch size = 32 × 4 = 128
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 📈 Phase 5: 평가 및 검증

### 🎯 5.1 평가 데이터셋

**Evaluation Sets:**
1. **Test set** (held-out 10%)
   - Base QD pairs에서 분리
   - ~10K pairs

2. **Cross-lingual test**
   - 한국어 query + 영어 term 포함 문서
   - ~1K pairs

3. **Real-world queries**
   - 실제 검색 로그 (있다면)
   - OpenSearch 공식 문서 검색 등

### 📊 5.2 평가 메트릭

**Retrieval Metrics:**
```python
# Mean Reciprocal Rank
MRR = avg(1 / rank_of_first_relevant)

# Normalized Discounted Cumulative Gain
NDCG@10 = DCG@10 / IDCG@10

# Recall@K
Recall@10 = (relevant_in_top10 / total_relevant)

# Precision@K
Precision@10 = (relevant_in_top10 / 10)
```

**Sparsity Metrics:**
```python
# Average number of non-zero terms
avg_active_terms = mean(count(sparse_vec > threshold))

# FLOPS (floating point operations)
flops = sum(sparse_vec)
```

**Cross-lingual Metrics:**
```python
# 한영 동의어 activation 유사도
synonym_similarity = cosine_sim(vec["모델"], vec["model"])
```

### 🔍 5.3 분석 및 디버깅

**분석 항목:**
1. **Top-K activated terms 분석**
   - Query: "한국어 검색"
   - Activated: {한국어: 0.9, 검색: 0.85, search: 0.4, ...}

2. **한영 동의어 매칭 검증**
   - "모델" vs "model" activation 비교
   - Cross-lingual retrieval 성공률

3. **Failure case 분석**
   - Retrieval 실패 사례 수집
   - Common patterns 파악

---

## 🚢 Phase 6: OpenSearch 통합 및 배포

### 🔌 6.1 OpenSearch 플러그인 변환

**모델 Export:**
```python
# PyTorch → ONNX
torch.onnx.export(
    model,
    dummy_input,
    "neural_sparse_korean.onnx",
    opset_version=14
)

# ONNX → TorchScript (OpenSearch compatible)
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("neural_sparse_korean.pt")
```

**OpenSearch 설정:**
```json
{
  "model_id": "neural-sparse-korean-v1",
  "model_format": "TORCH_SCRIPT",
  "model_config": {
    "model_type": "bert",
    "embedding_dimension": 768,
    "framework_type": "sentence_transformers"
  }
}
```

### 🧪 6.2 통합 테스트

**End-to-End Test:**
1. 모델 업로드 to OpenSearch
2. Index 생성 with neural sparse pipeline
3. Document 인덱싱
4. 검색 쿼리 실행
5. 결과 검증

```bash
# OpenSearch neural sparse search
POST /my-index/_search
{
  "query": {
    "neural_sparse": {
      "my_text_field": {
        "query_text": "한국어 검색 모델",
        "model_id": "neural-sparse-korean-v1"
      }
    }
  }
}
```

### 📊 6.3 성능 벤치마크

**Baseline 비교:**
- BM25 (keyword)
- Dense retrieval (DPR, ANCE)
- Hybrid (BM25 + Dense)
- **Neural Sparse (Ours)**

**측정 항목:**
- Retrieval quality (NDCG@10)
- Latency (ms per query)
- Throughput (QPS)
- Index size

---

## 📝 Phase 7: 문서화 및 배포

### 📚 7.1 문서 작성

- [ ] `MODEL_CARD.md` - 모델 설명 및 사용법
- [ ] `TRAINING_GUIDE.md` - 학습 가이드
- [ ] `DEPLOYMENT_GUIDE.md` - OpenSearch 배포 가이드
- [ ] `API_REFERENCE.md` - API 문서
- [ ] `EVALUATION_REPORT.md` - 평가 리포트

### 🎁 7.2 배포 준비

- [ ] HuggingFace Hub 업로드
- [ ] Docker image 빌드
- [ ] OpenSearch plugin packaging
- [ ] Example notebooks
- [ ] Demo application

---

## 🗓️ 타임라인

### Week 1-2: 데이터 및 기반 구축 ✅
- [x] 데이터 수집 및 전처리
- [x] LLM 통합 및 합성 데이터 생성
- [x] 한영 동의어 사전 구축

### Week 3: 데이터 증강 및 파이프라인
- [ ] Hard negative mining 구현
- [ ] Cross-lingual augmentation
- [ ] DataLoader 및 학습 파이프라인 구축
- [ ] Notebook 2 재실행 (대량 합성 데이터)

### Week 4: 모델 구현
- [ ] Neural Sparse Encoder 구현
- [ ] Loss functions 구현
- [ ] Trainer 구현
- [ ] 초기 학습 실험

### Week 5: 학습 및 최적화
- [ ] Full training run
- [ ] Hyperparameter tuning
- [ ] 모델 최적화 (pruning, quantization)

### Week 6: 평가 및 분석
- [ ] 평가 데이터셋 구축
- [ ] 종합 평가 실행
- [ ] Failure case 분석
- [ ] 개선 반복

### Week 7: OpenSearch 통합
- [ ] 모델 export (ONNX, TorchScript)
- [ ] OpenSearch 플러그인 통합
- [ ] End-to-end 테스트
- [ ] 성능 벤치마크

### Week 8: 문서화 및 배포
- [ ] 문서 작성
- [ ] HuggingFace Hub 업로드
- [ ] Demo 애플리케이션
- [ ] 최종 검토 및 배포

---

## 📦 현재 디렉토리 구조

```
opensearch-neural-pre-train/
├── dataset/
│   ├── base_model/              # Phase 1 완료 ✅
│   │   ├── documents.json
│   │   ├── qd_pairs_base.pkl
│   │   └── bilingual_synonyms.json
│   └── llm_generated/           # Phase 1 완료 ✅
│       ├── synthetic_qd_pairs.pkl
│       └── enhanced_synonyms.json
│
├── src/
│   ├── dataset_manager.py       # 완료 ✅
│   ├── llm_loader.py            # 완료 ✅
│   ├── wikipedia_loader.py      # 완료 ✅
│   ├── synthetic_data_generator.py  # 완료 ✅
│   ├── cross_lingual_synonyms.py    # 완료 ✅
│   │
│   ├── models/                  # TODO 🔨
│   │   ├── neural_sparse_encoder.py
│   │   └── model_config.py
│   │
│   ├── training/                # TODO 🔨
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   ├── data_collator.py
│   │   └── hard_negative_miner.py
│   │
│   └── evaluation/              # TODO 🔨
│       ├── metrics.py
│       └── evaluator.py
│
├── notebooks/
│   ├── 01_neural_sparse_base_training.ipynb  # 완료 ✅
│   ├── 02_llm_synthetic_data_generation.ipynb  # 완료 ✅
│   ├── 03_llm_enhanced_training.ipynb  # 진행 예정
│   ├── 04_data_augmentation.ipynb      # TODO 🔨
│   ├── 05_model_training.ipynb         # TODO 🔨
│   ├── 06_evaluation.ipynb             # TODO 🔨
│   └── 07_opensearch_integration.ipynb # TODO 🔨
│
├── config/
│   └── training_config.yaml     # TODO 🔨
│
├── tests/
│   ├── test_korean_generation.py  # 완료 ✅
│   └── test_neural_sparse_encoder.py  # TODO 🔨
│
├── plan.md                      # 이 문서
└── README.md
```

---

## 🎯 다음 즉시 할 일 (Priority)

### 🔥 High Priority (이번 주)

1. **Notebook 2 재실행 - 대량 합성 데이터 생성**
   ```python
   # notebooks/02_llm_synthetic_data_generation.ipynb
   # max_documents를 10 → 1000으로 변경
   synthetic_pairs = generate_synthetic_qd_pairs(
       documents=documents[:1000],  # 1000 documents
       num_queries_per_doc=3,
   )
   # 예상 출력: 3000 synthetic pairs
   ```

2. **Hard Negative Mining 구현**
   ```python
   # src/training/hard_negative_miner.py
   - BM25 기반 negative sampling
   - Top-K 후보 중 positive 제외
   - Batch processing for efficiency
   ```

3. **Neural Sparse Encoder 기본 구현**
   ```python
   # src/models/neural_sparse_encoder.py
   - BERT base + projection layer
   - Forward pass 구현
   - 간단한 테스트
   ```

### 📌 Medium Priority (다음 주)

4. **Loss Functions 구현**
   - Ranking loss
   - Cross-lingual loss
   - FLOPS regularization

5. **Training Loop 구현**
   - Trainer class
   - Logging and checkpointing
   - Early stopping

6. **Evaluation Framework**
   - Test set 분리
   - Metrics 계산
   - Baseline 비교

### 💡 Low Priority (나중에)

7. **OpenSearch 통합**
8. **문서화**
9. **배포 준비**

---

## ❓ 의사결정 필요 항목

### 1. Base Model 선택
- [ ] **klue/bert-base** (한국어 특화, 권장)
- [ ] **xlm-roberta-base** (다국어)
- [ ] **기타**: _____________

### 2. 학습 데이터 규모
- [ ] Small (100K pairs) - 빠른 실험
- [ ] **Medium (500K pairs)** - 권장
- [ ] Large (1M+ pairs) - 최고 성능

### 3. 평가 전략
- [ ] Offline evaluation only
- [ ] **Online A/B testing** (OpenSearch 통합 후)

### 4. 배포 우선순위
- [ ] HuggingFace Hub
- [ ] **OpenSearch Plugin**
- [ ] Docker Container
- [ ] API Server

---

## 📞 참고 자료

### Papers
- [SPLADE](https://arxiv.org/abs/2107.05720) - Sparse Lexical and Expansion Model
- [DeepImpact](https://arxiv.org/abs/2104.12016) - Neural Text Ranking
- [uniCOIL](https://arxiv.org/abs/2106.14807) - Contextualized Term Weighting

### Code References
- [naver/splade](https://github.com/naver/splade)
- [OpenSearch Neural Search](https://opensearch.org/docs/latest/neural-search-plugin/index/)
- [Sentence Transformers](https://www.sbert.net/)

### Datasets
- [KLUE Benchmark](https://klue-benchmark.com/)
- [KorQuAD](https://korquad.github.io/)
- [Korean Wikipedia](https://ko.wikipedia.org/)

---

## ✅ Success Criteria

프로젝트 성공 기준:

1. **성능**
   - NDCG@10 > 0.6 (baseline BM25 대비 +10%)
   - 한영 cross-lingual retrieval 지원

2. **효율성**
   - Query latency < 50ms
   - Average active terms < 100

3. **배포**
   - OpenSearch 플러그인 동작 확인
   - 실제 검색 서비스 적용 가능

4. **재현성**
   - 전체 학습 파이프라인 자동화
   - 문서화 완료

---

**Last Updated:** 2025-11-16
**Status:** Phase 2 시작 (Neural Sparse 모델 아키텍처 설계)
**Next Milestone:** Hard negative mining 구현 및 대량 합성 데이터 생성
