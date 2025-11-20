# DGX Spark 빠른 시작 가이드

Nvidia DGX Spark (ARM + GB10 GPU) 환경을 위한 SPLADE-doc 학습 가이드입니다.

## 🖥️ 시스템 환경

**확인된 사양**:
- **CPU**: ARM64 (aarch64)
- **GPU**: NVIDIA GB10 (Blackwell 아키텍처)
- **VRAM**: 119.70 GB
- **CUDA**: 13.0
- **cuDNN**: 91300
- **Python**: 3.12.3
- **PyTorch**: 2.10.0 (dev/nightly, CUDA 13.0)

**최적화 기능**:
- ✅ BF16 mixed precision (Blackwell GPU 최적화)
- ✅ 대용량 배치 사이즈 (119GB VRAM 활용)
- ✅ ARM64 네이티브 지원
- ✅ 자동 mixed precision training

---

## 🚀 빠른 시작

### 1단계: 환경 활성화

```bash
# venv 활성화
source .venv/bin/activate

# GPU 환경 테스트
python test_dgx_setup.py
```

**예상 출력**:
```
======================================================================
Testing SPLADE-doc on Nvidia DGX Spark (ARM + GB10)
======================================================================

[1/5] GPU Information
  GPU: NVIDIA GB10
  CUDA Version: 13.0
  BF16 Support: True
  Total VRAM: 119.70 GB

[2/5] Loading SPLADE-doc model
  ✓ Model loaded: 178,444,801 parameters

...

✓ All tests passed! DGX setup is ready for training.
```

---

### 2단계: 베이스라인 학습 (권장 - 빠른 테스트)

**2-1. 데이터 준비 (10K samples)**

```bash
# Korean Wikipedia (5K) + NamuWiki (5K) 샘플링
python scripts/prepare_baseline_data.py
```

**예상 출력**:
```
======================================================================
Preparing Baseline Training Data (10K samples)
======================================================================

[1/4] Loading Korean Wikipedia data
  Total Korean Wikipedia: 600,000 samples

[2/4] Loading NamuWiki data
  Total NamuWiki: 1,500,000 samples

[3/4] Sampling data
  Sampled 5,000 from Korean Wikipedia
  Sampled 5,000 from NamuWiki
  Total samples: 10,000

[4/4] Splitting into train/val
  Train: 9,000 samples
  Val: 1,000 samples

✓ Baseline data preparation complete!
```

**2-2. 학습 실행**

```bash
# BF16 mixed precision으로 베이스라인 학습
python train.py --config configs/baseline_dgx.yaml
```

**학습 설정**:
- Batch size: 16
- Gradient accumulation: 2 (effective batch = 32)
- Epochs: 3
- Mixed precision: BF16
- 예상 시간: ~10분 (GB10 GPU)

**예상 출력**:
```
================================================================================
Starting training
================================================================================
Epochs: 3
Batch size: 16
Gradient accumulation: 2
================================================================================

Epoch 1/3
Training (Step 0): 100%|████████████| 563/563 [03:45<00:00]
Validation loss: 2.3456

...

================================================================================
Training complete!
Best validation loss: 2.1234
================================================================================
```

---

### 3단계: 대규모 Pre-training (Production)

**3-1. 데이터 확인**

먼저 notebook 01에서 생성한 데이터가 있는지 확인:

```bash
ls -lh dataset/paired_data_split/
```

**필요한 데이터**:
- `ko_wiki_*_train_*.jsonl` - Korean Wikipedia (~600K articles)
- `namuwiki_*_train_*.jsonl` - NamuWiki (~1.5M articles)
- `modu_*_train_*.jsonl` - 모두의 말뭉치
- `en_wiki_*_train_*.jsonl` - English Wikipedia

**3-2. Pre-training 실행**

```bash
# 전체 데이터로 pre-training
python train.py --config configs/pretrain_korean_dgx.yaml
```

**학습 설정**:
- Batch size: 32 (GB10의 119GB VRAM 활용)
- Gradient accumulation: 2 (effective batch = 64)
- Epochs: 10
- Mixed precision: BF16
- Learning rate: 2e-5
- 예상 시간: 수 시간 ~ 1일 (데이터 규모에 따라)

**체크포인트 저장 위치**:
```
outputs/pretrain_korean_dgx/
├── best_model/
│   └── checkpoint.pt
├── epoch_1/
│   └── checkpoint.pt
├── epoch_2/
│   └── checkpoint.pt
...
└── training_log.jsonl
```

---

## 📊 모니터링

### 학습 로그 확인

```bash
# 실시간 로그 확인
tail -f outputs/baseline_dgx/training_log.jsonl

# 또는 pretrain 로그
tail -f outputs/pretrain_korean_dgx/training_log.jsonl
```

### GPU 사용률 모니터링

```bash
# 다른 터미널에서 실행
watch -n 1 nvidia-smi
```

**예상 메모리 사용**:
- Baseline (batch=16): ~8-12 GB
- Pre-training (batch=32): ~20-30 GB
- 여유 VRAM: ~90-100 GB (119GB 중)

---

## 🔧 트러블슈팅

### 1. CUDA Out of Memory

배치 사이즈를 줄이세요:

```yaml
# configs/baseline_dgx.yaml or configs/pretrain_korean_dgx.yaml
data:
  batch_size: 16  # 32 → 16으로 줄이기
```

### 2. DataLoader 오류

데이터 파일이 없는 경우 notebook 01을 먼저 실행:

```bash
jupyter notebook notebooks/pretraining-neural-sparse-model/01_wikipedia_data_extraction.ipynb
```

### 3. ARM 호환성 경고

GB10 GPU는 Compute Capability 12.1이지만 PyTorch는 12.0까지만 공식 지원합니다.
이는 **정상**이며 대부분의 기능은 작동합니다.

### 4. BF16 오류

BF16이 지원되지 않는 경우 FP16으로 변경:

```yaml
# configs/*.yaml
training:
  mixed_precision: "fp16"  # "bf16" → "fp16"
```

---

## 📈 성능 최적화 팁

### 1. 배치 사이즈 증가

GB10의 119GB VRAM을 활용하여 배치 사이즈를 늘릴 수 있습니다:

```yaml
data:
  batch_size: 64  # 32 → 64
training:
  gradient_accumulation_steps: 1  # 2 → 1
```

### 2. 멀티 GPU (미래)

여러 GPU가 있는 경우 PyTorch DDP 사용 가능:

```bash
# 예시 (미구현)
torchrun --nproc_per_node=2 train.py --config configs/pretrain_korean_dgx.yaml
```

### 3. Gradient Checkpointing

메모리가 부족한 경우 활성화:

```yaml
training:
  gradient_checkpointing: true
```

---

## 📝 다음 단계

### 1. MS MARCO Fine-tuning

Pre-training 완료 후 MS MARCO로 fine-tuning:

```bash
python train.py --config configs/finetune_msmarco.yaml
```

### 2. BEIR 평가

학습된 모델 평가:

```bash
python evaluate.py --model outputs/pretrain_korean_dgx/best_model
```

### 3. OpenSearch 배포

모델을 OpenSearch에 업로드하여 실제 검색 서비스에 사용

---

## 🎯 요약

**베이스라인 학습 (빠른 테스트)**:
```bash
source .venv/bin/activate
python scripts/prepare_baseline_data.py
python train.py --config configs/baseline_dgx.yaml
```

**대규모 Pre-training**:
```bash
source .venv/bin/activate
python train.py --config configs/pretrain_korean_dgx.yaml
```

**DGX Spark 최적화 포인트**:
- ✅ BF16 mixed precision
- ✅ Large batch sizes (32-64)
- ✅ ARM64 native support
- ✅ 119GB VRAM utilization

Happy Training! 🚀
