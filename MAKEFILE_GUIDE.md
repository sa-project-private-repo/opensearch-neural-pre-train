# Makefile 사용 가이드

DGX Spark 환경에서 SPLADE-doc 학습을 위한 Makefile 완전 가이드입니다.

## 📋 목차

1. [빠른 시작](#빠른-시작)
2. [주요 명령어](#주요-명령어)
3. [학습 워크플로우](#학습-워크플로우)
4. [모니터링](#모니터링)
5. [유틸리티](#유틸리티)
6. [트러블슈팅](#트러블슈팅)

---

## 빠른 시작

### 한 줄 명령으로 전체 실행

```bash
make quickstart
```

이 명령은 다음을 자동으로 수행합니다:
1. ✅ 환경 테스트 (`make setup`)
2. ✅ 데이터 준비 (`make prepare-baseline`)
3. ✅ 베이스라인 학습 (`make train-baseline`)

**예상 시간**: ~15분 (GB10 GPU 기준)

---

## 주요 명령어

### 📚 도움말

```bash
# 모든 사용 가능한 명령어 확인
make help

# 시스템 정보 확인
make info
```

### 🔧 환경 설정

```bash
# GPU 환경 테스트
make setup

# 간단한 테스트만
make test
```

**출력 예시**:
```
======================================================================
Testing SPLADE-doc on Nvidia DGX Spark (ARM + GB10)
======================================================================

[1/5] GPU Information
  GPU: NVIDIA GB10
  CUDA Version: 13.0
  BF16 Support: True
  Total VRAM: 119.70 GB

...

✓ All tests passed! DGX setup is ready for training.
```

### 📊 데이터 준비

```bash
# 베이스라인 데이터 준비 (10K samples)
make prepare-baseline
```

**생성되는 데이터**:
- `dataset/baseline_samples/train_baseline.jsonl` (9,000 samples)
- `dataset/baseline_samples/val_baseline.jsonl` (1,000 samples)

**소스**:
- Korean Wikipedia: 5,000 samples
- NamuWiki: 5,000 samples

---

## 학습 워크플로우

### 1️⃣ 베이스라인 학습 (빠른 테스트)

```bash
make train-baseline
```

**설정**:
- 데이터: 10K samples
- Epochs: 3
- Batch size: 16 (effective: 32)
- Mixed precision: BF16
- 예상 시간: ~10분

**출력 위치**: `outputs/baseline_dgx/`

### 2️⃣ 대규모 Pre-training

```bash
make train-pretrain
```

**설정**:
- 데이터: 전체 Korean + English Wikipedia
- Epochs: 10
- Batch size: 32 (effective: 64)
- Mixed precision: BF16
- 예상 시간: 수 시간 ~ 1일

**출력 위치**: `outputs/pretrain_korean_dgx/`

### 3️⃣ MS MARCO Fine-tuning

```bash
make train-finetune
```

**설정**:
- 데이터: MS MARCO triples
- Epochs: 3
- Batch size: 8 (effective: 64)
- 사전 학습 모델 필요

**출력 위치**: `outputs/finetune_msmarco/`

---

## 모니터링

### GPU 사용률 모니터링

```bash
# 실시간 GPU 사용률 확인 (Ctrl+C로 종료)
make monitor
```

**출력**:
```
Every 1.0s: nvidia-smi

+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GB10                    On  |   00000000:01:00.0 Off |                    0 |
| N/A   45C    P0             75W /  300W |   12345MiB / 122576MiB |     95%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

### 학습 로그 확인

```bash
# 베이스라인 학습 로그
make logs-baseline

# Pre-training 로그
make logs-pretrain

# Fine-tuning 로그
make logs-finetune
```

**로그 형식 (JSONL)**:
```json
{"step": 100, "epoch": 0, "total": 8793.32, "contrastive": 8774.28, "flops": 19.04}
{"step": 200, "epoch": 0, "total": 6234.56, "contrastive": 6220.12, "flops": 14.44}
...
```

---

## 유틸리티

### 🧹 정리 명령어

```bash
# 학습 출력 파일 삭제
make clean-outputs

# 베이스라인 샘플 데이터 삭제
make clean-data

# Python 캐시 파일 삭제
make clean-cache

# 전체 정리 (outputs + cache)
make clean
```

### 📓 Jupyter Notebook

```bash
# Jupyter notebook 서버 시작
make notebook
```

자동으로 `notebooks/pretraining-neural-sparse-model/` 디렉토리에서 시작됩니다.

### 🔍 코드 품질

```bash
# 코드 품질 검사
make lint

# 코드 자동 포맷팅 (black)
make format
```

### 📝 Git 명령어

```bash
# 변경사항 커밋 (메시지 입력 프롬프트)
make commit

# 원격 저장소에 푸시
make push
```

---

## 학습 워크플로우 예시

### 시나리오 1: 빠른 테스트

```bash
# 1. 한 번에 실행
make quickstart

# 2. 로그 확인
make logs-baseline

# 3. GPU 모니터링 (별도 터미널)
make monitor
```

### 시나리오 2: 단계별 실행

```bash
# 1. 환경 확인
make info
make setup

# 2. 데이터 준비
make prepare-baseline

# 3. 학습 시작
make train-baseline

# 4. 로그 모니터링 (다른 터미널에서)
make logs-baseline

# 5. GPU 모니터링 (또 다른 터미널에서)
make monitor
```

### 시나리오 3: 대규모 학습

```bash
# 1. 환경 테스트
make test

# 2. 전체 데이터로 pre-training
make train-pretrain

# 3. 실시간 로그 확인
make logs-pretrain

# 4. 학습 완료 후 fine-tuning
make train-finetune
```

---

## 트러블슈팅

### 문제 1: venv가 없다는 오류

**오류 메시지**:
```
Error: venv not found. Run: python3 -m venv .venv
```

**해결 방법**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 문제 2: 데이터 파일이 없음

**오류 메시지**:
```
No files found matching pattern: dataset/paired_data_split/ko_wiki_*
```

**해결 방법**:
```bash
# 먼저 notebook 01을 실행하여 데이터 생성
make notebook
# 또는 베이스라인만 사용
make prepare-baseline
```

### 문제 3: CUDA Out of Memory

**오류 메시지**:
```
RuntimeError: CUDA out of memory
```

**해결 방법**:

Option 1: 설정 파일에서 batch size 줄이기
```yaml
# configs/baseline_dgx.yaml
data:
  batch_size: 8  # 16 → 8로 줄이기
```

Option 2: Gradient checkpointing 활성화
```yaml
# configs/baseline_dgx.yaml
training:
  gradient_checkpointing: true
```

### 문제 4: 로그 파일이 없음

**오류 메시지**:
```
No logs found. Start training first with: make train-baseline
```

**해결 방법**:
```bash
# 먼저 학습을 시작해야 합니다
make train-baseline
```

---

## 고급 사용법

### 배치 사이즈 최적화

GB10의 119GB VRAM을 최대한 활용:

```bash
# 1. 현재 메모리 사용 확인
make monitor

# 2. configs/*.yaml 수정
# data.batch_size를 32 → 64로 증가
# training.gradient_accumulation_steps를 2 → 1로 감소

# 3. 학습 시작
make train-pretrain
```

### 멀티 터미널 워크플로우

**터미널 1** (학습):
```bash
make train-pretrain
```

**터미널 2** (로그 모니터링):
```bash
make logs-pretrain
```

**터미널 3** (GPU 모니터링):
```bash
make monitor
```

### 자동화 스크립트

```bash
#!/bin/bash
# auto_train.sh

# 전체 파이프라인 자동 실행
make setup
make prepare-baseline
make train-baseline
make train-pretrain
make train-finetune

echo "✓ All training completed!"
```

---

## 명령어 치트시트

| 명령어 | 설명 | 시간 |
|--------|------|------|
| `make help` | 도움말 출력 | <1초 |
| `make info` | 시스템 정보 | <1초 |
| `make quickstart` | 전체 파이프라인 | ~15분 |
| `make setup` | 환경 테스트 | ~10초 |
| `make test` | 간단한 테스트 | ~10초 |
| `make prepare-baseline` | 데이터 준비 | ~1분 |
| `make train-baseline` | 베이스라인 학습 | ~10분 |
| `make train-pretrain` | Pre-training | 수 시간 |
| `make train-finetune` | Fine-tuning | ~1시간 |
| `make monitor` | GPU 모니터링 | - |
| `make logs-*` | 로그 확인 | - |
| `make clean` | 정리 | ~1초 |
| `make notebook` | Jupyter 시작 | ~2초 |

---

## 추가 자료

- **DGX Spark 상세 가이드**: `DGX_QUICKSTART.md`
- **전체 프로젝트 문서**: `README.md`
- **학습 설정 파일**: `configs/`
- **소스 코드**: `src/`

---

**Happy Training with Makefile! 🚀**
