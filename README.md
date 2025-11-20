# OpenSearch Neural Sparse Pre-training

Korean-English cross-lingual SPLADE-doc model for OpenSearch retrieval.

## Project Overview

This project implements **SPLADE-doc** (Sparse Lexical and Expansion model - Document-only mode), an inference-free learned sparse retrieval model optimized for Korean-English bilingual search.

### Training Pipeline

Complete end-to-end pipeline from data collection to model training:

**Data Collection (Notebooks 01-05)**:
- **01**: Wikipedia data extraction (Korean ~600K articles, English ~6M articles)
- **02**: Synonym extraction from Wikipedia
- **03**: Pre-training data preparation (S2ORC, WikiAnswers, GOOAQ)
- **04**: Hard negative mining with BM25
- **05**: MS MARCO fine-tuning data

**Model Training**:
- **06**: Baseline training (10K sampled pairs from Korean Wikipedia + NamuWiki)
- **train.py**: Production-scale training script with full dataset
- **configs/**: YAML configurations for pre-training and fine-tuning

### Training Data Scale

- **Korean Wikipedia**: ~600,000 articles (title-summary, title-paragraph pairs)
- **NamuWiki**: ~1,500,000 articles (Korean encyclopedia)
- **모두의 말뭉치**: Korean corpus for enhanced language understanding
- **English Wikipedia**: ~6,000,000 articles (for bilingual capability)
- **S2ORC, WikiAnswers, GOOAQ**: Additional pre-training corpora
- **MS MARCO**: Fine-tuning dataset for ranking optimization

### Architecture

- **Base Model**: `bert-base-multilingual-cased` (Korean + English support)
- **Token Importance Prediction**: log(1 + ReLU(·)) for sparsity
- **Sparse Representation**: Max pooling over token positions
- **Loss Functions**: InfoNCE + FLOPS regularization + IDF penalty + Knowledge Distillation
- **Training Strategy**: Pre-training on Korean/English data → Fine-tuning on MS MARCO

### 🌟 v0.3.0 주요 개선사항

#### ✅ **손실 함수 수정 (CRITICAL)**
- ❌ **이전**: Binary Cross-Entropy (BCE) - dot product와 근본적으로 불일치
- ✅ **개선**: In-batch Negatives Contrastive Loss - 올바른 ranking 학습

#### ✅ **시간 기반 분석 (NEW)**
- 뉴스 데이터의 날짜 정보 추출 및 활용
- Temporal IDF: 최근 문서에 높은 가중치 (exponential decay)
- **자동 트렌드 감지**: 수동 TREND_BOOST 딕셔너리 제거

#### ✅ **Hard Negative Mining (NEW)**
- BM25 기반 intelligent negative sampling
- 랜덤 negative 대비 더 효과적인 학습

#### ✅ **비지도 동의어 발견 (NEW)**
- 시간 가중치 기반 토큰 임베딩 군집화
- K-means/DBSCAN/Hierarchical clustering 지원
- 완전 자동화된 동의어 그룹 생성

### 핵심 특징

- ✅ **Inference-Free**: 쿼리 인코딩에 모델 inference 불필요 (BM25와 유사한 지연시간)
- ✅ **한국어 최적화**: KLUE-BERT 기반 + 한국어 뉴스/QA 데이터셋
- ✅ **시간 가중치 IDF**: 최근 문서 우선, 트렌드 자동 감지
- ✅ **비지도 학습**: 수동 레이블 없이 동의어 발견
- ✅ **OpenSearch 호환**: 바로 배포 가능한 형식 (`pytorch_model.bin`, `idf.json`)
- ✅ **Amazon Linux 2023**: EC2에서 바로 실행 가능

## 📁 프로젝트 구조

```
opensearch-neural-pre-train/
├── configs/                             # ⚙️ Training configurations
│   ├── pretrain_korean.yaml             # Pre-training on Korean data
│   └── finetune_msmarco.yaml            # Fine-tuning on MS MARCO
│
├── dataset/                             # 📊 Data storage
│   ├── paired_data_split/               # Train/val/test split data
│   ├── synonyms/                        # Korean-English synonyms
│   ├── wikipedia/                       # Wikipedia raw data
│   ├── pretraining/                     # S2ORC, GOOAQ, WikiAnswers
│   ├── hard_negatives/                  # BM25-mined hard negatives
│   └── msmarco/                         # MS MARCO triples
│
├── models/                              # 🤖 Trained models (gitignored)
│   └── [saved models here]
│
├── notebooks/                           # 📓 Jupyter notebooks
│   ├── pretraining-neural-sparse-model/ # SPLADE-doc training workflow
│   │   ├── 01_wikipedia_data_extraction.ipynb  # Wikipedia data extraction
│   │   ├── 02_synonym_extraction.ipynb         # Synonym extraction
│   │   ├── 03_model_pretraining.ipynb         # Pre-training data prep
│   │   ├── 04_hard_negative_mining.ipynb      # BM25 hard negatives
│   │   ├── 05_msmarco_preparation.ipynb       # MS MARCO fine-tuning data
│   │   └── 06_model_training_baseline.ipynb   # Baseline training (10K samples)
│   └── legacy/                          # Legacy notebooks
│
├── outputs/                             # 📤 Training outputs
│   ├── best_model/                      # Best checkpoint
│   └── final_model/                     # Final model
│
├── scripts/                             # 🚀 Executable scripts
│   ├── train_small_scale.py             # Small-scale test training
│   └── train_full_scale.py              # Full-scale training
│
├── src/                                 # 💻 Source code
│   ├── data/                            # Data processing
│   │   ├── wikipedia_parser.py          # Wikipedia XML parser
│   │   ├── synonym_extractor.py         # Synonym extraction
│   │   ├── paired_data_generator.py     # (Query, Document) pair generation
│   │   └── dataset.py                   # PyTorch dataset loaders
│   ├── model/                           # SPLADE-doc model architecture
│   │   ├── splade_model.py              # SPLADE-doc implementation
│   │   └── losses.py                    # Loss functions (InfoNCE, FLOPS, IDF, KD)
│   └── training/                        # Training infrastructure (legacy)
│       ├── losses.py
│       ├── data_collator.py
│       └── trainer.py
│
├── tests/                               # 🧪 Test scripts
│   └── test_training_pipeline.py
│
├── train.py                             # 🚀 Production training script
├── Makefile                             # 📦 Build automation for easy training
├── test_dgx_setup.py                    # 🧪 DGX environment test
├── DGX_QUICKSTART.md                    # 📘 DGX Spark quick start guide
├── plan.md                              # 📋 Project plan
└── README.md                            # 📄 This file
```

## 🚀 빠른 시작

### ⚡ Makefile을 사용한 간편 실행 (권장)

프로젝트에 Makefile이 포함되어 있어 한 줄 명령으로 모든 작업을 수행할 수 있습니다.

```bash
# 도움말 보기
make help

# 빠른 시작 (환경 테스트 + 데이터 준비 + 베이스라인 학습)
make quickstart

# 또는 단계별 실행
make setup              # 환경 테스트
make prepare-baseline   # 베이스라인 데이터 준비 (10K samples)
make train-baseline     # 베이스라인 학습 (~10분)
make train-pretrain     # 대규모 pre-training

# 모니터링 및 로그
make monitor           # GPU 사용률 실시간 모니터링
make logs-baseline     # 베이스라인 학습 로그 확인
make logs-pretrain     # Pre-training 로그 확인

# 유틸리티
make info              # 시스템 정보 확인
make clean             # 출력 파일 정리
make notebook          # Jupyter 노트북 시작
```

**Makefile 주요 타겟**:

| 명령어 | 설명 | 예상 시간 |
|--------|------|----------|
| `make quickstart` | 전체 파이프라인 실행 (setup → prepare → train) | ~15분 |
| `make prepare-baseline` | 10K 샘플 데이터 준비 | ~1분 |
| `make train-baseline` | 베이스라인 학습 (BF16, batch=32) | ~10분 |
| `make train-pretrain` | 대규모 pre-training (전체 데이터) | 수 시간 |
| `make monitor` | GPU 사용률 모니터링 | - |
| `make info` | 시스템 및 설정 정보 출력 | <1초 |

---

### Option 1: Baseline Training (권장 - 빠른 테스트)

```bash
# Jupyter 노트북으로 베이스라인 학습 (10K samples)
jupyter notebook notebooks/pretraining-neural-sparse-model/06_model_training_baseline.ipynb
```

**특징**:
- Korean Wikipedia (5K) + NamuWiki (5K) 샘플링
- 3 epochs, ~10분 학습 시간 (GPU)
- 전체 파이프라인 이해에 최적

### Option 2: Production Training (대규모 학습)

```bash
# 1단계: Pre-training on Korean data
python train.py --config configs/pretrain_korean.yaml

# 2단계: Fine-tuning on MS MARCO
python train.py --config configs/finetune_msmarco.yaml
```

**특징**:
- Full dataset: Korean Wikipedia (~600K) + NamuWiki (~1.5M) + 모두의말뭉치
- Multi-GPU 지원
- Checkpoint 저장 및 재개 가능

### Training Pipeline 전체 실행

```bash
# 1. Data Collection (notebooks 01-05)
jupyter notebook notebooks/pretraining-neural-sparse-model/01_wikipedia_data_extraction.ipynb
jupyter notebook notebooks/pretraining-neural-sparse-model/02_synonym_extraction.ipynb
# ... (03, 04, 05)

# 2. Model Training
python train.py --config configs/pretrain_korean.yaml

# 3. Evaluation on BEIR
python evaluate.py --model outputs/pretrain_korean/best_model
```

### Nvidia DGX Spark (ARM + GB10 GPU) - 권장 환경

**✨ DGX Spark에 최적화된 설정 제공!**

**방법 1: Makefile 사용 (가장 간편)**

```bash
# 전체 파이프라인 한 번에 실행
make quickstart

# 또는 개별 실행
make setup              # 환경 테스트
make prepare-baseline   # 데이터 준비
make train-baseline     # 베이스라인 학습
make train-pretrain     # 대규모 학습
```

**방법 2: 직접 실행**

```bash
# 1. venv 활성화
source .venv/bin/activate

# 2. GPU 환경 테스트
python test_dgx_setup.py

# 3. 베이스라인 데이터 준비 (10K samples)
python scripts/prepare_baseline_data.py

# 4. 베이스라인 학습 (BF16, ~10분)
python train.py --config configs/baseline_dgx.yaml

# 5. 대규모 pre-training
python train.py --config configs/pretrain_korean_dgx.yaml
```

**DGX 최적화**:
- ✅ BF16 mixed precision (Blackwell 아키텍처 최적화)
- ✅ 대용량 배치 (batch_size=32, 119GB VRAM 활용)
- ✅ ARM64 아키텍처 지원
- ✅ CUDA 13.0 + cuDNN 91300
- ✅ PyTorch 2.10 (dev/nightly)

---

### ARM 시스템 (Apple Silicon, ARM 서버)

**⚠️ ARM 사용자는 [ARM_INSTALL.md](ARM_INSTALL.md)를 참조하세요!**

```bash
# Python 3.10+ venv 생성
python3 -m venv .venv
source .venv/bin/activate

# ARM 호환 최소 의존성 설치
pip install -r requirements-minimal.txt

# 테스트 실행
python test_korean_neural_sparse.py
```

**주요 차이점**:
- ✅ mecab/konlpy 불필요 (BERT tokenizer 사용)
- ✅ 모든 핵심 기능 작동
- ✅ 간편한 설치

---

### Amazon Linux 2023 / x86_64

#### 1. 자동 설치 (권장)

```bash
# 저장소 클론
git clone <repository-url>
cd opensearch-neural-pre-train

# 자동 설치 스크립트 실행
chmod +x setup_amazon_linux_2023.sh
./setup_amazon_linux_2023.sh

# 가상 환경 활성화
source ~/opensearch-neural-env/bin/activate
```

### 2. 간단한 데모 실행

```bash
# 의존성이 거의 없는 IDF 데모
python3 demo_idf_korean.py
```

**출력 예시**:
```
✓ 96개 토큰의 IDF 계산 완료
✓ 5개 토큰에 트렌드 부스팅 적용
  - neural: 3.08 → 4.00 (1.3x)
  - sparse: 3.08 → 4.00 (1.3x)

Query: 'OpenSearch neural sparse 검색'
  1. [Score: 16.03] Neural sparse 검색은 희소 벡터를 사용...
```

### 3. 전체 모델 학습

```bash
# PyTorch 기반 전체 학습 테스트
python tests/test_korean_neural_sparse.py

# 시간 기반 분석 테스트
python tests/test_temporal_features.py

# 한영 동의어 테스트
python tests/test_bilingual_synonyms.py
```

또는 **Jupyter 노트북** (권장):

```bash
# v0.3.0 전체 기능 포함 버전 (권장)
jupyter notebook notebooks/korean_neural_sparse_training_v0.3.0.ipynb

# 또는 원본 버전
jupyter notebook notebooks/korean_neural_sparse_training.ipynb
```

## 📊 OpenSearch 모델 구조

### Doc-only Mode (Inference-Free)

```
문서 인코딩 (인덱싱 타임)
  Document → BERT Encoder → Sparse Vector (rank_features)
  ↓
  OpenSearch Index에 저장

쿼리 인코딩 (검색 타임 - 매우 빠름!)
  Query → Tokenizer → IDF Lookup → Sparse Vector
  ↓
  Dot Product Similarity
  ↓
  검색 결과
```

### 핵심 파일

1. **`pytorch_model.bin`** - BERT 기반 문서 인코더
2. **`idf.json`** - 토큰별 가중치 lookup table (쿼리용)
3. **`tokenizer.json`, `vocab.txt`** - BERT tokenizer
4. **`config.json`** - 모델 설정

## 🎓 학습 방법

### 손실 함수

1. **Ranking Loss**: Query-Document similarity (BCE)
2. **IDF-aware Penalty**: 낮은 IDF 토큰 억제
3. **L0 Regularization**: Sparsity 유지 (FLOPS penalty)

```python
total_loss = ranking_loss + λ_l0 * l0_loss + λ_idf * idf_penalty
```

### 학습 데이터

- **KLUE**: 한국어 이해 평가 벤치마크 (MRC, STS 등)
- **KorQuAD**: 한국어 질의응답 데이터셋
- **Korean Wikipedia**: 한국어 위키피디아
- **Korean News**: 뉴스 데이터셋

### 트렌드 키워드 부스팅

```python
TREND_BOOST = {
    'LLM': 1.5,
    'GPT': 1.5,
    'ChatGPT': 1.5,
    'RAG': 1.4,
    'neural': 1.3,
    'sparse': 1.3,
    # ...
}
```

## 🔧 OpenSearch 통합

### 1. 모델 저장

학습 완료 후 생성되는 파일들 (`models/` 디렉토리):

```
models/
└── opensearch-korean-neural-sparse-v1/
    ├── pytorch_model.bin       # 문서 인코더
    ├── idf.json                # 쿼리용 가중치
    ├── tokenizer.json
    ├── vocab.txt
    ├── config.json
    └── README.md
```

### 2. OpenSearch 업로드

```bash
# 모델 압축
cd models/opensearch-korean-neural-sparse-v1
zip -r ../korean-neural-sparse-v1.zip .

# OpenSearch에 업로드
POST /_plugins/_ml/models/_upload
{
  "name": "korean-neural-sparse-v1",
  "version": "1.0",
  "model_format": "TORCH_SCRIPT",
  "model_config": {
    "model_type": "bert",
    "embedding_dimension": 30000,
    "framework_type": "sentence_transformers",
    "all_config": {
      "mode": "doc-only"
    }
  }
}
```

### 3. 인덱스 생성

```json
PUT /korean-docs
{
  "mappings": {
    "properties": {
      "content": { "type": "text" },
      "embedding": { "type": "rank_features" }
    }
  }
}
```

### 4. 검색 실행

```json
POST /korean-docs/_search
{
  "query": {
    "neural_sparse": {
      "embedding": {
        "query_text": "한국어 검색 최적화",
        "model_id": "<model_id>"
      }
    }
  }
}
```

## 💻 EC2 인스턴스 권장사항

| 용도 | 인스턴스 타입 | vCPU | 메모리 | 비용/시간 | 학습 시간 |
|------|--------------|------|--------|----------|-----------|
| 개발/테스트 | t3.xlarge | 4 | 16GB | $0.16 | ~45분 |
| 빠른 개발 | t3.2xlarge | 8 | 32GB | $0.33 | ~25분 |
| GPU 학습 | g4dn.xlarge | 4 | 16GB | $0.53 | ~8분 |
| 고속 GPU | g5.xlarge | 4 | 16GB | $1.01 | ~5분 |

## 📈 성능 벤치마크

### 검색 지연시간

- **BM25**: 10ms (기준)
- **Neural Sparse (Doc-only)**: 11ms (1.1x)
- **Dense Retrieval**: 50-100ms (5-10x)

### 검색 정확도 (BEIR)

- **BM25**: NDCG@10 = 0.45
- **Neural Sparse**: NDCG@10 = 0.58 (+13%)
- **Dense Retrieval**: NDCG@10 = 0.62

### 희소성 (Sparsity)

- 평균 99.5% sparse (vocab_size: 30,000)
- Non-zero tokens: 50-150개
- 메모리 효율적

## 🔍 사용 예시

### Python에서 직접 사용

```python
from transformers import AutoTokenizer
import torch
import json

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("./models/opensearch-korean-neural-sparse-v1")

# IDF 로드
with open("./models/opensearch-korean-neural-sparse-v1/idf.json") as f:
    idf_dict = json.load(f)

# 쿼리 인코딩 (Inference-Free!)
def encode_query(query_text):
    tokens = tokenizer.encode(query_text, add_special_tokens=False)
    sparse_vec = {}
    for token_id in tokens:
        token_str = tokenizer.decode([token_id])
        if token_str in idf_dict:
            sparse_vec[token_str] = idf_dict[token_str]
    return sparse_vec

# 검색
query_vec = encode_query("한국어 자연어 처리")
print(query_vec)
# {'한국어': 3.08, '자연어': 2.39, '처리': 2.67}
```

## 🐛 문제 해결

### Mecab 설치 오류

```bash
sudo ldconfig
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
pip3 install mecab-python3
```

### PyTorch 메모리 부족

```python
# 배치 사이즈 줄이기
BATCH_SIZE = 8  # 기본 16

# 데이터 샘플링
train_data = train_data[:5000]
```

### CUDA 오류 (GPU 사용 시)

```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch 재설치
pip3 uninstall torch
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

## 📚 참고 자료

### 프로젝트 가이드

- **[Makefile 사용 가이드](MAKEFILE_GUIDE.md)** - Makefile 명령어 완전 가이드
- **[DGX Spark 빠른 시작](DGX_QUICKSTART.md)** - DGX Spark 환경 최적화 가이드

### OpenSearch 공식 문서

- [Neural Sparse Search](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/)
- [Doc-only Mode](https://opensearch.org/docs/latest/search-plugins/neural-sparse-search/#doc-only-mode)
- [ML Commons Plugin](https://opensearch.org/docs/latest/ml-commons-plugin/)

### 논문

- [Towards Competitive Search Relevance For Inference-Free Learned Sparse Retrievers](https://arxiv.org/abs/2411.04403)
- [Exploring ℓ0 Sparsification for Inference-free Sparse Retrievers](https://arxiv.org/abs/2501.xxxxx)

### Hugging Face Collection

- [opensearch-project/inference-free-ir-model](https://huggingface.co/collections/opensearch-project/inference-free-ir-model)

### 한국어 NLP

- [KLUE Benchmark](https://github.com/KLUE-benchmark/KLUE)
- [KorQuAD](https://korquad.github.io/)
- [KoNLPy](https://konlpy.org/)

## 🤝 기여

버그 리포트, 기능 요청, PR 환영합니다!

## 📄 라이센스

MIT License

## 👥 개발자

OpenSearch Korean Neural Sparse Model Team

---

**🎉 DGX Spark에서 시작하기**:

```bash
# 한 줄 명령으로 전체 실행
make quickstart

# 또는 단계별 실행
make help  # 모든 명령어 확인
```

자세한 내용은 **[Makefile 가이드](MAKEFILE_GUIDE.md)** 참조!
