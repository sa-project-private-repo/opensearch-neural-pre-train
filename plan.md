# Plan: LLM 기반 합성 데이터 생성 및 한영 통합 동의어 사전 추가 (ARM 최적화)

## 📋 프로젝트 개요

**목표**: `korean_neural_sparse_training.ipynb`에 LLM 기반 합성 데이터 생성 기능과 임베딩 기반 한영 동의어 사전 생성 기능 추가

**핵심 요구사항**:
1. LLM을 통한 합성 데이터 생성 (Query-Document pairs)
2. 한영 통합 동의어 사전 구축 (임베딩 기반)
3. Local에 경량 LLM 모델 로딩 및 활용 (ARM 호환)
4. 기존 워크플로우와 통합

**시스템 환경**:
- **아키텍처**: ARM aarch64 (Blackwell GB10)
- **GPU**: NVIDIA GB10 (CUDA 13.0 지원)
- **메모리**: 제한적 (현재 4.5GB GPU 사용 중)
- **Python**: 3.12 (venv 환경)
- **제약사항**: vLLM은 ARM 지원 제한적 → 대안 필요

---

## 🔍 현황 분석

### 현재 구현된 기능
- ✅ 한영 동의어 사전 기초 구현 (`src/cross_lingual_synonyms.py`)
  - Pattern-based extraction (e.g., "모델(model)")
  - Embedding similarity 기반 동의어 발견
  - Manual curated pairs
  - 노트북 Cell 14에서 사용 중

### 추가 필요 기능
- ❌ LLM 기반 합성 데이터 생성
- ❌ ARM 호환 LLM 로딩 및 추론
- ❌ LLM을 활용한 고품질 Query-Document pair 생성
- ❌ LLM 기반 동의어 검증 및 확장

---

## 📦 Phase 1: 환경 설정 및 ARM 호환 LLM 로딩

### 1.1 Python 환경 설정
**Python 버전**: 3.12 (venv)

```bash
# venv 생성 및 활성화
python3.12 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는 .venv\Scripts\activate  # Windows

# pip 업그레이드
pip install --upgrade pip setuptools wheel
```

**Python 3.12 호환성**:
- ✅ PyTorch 2.5.1 (Python 3.12 지원)
- ✅ Transformers 4.46.3 (Python 3.12 지원)
- ✅ AutoAWQ 0.2.7 (Python 3.12 지원)
- ⚠️ llama-cpp-python: 빌드 필요할 수 있음 (ARM + Python 3.12)

### 1.2 의존성 추가 (ARM + Python 3.12 최적화)
**파일**: `requirements.txt`

추가할 패키지:
```txt
# Python 3.12 compatible versions
# LLM inference (ARM-compatible)
# vLLM은 ARM 지원 제한적이므로 제외
accelerate==1.1.1         # Already exists - 메모리 최적화
autoawq==0.2.7            # AWQ quantization (Qwen3 권장, Python 3.12 OK)
optimum==1.23.3           # ONNX Runtime 최적화
sentencepiece==0.2.0      # Already exists - tokenizer

# gpt-oss-20b 사용 시 (GGUF)
# llama-cpp-python==0.3.4  # Optional: gpt-oss-20b GGUF 지원
#                          # ARM + Python 3.12: 소스 빌드 필요할 수 있음
```

**전략**: Hugging Face Transformers + AutoAWQ quantization 사용
- Qwen3: AutoAWQ로 4-bit 양자화 (ARM + Python 3.12 검증)
- gpt-oss-20b: GGUF + llama.cpp (ARM 최적화, Python 3.12 빌드 필요)
- accelerate로 멀티 GPU/CPU offloading
- 모든 패키지 Python 3.12 호환 버전 사용

### 1.3 모델 로더 모듈 구현 (ARM + Python 3.12 최적화)
**새 파일**: `src/llm_loader.py`

기능:
- ARM 호환 경량 LLM 로딩 (Hugging Face)
- GPU 메모리 최적화 (INT8/INT4 quantization via bitsandbytes)
- Batch inference 지원
- Prompt template 관리
- CPU offloading 지원 (메모리 부족 시)

**사용 모델 (요구사항)**:
1. **gpt-oss-20b** (OpenAI, 21B params, 3.6B active)
2. **Qwen3** 시리즈 (Alibaba, 다양한 크기)

**선택 전략**: GPU 메모리 고려하여 Qwen3-14B 또는 gpt-oss-20b (GGUF) 추천

체크리스트:
- [ ] `load_qwen3_awq()` 함수 구현 (AWQ 4-bit)
- [ ] `load_gpt_oss_gguf()` 함수 구현 (GGUF, optional)
- [ ] `generate_text()` 함수 구현
- [ ] `generate_batch()` 배치 추론 함수
- [ ] Prompt template 정의 (Qwen3/gpt-oss 최적화)
- [ ] GPU 메모리 모니터링 유틸리티
- [ ] CPU offloading 옵션

---

## 📝 Phase 2: LLM 기반 합성 데이터 생성

### 2.1 합성 데이터 생성 모듈
**새 파일**: `src/synthetic_data_generator.py`

기능:
- Document → Query 생성 (역방향 생성)
- Query → Document 생성 (정방향 생성)
- Query augmentation (동의어, paraphrase)
- Hard negative document 생성
- 품질 필터링 (길이, 중복 제거)

체크리스트:
- [ ] `generate_queries_from_documents()` 함수
- [ ] `generate_documents_from_queries()` 함수
- [ ] `augment_query()` 함수 (paraphrasing)
- [ ] `generate_hard_negatives()` 함수
- [ ] `filter_synthetic_pairs()` 품질 필터

### 2.2 Prompt Engineering
**Prompt 예시**:

```python
DOC_TO_QUERY_PROMPT = """
다음 문서를 읽고 사용자가 이 문서를 찾기 위해 검색할 만한 쿼리 3개를 생성하세요.

문서: {document}

검색 쿼리 (JSON 형식으로 응답):
"""

SYNONYM_DISCOVERY_PROMPT = """
다음 두 단어가 같은 의미를 가지는지 판단하세요.

단어 1: {word1}
단어 2: {word2}

같은 의미이거나 동의어라면 "예", 아니면 "아니오"로 답하고 간단한 이유를 설명하세요.
"""
```

체크리스트:
- [ ] Document → Query prompt 작성
- [ ] Query → Document prompt 작성
- [ ] Synonym verification prompt 작성
- [ ] Hard negative generation prompt 작성

---

## 🌐 Phase 3: LLM 기반 한영 동의어 사전 확장

### 3.1 동의어 검증 및 확장
**파일**: `src/cross_lingual_synonyms.py` 확장

새 함수:
- `verify_synonyms_with_llm()`: LLM으로 동의어 쌍 검증
- `discover_synonyms_with_llm()`: LLM으로 새 동의어 발견
- `enhance_bilingual_dict_with_llm()`: 기존 사전 품질 향상

체크리스트:
- [ ] `verify_synonyms_with_llm()` 구현
- [ ] `discover_synonyms_with_llm()` 구현
- [ ] `enhance_bilingual_dict_with_llm()` 구현
- [ ] Batch processing 최적화

### 3.2 임베딩 + LLM 하이브리드 접근
**전략**:
1. 임베딩 기반으로 후보 동의어 발견 (기존 방식)
2. LLM으로 후보 검증 및 필터링 (새로운 방식)
3. 검증된 동의어만 최종 사전에 추가

체크리스트:
- [ ] 임베딩 기반 후보 추출 파이프라인
- [ ] LLM 검증 파이프라인
- [ ] 하이브리드 통합 함수

---

## 📓 Phase 4: Notebook 통합

### 4.1 새 Cell 추가
**파일**: `notebooks/korean_neural_sparse_training.ipynb`

추가할 Cell 위치: Cell 14 (한영 동의어 섹션) 앞에 삽입

**새 섹션 1**: LLM 모델 로딩
```python
# Cell: LLM 모델 로딩
from src.llm_loader import load_llm_model, check_gpu_memory

print("🤖 LLM 모델 로딩 중...")
llm_model, llm_tokenizer = load_llm_model(
    model_name="gpt-odd-20b",  # 또는 로컬 경로
    device="cuda",
    quantization="int8",  # 메모리 절약
)
```

**새 섹션 2**: 합성 데이터 생성
```python
# Cell: 합성 데이터 생성
from src.synthetic_data_generator import generate_synthetic_qd_pairs

synthetic_pairs = generate_synthetic_qd_pairs(
    documents=documents[:1000],  # 샘플
    llm_model=llm_model,
    llm_tokenizer=llm_tokenizer,
    num_queries_per_doc=3,
)
```

**새 섹션 3**: LLM 기반 동의어 검증
```python
# Cell: LLM으로 동의어 검증 및 확장
from src.cross_lingual_synonyms import enhance_bilingual_dict_with_llm

enhanced_bilingual_dict = enhance_bilingual_dict_with_llm(
    initial_dict=bilingual_dict,
    llm_model=llm_model,
    llm_tokenizer=llm_tokenizer,
    verification_threshold=0.8,
)
```

체크리스트:
- [ ] LLM 로딩 Cell 추가
- [ ] 합성 데이터 생성 Cell 추가
- [ ] 동의어 검증 Cell 추가
- [ ] 기존 학습 데이터에 합성 데이터 병합
- [ ] 결과 시각화 및 통계

### 4.2 통합 워크플로우
```
1. 데이터 로드 (기존)
2. [NEW] LLM 모델 로딩
3. [NEW] 합성 데이터 생성
4. IDF 계산 (기존)
5. 트렌드 감지 (기존)
6. [ENHANCED] LLM + 임베딩 기반 동의어 사전 구축
7. 모델 학습 (합성 데이터 포함)
8. 평가 및 저장
```

---

## 🔧 Phase 5: 최적화 및 테스트

### 5.1 성능 최적화
체크리스트:
- [ ] LLM inference batching
- [ ] GPU 메모리 모니터링 및 최적화
- [ ] 합성 데이터 캐싱
- [ ] Parallel processing (가능한 경우)

### 5.2 품질 검증
체크리스트:
- [ ] 합성 데이터 품질 평가 (수동 샘플링)
- [ ] 동의어 정확도 측정
- [ ] 학습 성능 비교 (합성 데이터 유/무)
- [ ] Ablation study (LLM vs. 임베딩 only)

### 5.3 문서화
체크리스트:
- [ ] `src/llm_loader.py` docstring 작성
- [ ] `src/synthetic_data_generator.py` docstring 작성
- [ ] `src/__init__.py` 업데이트 (새 함수 export)
- [ ] README 업데이트 (새 기능 설명)
- [ ] Notebook에 설명 markdown cell 추가

---

## ⚙️ 기술적 고려사항 (ARM GB10 환경)

### GPU 메모리 요구사항 (현재: GB10)
- **현재 사용량**: 4.5GB (Jupyter 프로세스)
- **사용 가능 메모리**: 예상 ~12-16GB (GB10 총 메모리 미확인)
- **BERT 학습 메모리**: ~4-6GB (현재 사용 중)
- **LLM 추론 메모리** (요구사항 모델):
  - Qwen3-14B (AWQ 4-bit): ~4GB ⭐
  - Qwen3-7B (AWQ 4-bit): ~2GB
  - gpt-oss-20b (GGUF Q4): ~5GB
  - Qwen3-0.6B (INT8): ~0.3GB (테스트용)

**권장 전략**:
- **Option A**: Qwen3-14B-AWQ 사용 (4-bit, ~4GB) - 성능 우선
- **Option B**: Qwen3-7B-AWQ 사용 (4-bit, ~2GB) - 안정성 우선
- BERT 학습 완료 후 LLM 로딩 (순차 실행 권장)
- 필요 시 CPU offloading 활용 (accelerate)

### LLM 선택지 (요구사항: gpt-oss-20b 또는 Qwen3)

#### Option 1: Qwen3-14B-Instruct ⭐ 최우선 추천
- **크기**: 14B params (~28GB FP16, ~7GB INT8, ~4GB Q4)
- **장점**:
  - ARM aarch64 완벽 지원 (검증됨)
  - 한국어 우수 (다국어 모델)
  - 4-bit/8-bit quantization 성능 우수
  - bitsandbytes, AWQ, GPTQ 모두 지원
- **단점**: 메모리 사용량 높음
- **Hugging Face**: `Qwen/Qwen3-14B-Instruct`
- **Quantized**: `Qwen/Qwen3-14B-AWQ` (4-bit)

#### Option 2: Qwen3-7B-Instruct
- **크기**: 7B params (~14GB FP16, ~3.5GB INT8)
- **장점**:
  - 메모리 효율적
  - ARM 호환
  - 한국어 성능 우수
  - 빠른 추론
- **단점**: 14B 대비 성능 낮음
- **Hugging Face**: `Qwen/Qwen3-7B-Instruct`
- **Quantized**: `Qwen/Qwen3-7B-AWQ`

#### Option 3: gpt-oss-20b (GGUF)
- **크기**: 21B params (3.6B active MoE), ~16GB MXFP4
- **장점**:
  - ARM 자동 최적화 (GGUF)
  - MoE 구조로 메모리 효율적
  - llama.cpp 지원
  - Q4_0, IQ4_NL quantization (ARM 최적화)
- **단점**:
  - Transformers 직접 지원 제한적 (GGUF 사용 필요)
  - llama.cpp 의존성
- **Hugging Face**: `openai/gpt-oss-20b`
- **GGUF**: `ggml-org/gpt-oss-20b-GGUF`

#### Option 4: Qwen3-0.6B (경량 테스트용)
- **크기**: 0.6B params (~1.2GB FP16, ~0.3GB INT8)
- **장점**: 매우 경량, 빠른 실험
- **단점**: 성능 제한적
- **Hugging Face**: `Qwen/Qwen3-0.6B-Instruct`

**최종 추천**:
- **메모리 여유 있음**: Qwen3-14B-AWQ (4-bit, ~4GB) ⭐
- **메모리 제한적**: Qwen3-7B-AWQ (4-bit, ~2GB)
- **gpt-oss-20b 필수**: GGUF Q4_0 버전 (~5GB)

### 품질 vs. 비용 트레이드오프
- **고품질 전략**: LLM으로 모든 동의어 검증 (느림, 비용 높음)
- **균형 전략**: 임베딩으로 후보 추출 + LLM으로 일부 검증 (권장)
- **저비용 전략**: 임베딩만 사용 + 수동 큐레이션

---

## 📅 구현 순서 및 우선순위

### High Priority (Core)
1. ✅ Phase 1.2: 모델 로더 구현 (`src/llm_loader.py`)
2. ✅ Phase 2.1: 합성 데이터 생성기 구현 (`src/synthetic_data_generator.py`)
3. ✅ Phase 4.1: Notebook 통합 (새 Cell 추가)

### Medium Priority (Enhancement)
4. ✅ Phase 3.1: LLM 기반 동의어 검증
5. ✅ Phase 5.2: 품질 검증

### Low Priority (Optimization)
6. ⏸️ Phase 5.1: 성능 최적화
7. ⏸️ Phase 5.3: 문서화 완성

---

## 🎯 성공 지표

- [ ] Qwen3-14B-AWQ 또는 gpt-oss-20b 모델 로딩 성공
- [ ] GPU 메모리 사용량 12GB 이내 유지
- [ ] 최소 1,000개 이상의 합성 Query-Document pairs 생성
- [ ] 한영 동의어 사전 크기 2배 이상 증가
- [ ] 합성 데이터로 학습 시 검색 정확도 향상 (MRR/NDCG)
- [ ] Notebook 전체 실행 시간 4시간 이내 (ARM GPU 환경)

---

## 🚨 리스크 및 대응

### 리스크 1: GPU 메모리 부족
**대응**:
- AWQ 4-bit quantization 사용
- Smaller batch size
- Gradient checkpointing
- CPU offloading (속도 저하 감수)

### 리스크 4: Python 3.12 호환성 문제
**대응**:
- llama-cpp-python: CMAKE로 소스 빌드
- autoawq: 최신 버전 사용 (0.2.7+)
- 의존성 충돌 시 requirements.txt 버전 조정
- venv 환경 격리로 시스템 Python과 분리

### 리스크 2: LLM 생성 품질 낮음
**대응**:
- Prompt engineering 개선
- Few-shot examples 추가
- Temperature/Top-p 조정
- 다른 LLM 모델 시도

### 리스크 3: 합성 데이터 과적합
**대응**:
- 합성/실제 데이터 비율 조정 (1:1 또는 1:2)
- Validation set은 실제 데이터만 사용
- Diversity penalty 추가

---

## 📚 참고 자료

- [Hugging Face Transformers - Text Generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [bitsandbytes - INT8/INT4 Quantization](https://github.com/TimDettmers/bitsandbytes)
- [Accelerate - Memory Optimization](https://huggingface.co/docs/accelerate/index)
- [InPars: Data Augmentation for Information Retrieval](https://arxiv.org/abs/2202.05144)
- [Promptagator: Few-shot Dense Retrieval](https://arxiv.org/abs/2209.11755)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-14B-Instruct)
- [Qwen3 AWQ Quantization](https://huggingface.co/Qwen/Qwen3-14B-Instruct-AWQ)
- [gpt-oss-20b Model Card](https://huggingface.co/openai/gpt-oss-20b)
- [gpt-oss-20b GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF)
- [AutoAWQ Documentation](https://github.com/casper-hansen/AutoAWQ)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)

---

## ✅ Checklist Summary

**Phase 1**: 환경 설정 및 모델 로딩
- [ ] Python 3.12 venv 환경 설정
- [ ] requirements.txt 업데이트 (Python 3.12 호환)
- [ ] src/llm_loader.py 구현
- [ ] GPU 메모리 체크 및 최적화

**Phase 2**: 합성 데이터 생성
- [ ] src/synthetic_data_generator.py 구현
- [ ] Prompt templates 작성
- [ ] 품질 필터링 로직

**Phase 3**: 동의어 사전 확장
- [ ] src/cross_lingual_synonyms.py 확장
- [ ] LLM 검증 함수 추가
- [ ] 하이브리드 파이프라인 구축

**Phase 4**: Notebook 통합
- [ ] 새 Cell 추가 (LLM 로딩, 합성 데이터, 동의어)
- [ ] 기존 워크플로우와 통합
- [ ] 결과 시각화

**Phase 5**: 최적화 및 검증
- [ ] 성능 최적화
- [ ] 품질 평가
- [ ] 문서화

---

---

## 🚀 Quick Start (ARM 환경)

### Step 1: Python 3.12 venv 환경 설정 및 의존성 설치

```bash
# venv 생성 (Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate

# pip 업그레이드
pip install --upgrade pip setuptools wheel

# PyTorch 설치 (CUDA 12.1 for GB10)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Qwen3 사용 시 (권장)
pip install autoawq optimum accelerate transformers

# gpt-oss-20b 사용 시 (추가) - ARM + Python 3.12 빌드
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

**Python 3.12 주의사항**:
- llama-cpp-python은 소스 빌드가 필요할 수 있음 (ARM + CUDA)
- CMAKE_ARGS로 CUDA 지원 활성화
- Qwen3-AWQ는 Python 3.12에서 별도 빌드 불필요

### Step 2: LLM 모델 다운로드

#### Option A: Qwen3-14B (AWQ 4-bit) - 권장 ⭐
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-14B-Instruct-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Auto GPU/CPU placement
    low_cpu_mem_usage=True,
)
```

#### Option B: Qwen3-7B (AWQ 4-bit) - 메모리 제약 시
```python
model_name = "Qwen/Qwen3-7B-Instruct-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)
```

#### Option C: gpt-oss-20b (GGUF) - llama.cpp 필요
```bash
# llama.cpp 설치
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make
```

```python
# Python binding 사용
from llama_cpp import Llama

llm = Llama(
    model_path="gpt-oss-20b-Q4_0.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,  # All layers to GPU
)
```

### Step 3: 합성 데이터 생성
```python
from src.llm_loader import load_llm_model_quantized
from src.synthetic_data_generator import generate_synthetic_qd_pairs

# Qwen3 모델 로딩
llm_model, llm_tokenizer = load_llm_model_quantized(
    model_name="Qwen/Qwen3-14B-Instruct-AWQ",  # 또는 Qwen3-7B-Instruct-AWQ
    use_awq=True,
)

# 합성 데이터 생성
synthetic_pairs = generate_synthetic_qd_pairs(
    documents=documents[:100],
    llm_model=llm_model,
    llm_tokenizer=llm_tokenizer,
    batch_size=2,  # Qwen3-14B에 최적화
)
```

---

**Updated**: 2025-11-13
**Status**: ARM + Python 3.12 최적화 완료, Ready for implementation
**Environment**:
- ARM aarch64 (Blackwell GB10)
- NVIDIA GB10 GPU (CUDA 13.0)
- Python 3.12 (venv)
- PyTorch 2.5.1 (CUDA 12.1)
