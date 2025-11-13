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

### 1.1 의존성 추가 (ARM 최적화)
**파일**: `requirements.txt`

추가할 패키지:
```txt
# LLM inference (ARM-compatible)
# vLLM은 ARM 지원 제한적이므로 제외
accelerate==1.1.1         # Already exists - 메모리 최적화
bitsandbytes==0.44.1      # INT8/INT4 quantization (ARM 지원)
optimum==1.23.3           # ONNX Runtime 최적화
sentencepiece==0.2.0      # Already exists - tokenizer
```

**전략**: Hugging Face Transformers + bitsandbytes quantization 사용
- vLLM 대신 기본 transformers 사용 (ARM 호환)
- bitsandbytes로 INT8/INT4 양자화 (메모리 절약)
- accelerate로 멀티 GPU/CPU offloading

### 1.2 모델 로더 모듈 구현 (ARM 최적화)
**새 파일**: `src/llm_loader.py`

기능:
- ARM 호환 경량 LLM 로딩 (Hugging Face)
- GPU 메모리 최적화 (INT8/INT4 quantization via bitsandbytes)
- Batch inference 지원
- Prompt template 관리
- CPU offloading 지원 (메모리 부족 시)

**권장 모델 (ARM 호환 + 한국어 지원)**:
1. **Polyglot-Ko-5.8B** (한국어 특화, 11GB → 3GB with INT8)
2. **Llama-3.2-3B-Instruct** (다국어, 6GB → 1.5GB with INT8)
3. **Gemma-2-2B-it** (경량, 4GB → 1GB with INT8)
4. **EEVE-Korean-10.8B** (한국어 우수, 20GB → 5GB with INT8)

**선택 전략**: GPU 메모리 고려하여 Llama-3.2-3B 또는 Gemma-2-2B 추천

체크리스트:
- [ ] `load_llm_model_quantized()` 함수 구현 (INT8/INT4)
- [ ] `generate_text()` 함수 구현
- [ ] `generate_batch()` 배치 추론 함수
- [ ] Prompt template 정의 (한국어 최적화)
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
- **LLM 추론 메모리** (예상):
  - Llama-3.2-3B (INT8): ~1.5GB
  - Gemma-2-2B (INT8): ~1GB
  - Polyglot-Ko-5.8B (INT8): ~3GB
  - EEVE-Korean-10.8B (INT8): ~5GB

**권장 전략**:
- BERT 학습 중이 아닐 때 LLM 로딩 (순차 실행)
- 또는 INT8 quantization으로 Llama-3.2-3B 사용 (가장 안전)
- 필요 시 CPU offloading 활용

### LLM 선택지 (ARM 호환, 우선순위 순)

#### Option 1: Llama-3.2-3B-Instruct ⭐ 추천
- **크기**: 3B params (~6GB FP16, ~1.5GB INT8)
- **장점**: ARM 완벽 지원, 다국어(한국어 포함), 최신 모델
- **단점**: 한국어 전문성 낮음
- **Hugging Face**: `meta-llama/Llama-3.2-3B-Instruct`

#### Option 2: Gemma-2-2B-it
- **크기**: 2B params (~4GB FP16, ~1GB INT8)
- **장점**: 매우 경량, ARM 지원, 빠른 추론
- **단점**: 한국어 성능 제한적
- **Hugging Face**: `google/gemma-2-2b-it`

#### Option 3: Polyglot-Ko-5.8B
- **크기**: 5.8B params (~11GB FP16, ~3GB INT8)
- **장점**: 한국어 특화, 우수한 성능
- **단점**: 메모리 더 필요
- **Hugging Face**: `EleutherAI/polyglot-ko-5.8b`

#### Option 4: EEVE-Korean-10.8B (고급 옵션)
- **크기**: 10.8B params (~20GB FP16, ~5GB INT8)
- **장점**: 한국어 최고 성능
- **단점**: 메모리 많이 필요, 느림
- **Hugging Face**: `yanolja/EEVE-Korean-Instruct-10.8B-v1.0`

#### Option 5: OpenAI API (클라우드 대안)
- **모델**: GPT-4o-mini 또는 GPT-3.5-turbo
- **장점**: 로컬 메모리 불필요, 한국어 우수
- **단점**: 비용 발생, 인터넷 필요
- **사용량 예상**: 1,000 쿼리 생성 시 ~$0.5-1

**최종 추천**: Llama-3.2-3B-Instruct (INT8) - ARM 호환 + 메모리 효율

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

- [ ] ARM 호환 LLM 모델 로딩 성공 (Llama-3.2-3B INT8)
- [ ] GPU 메모리 사용량 10GB 이내 유지
- [ ] 최소 1,000개 이상의 합성 Query-Document pairs 생성
- [ ] 한영 동의어 사전 크기 2배 이상 증가
- [ ] 합성 데이터로 학습 시 검색 정확도 향상 (MRR/NDCG)
- [ ] Notebook 전체 실행 시간 4시간 이내 (ARM GPU 환경)

---

## 🚨 리스크 및 대응

### 리스크 1: GPU 메모리 부족
**대응**:
- INT8 quantization 사용
- Smaller batch size
- Gradient checkpointing
- CPU offloading (속도 저하 감수)

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
- [Llama-3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [Polyglot-Ko Korean LLM](https://huggingface.co/EleutherAI/polyglot-ko-5.8b)

---

## ✅ Checklist Summary

**Phase 1**: 환경 설정 및 모델 로딩
- [ ] requirements.txt 업데이트
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

### Step 1: 의존성 설치
```bash
pip install bitsandbytes optimum
```

### Step 2: LLM 모델 다운로드 (선택)
```python
# Llama-3.2-3B-Instruct (권장)
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # INT8 quantization
    device_map="auto",  # Auto GPU/CPU placement
)
```

### Step 3: 합성 데이터 생성
```python
from src.llm_loader import load_llm_model_quantized
from src.synthetic_data_generator import generate_synthetic_qd_pairs

llm_model, llm_tokenizer = load_llm_model_quantized(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    quantization_bits=8,
)

synthetic_pairs = generate_synthetic_qd_pairs(
    documents=documents[:100],
    llm_model=llm_model,
    llm_tokenizer=llm_tokenizer,
    batch_size=4,  # ARM 환경 최적화
)
```

---

**Updated**: 2025-11-13
**Status**: ARM 최적화 완료, Ready for implementation
**Environment**: ARM aarch64 + NVIDIA GB10 + CUDA 13.0
