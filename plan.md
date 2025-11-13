# Plan: LLM 기반 합성 데이터 생성 및 한영 통합 동의어 사전 추가

## 📋 프로젝트 개요

**목표**: `korean_neural_sparse_training.ipynb`에 LLM 기반 합성 데이터 생성 기능과 임베딩 기반 한영 동의어 사전 생성 기능 추가

**핵심 요구사항**:
1. LLM을 통한 합성 데이터 생성 (Query-Document pairs)
2. 한영 통합 동의어 사전 구축 (임베딩 기반)
3. Local에 gpt-odd-20b 모델 로딩 및 활용
4. 기존 워크플로우와 통합

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
- ❌ gpt-odd-20b 모델 로딩 및 추론
- ❌ LLM을 활용한 고품질 Query-Document pair 생성
- ❌ LLM 기반 동의어 검증 및 확장

---

## 📦 Phase 1: 환경 설정 및 gpt-odd-20b 모델 로딩

### 1.1 의존성 추가
**파일**: `requirements.txt`

추가할 패키지:
```txt
# LLM inference (Local model support)
vllm==0.6.4.post1         # Fast LLM inference with GPU
torch==2.5.1              # Already exists
transformers==4.46.3      # Already exists
```

**대안**: vLLM 대신 transformers만 사용 가능 (메모리 효율은 낮지만 설치 간단)

### 1.2 모델 로더 모듈 구현
**새 파일**: `src/llm_loader.py`

기능:
- gpt-odd-20b 모델 로딩 (Hugging Face 또는 로컬 경로)
- GPU 메모리 최적화 (int8/fp16 quantization)
- Batch inference 지원
- Prompt template 관리

체크리스트:
- [ ] `load_llm_model()` 함수 구현
- [ ] `generate_text()` 함수 구현
- [ ] Prompt template 정의
- [ ] GPU 메모리 체크 및 최적화

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

## ⚙️ 기술적 고려사항

### GPU 메모리 요구사항
- **gpt-odd-20b 모델 크기**: ~40GB (FP16), ~20GB (INT8)
- **BERT 학습 메모리**: ~8-12GB
- **총 필요 메모리**: ~30GB 이상 권장
- **대안**:
  - Smaller LLM 사용 (e.g., GPT-2-XL, Llama-7B)
  - CPU offloading
  - Quantization (INT4/INT8)

### LLM 선택지
1. **gpt-odd-20b** (요구사항) - 성능 우수, 메모리 많이 필요
2. **대안 1**: GPT-J-6B (경량, 한국어 성능 낮음)
3. **대안 2**: Polyglot-Ko-12.8B (한국어 특화, 중간 크기)
4. **대안 3**: OpenAI API (클라우드, 비용 발생)

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

- [ ] gpt-odd-20b 모델 로딩 성공
- [ ] 최소 1,000개 이상의 합성 Query-Document pairs 생성
- [ ] 한영 동의어 사전 크기 2배 이상 증가
- [ ] 합성 데이터로 학습 시 검색 정확도 향상 (MRR/NDCG)
- [ ] Notebook 전체 실행 시간 3시간 이내 (GPU 환경)

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

- [vLLM Documentation](https://docs.vllm.ai/)
- [Hugging Face Transformers - Text Generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [InPars: Data Augmentation for Information Retrieval](https://arxiv.org/abs/2202.05144)
- [Promptagator: Few-shot Dense Retrieval](https://arxiv.org/abs/2209.11755)

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

**Updated**: 2025-11-13
**Status**: Ready for implementation
