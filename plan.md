# OpenSearch Neural Sparse Training - 노트북 분리 계획

## 📋 개요

현재 `korean_neural_sparse_training_v2_llm.ipynb`는 모든 작업을 원스톱으로 실행하도록 되어 있습니다.
이를 **모듈화된 3개의 노트북**으로 분리하여 독립적으로 실행 가능하도록 재구성합니다.

### 목표
- ✅ 각 노트북을 독립적으로 실행 가능
- ✅ LLM 모델 로딩 시간 절약 (한 번만 로드)
- ✅ 중간 결과물 재사용 가능
- ✅ 메모리 효율적 운영 (필요한 노트북만 실행)
- ✅ 디버깅 및 실험 용이

---

## 🗂️ 노트북 분리 구조

### 📓 노트북 1: `01_neural_sparse_base_training.ipynb`
**목적**: 기본 Neural Sparse 모델 학습 (LLM 없이)

**포함 섹션** (기존 섹션 1-12):
1. 환경 설정 및 라이브러리 설치
2. 한국어 데이터셋 수집
3. IDF 계산
4. 한국어 트렌드 키워드 가중치 추가
5. 자동 트렌드 감지 (Unsupervised)
6. 한영 통합 동의어 사전 (Cross-lingual)
7. OpenSearch 문서 인코더 모델 정의
8. 학습 데이터셋 준비
9. 손실 함수 정의
10. 학습 설정 및 실행
11. 모델 저장 (OpenSearch 호환 형식)
12. 모델 테스트

**저장 데이터** (`/dataset` 폴더):
```
dataset/
├── base_model/
│   ├── korean_documents.json          # 한국어 문서 데이터셋
│   ├── idf_statistics.pkl             # IDF 통계
│   ├── trend_keywords.json            # 트렌드 키워드
│   ├── bilingual_synonyms.json        # 기본 한영 동의어 사전
│   ├── qd_pairs_base.pkl              # 기본 Query-Document pairs
│   └── neural_sparse_v1_model/        # 학습된 v1 모델
│       ├── pytorch_model.bin
│       ├── config.json
│       └── tokenizer/
└── metadata.json                       # 데이터셋 메타정보
```

**실행 시간**: ~30-60분 (GPU 기준)

---

### 📓 노트북 2: `02_llm_synthetic_data_generation.ipynb`
**목적**: LLM 모델 로딩 및 합성 데이터 생성

**포함 섹션** (기존 섹션 13-15):
13. LLM 모델 로딩 및 초기화
14. LLM 기반 합성 Query-Document Pairs 생성
15. LLM 기반 한영 동의어 검증 및 확장

**로드 데이터** (`/dataset` 폴더에서):
- `korean_documents.json` - 합성 쿼리 생성용
- `bilingual_synonyms.json` - 동의어 검증 및 확장용

**저장 데이터** (`/dataset` 폴더):
```
dataset/
├── llm_generated/
│   ├── synthetic_qd_pairs.pkl         # LLM 생성 Query-Document pairs
│   ├── enhanced_synonyms.json         # LLM 검증/확장된 동의어 사전
│   └── generation_metadata.json       # 생성 통계 및 메타정보
└── llm_cache/
    └── model_cache/                    # LLM 모델 캐시 (Hugging Face)
```

**실행 시간**:
- 첫 실행: ~20-30분 (모델 다운로드 ~30GB)
- 이후 실행: ~10-15분 (캐시 사용)

**메모리 요구사항**: ~35GB GPU VRAM (FP8 모델)

---

### 📓 노트북 3: `03_llm_enhanced_training.ipynb`
**목적**: 합성 데이터 포함 재학습 및 성능 비교

**포함 섹션** (기존 섹션 16-17):
16. 합성 데이터 포함 모델 재학습
17. 성능 비교 분석 (기존 vs LLM 확장)

**로드 데이터** (`/dataset` 폴더에서):
- `base_model/` - 기본 학습 데이터 및 v1 모델
- `llm_generated/synthetic_qd_pairs.pkl` - 합성 데이터
- `llm_generated/enhanced_synonyms.json` - 확장된 동의어

**저장 데이터** (`/dataset` 폴더):
```
dataset/
├── enhanced_model/
│   ├── neural_sparse_v2_model/        # LLM 확장 v2 모델
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── tokenizer/
│   ├── training_history.json          # 학습 히스토리
│   └── performance_comparison.json    # v1 vs v2 성능 비교
└── evaluation/
    ├── v1_metrics.json                # v1 모델 평가 지표
    ├── v2_metrics.json                # v2 모델 평가 지표
    └── comparison_plots/               # 비교 시각화
        ├── mrr_comparison.png
        ├── ndcg_comparison.png
        └── precision_recall.png
```

**실행 시간**: ~40-50분 (GPU 기준)

---

## 📂 데이터 저장/로드 유틸리티

### 새 파일: `src/dataset_manager.py`

```python
"""
데이터셋 저장 및 로드 유틸리티
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

class DatasetManager:
    """노트북 간 데이터 공유를 위한 매니저"""

    def __init__(self, base_path: str = "dataset"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

    def save_json(self, data: Any, filename: str, subdir: str = ""):
        """JSON 형식으로 저장"""
        path = self.base_path / subdir / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved: {path}")

    def load_json(self, filename: str, subdir: str = ""):
        """JSON 파일 로드"""
        path = self.base_path / subdir / filename

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded: {path}")
        return data

    def save_pickle(self, data: Any, filename: str, subdir: str = ""):
        """Pickle 형식으로 저장 (Python 객체)"""
        path = self.base_path / subdir / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Saved: {path}")

    def load_pickle(self, filename: str, subdir: str = ""):
        """Pickle 파일 로드"""
        path = self.base_path / subdir / filename

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded: {path}")
        return data

    def save_model(self, model, tokenizer, model_dir: str, subdir: str = ""):
        """PyTorch 모델 저장"""
        path = self.base_path / subdir / model_dir
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        print(f"✓ Saved model: {path}")

    def load_model(self, model_class, model_dir: str, subdir: str = ""):
        """PyTorch 모델 로드"""
        path = self.base_path / subdir / model_dir

        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        from transformers import AutoTokenizer
        model = model_class.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"✓ Loaded model: {path}")
        return model, tokenizer

    def check_data_exists(self, filename: str, subdir: str = "") -> bool:
        """데이터 파일 존재 확인"""
        path = self.base_path / subdir / filename
        return path.exists()

    def list_files(self, subdir: str = "") -> List[str]:
        """특정 디렉토리의 파일 목록"""
        path = self.base_path / subdir
        if not path.exists():
            return []
        return [f.name for f in path.iterdir() if f.is_file()]
```

---

## 🔄 노트북 간 데이터 흐름

```
[01_base_training] → dataset/base_model/ → [02_llm_generation]
                                          ↓
                     dataset/llm_generated/ → [03_enhanced_training]
                                              ↓
                                       dataset/enhanced_model/
```

### 데이터 의존성 매트릭스

| 노트북 | 필요 데이터 | 생성 데이터 |
|--------|-------------|-------------|
| **01_base** | 없음 (외부 데이터 수집) | base_model/* |
| **02_llm** | korean_documents.json<br>bilingual_synonyms.json | llm_generated/* |
| **03_enhanced** | base_model/*<br>llm_generated/* | enhanced_model/*<br>evaluation/* |

---

## 📝 각 노트북의 시작 코드

### 노트북 1: Base Training

```python
# 01_neural_sparse_base_training.ipynb
# Cell 1: 초기화
from src.dataset_manager import DatasetManager
from datetime import datetime

# 데이터셋 매니저 초기화
dm = DatasetManager(base_path="dataset")

# 저장할 메타정보
metadata = {
    "notebook": "01_neural_sparse_base_training",
    "created_at": datetime.now().isoformat(),
    "python_version": "3.12",
    "gpu": "NVIDIA GB10",
}

print("✓ Dataset Manager initialized")
print(f"  Base path: {dm.base_path.absolute()}")
```

### 노트북 2: LLM Synthetic Data

```python
# 02_llm_synthetic_data_generation.ipynb
# Cell 1: 데이터 로드
import os
from src.dataset_manager import DatasetManager

# Disable Triton compilation (ARM compatibility)
os.environ["TRITON_INTERPRET"] = "1"
os.environ["DISABLE_TRITON"] = "1"

dm = DatasetManager(base_path="dataset")

# 필수 데이터 확인
required_files = [
    ("base_model", "korean_documents.json"),
    ("base_model", "bilingual_synonyms.json"),
]

print("Checking required data files...")
for subdir, filename in required_files:
    if not dm.check_data_exists(filename, subdir):
        print(f"❌ Missing: {subdir}/{filename}")
        print("\n💡 Please run notebook 1 first:")
        print("   01_neural_sparse_base_training.ipynb")
        raise FileNotFoundError(f"Missing: {subdir}/{filename}")
    print(f"✓ Found: {subdir}/{filename}")

print("\n✅ All required data files found")

# 데이터 로드
documents = dm.load_json("korean_documents.json", "base_model")
bilingual_dict = dm.load_json("bilingual_synonyms.json", "base_model")

print(f"\n📊 Loaded data:")
print(f"  Documents: {len(documents):,}")
print(f"  Bilingual dict: {len(bilingual_dict):,} terms")
```

### 노트북 3: Enhanced Training

```python
# 03_llm_enhanced_training.ipynb
# Cell 1: 데이터 로드
from src.dataset_manager import DatasetManager

dm = DatasetManager(base_path="dataset")

# 필수 데이터 확인
required_files = [
    ("base_model", "qd_pairs_base.pkl"),
    ("llm_generated", "synthetic_qd_pairs.pkl"),
    ("llm_generated", "enhanced_synonyms.json"),
]

print("Checking required data files...")
for subdir, filename in required_files:
    if not dm.check_data_exists(filename, subdir):
        print(f"❌ Missing: {subdir}/{filename}")
        print("\n💡 Please run previous notebooks first:")
        print("   1. 01_neural_sparse_base_training.ipynb")
        print("   2. 02_llm_synthetic_data_generation.ipynb")
        raise FileNotFoundError(f"Missing: {subdir}/{filename}")
    print(f"✓ Found: {subdir}/{filename}")

print("\n✅ All required data files found")

# 데이터 로드
base_qd_pairs = dm.load_pickle("qd_pairs_base.pkl", "base_model")
synthetic_qd_pairs = dm.load_pickle("synthetic_qd_pairs.pkl", "llm_generated")
enhanced_synonyms = dm.load_json("enhanced_synonyms.json", "llm_generated")

print(f"\n📊 Loaded data:")
print(f"  Base QD pairs: {len(base_qd_pairs):,}")
print(f"  Synthetic QD pairs: {len(synthetic_qd_pairs):,}")
print(f"  Enhanced synonyms: {len(enhanced_synonyms):,} terms")

# v1 모델 로드
from src.opensearch_sparse_encoder import OpenSearchSparseEncoder
print("\nLoading v1 model...")
v1_model, v1_tokenizer = dm.load_model(
    OpenSearchSparseEncoder,
    "neural_sparse_v1_model",
    "base_model"
)
print("✓ v1 model loaded")
```

---

## 🚀 실행 순서

### 첫 실행 (전체 파이프라인)

```bash
# 1. 기본 모델 학습 (섹션 1-12)
jupyter notebook 01_neural_sparse_base_training.ipynb
# → dataset/base_model/ 생성
# 실행 시간: ~30-60분

# 2. LLM 합성 데이터 생성 (섹션 13-15)
jupyter notebook 02_llm_synthetic_data_generation.ipynb
# → dataset/llm_generated/ 생성
# 실행 시간: ~10-15분 (첫 실행 시 모델 다운로드 +10분)

# 3. 확장 모델 학습 및 평가 (섹션 16-17)
jupyter notebook 03_llm_enhanced_training.ipynb
# → dataset/enhanced_model/ 생성
# 실행 시간: ~40-50분
```

### 재실행 시나리오

**시나리오 1**: LLM 합성 데이터만 재생성
```bash
# 기존 base_model 데이터 사용, LLM만 재실행
jupyter notebook 02_llm_synthetic_data_generation.ipynb
# 실행 시간: ~10-15분 (모델 캐시 사용)
```

**시나리오 2**: 다른 하이퍼파라미터로 재학습
```bash
# 기존 합성 데이터 사용, 학습만 재실행
jupyter notebook 03_llm_enhanced_training.ipynb
# 실행 시간: ~40-50분
```

**시나리오 3**: 처음부터 완전 재구축
```bash
# dataset 폴더 삭제 후 전체 실행
rm -rf dataset/
jupyter notebook 01_neural_sparse_base_training.ipynb
jupyter notebook 02_llm_synthetic_data_generation.ipynb
jupyter notebook 03_llm_enhanced_training.ipynb
# 총 실행 시간: ~90-120분
```

---

## 📊 예상 효과

### ⏱️ 시간 절약
- **기존 방식**: 모든 작업 원스톱 실행 → ~90-120분 (매번)
- **분리 후**:
  - 노트북 1: 30-60분 (1회만)
  - 노트북 2: 10-15분 (재사용 가능, LLM 캐시)
  - 노트북 3: 40-50분 (재사용 가능)
  - **재실험 시**: 노트북 3만 실행 → ~40분 ✅ **50% 시간 절약**

### 💾 메모리 효율
- **기존**: 모든 데이터를 메모리에 유지 → ~40GB RAM
- **분리 후**: 필요한 데이터만 로드 → ~15-20GB RAM per notebook ✅ **50% 메모리 절감**

### 🔧 디버깅 용이성
- ✅ 각 단계별 중간 결과물 확인 가능
- ✅ 오류 발생 시 해당 노트북만 재실행
- ✅ LLM 모델 로딩 시간 절약 (한 번만 로드, 캐시 사용)
- ✅ 데이터 버전 관리 및 롤백 가능

### 🧪 실험 편의성
- ✅ 다른 LLM 모델 테스트 (노트북 2만 재실행)
- ✅ 다른 학습 하이퍼파라미터 테스트 (노트북 3만 재실행)
- ✅ 합성 데이터 양 조절 실험
- ✅ 동의어 사전 필터링 전략 비교

---

## 🔧 구현 체크리스트

### Phase 1: 유틸리티 구현 ✅
- [ ] `src/dataset_manager.py` 생성
- [ ] JSON 저장/로드 구현
- [ ] Pickle 저장/로드 구현
- [ ] PyTorch 모델 저장/로드 구현
- [ ] 파일 존재 확인 기능
- [ ] 단위 테스트 작성

### Phase 2: 노트북 1 생성 📝
- [ ] `01_neural_sparse_base_training.ipynb` 생성
- [ ] 기존 섹션 1-12 복사 및 수정
- [ ] DatasetManager 통합
- [ ] 데이터 저장 로직 추가 (모든 섹션 끝)
- [ ] 실행 및 검증

### Phase 3: 노트북 2 생성 🤖
- [ ] `02_llm_synthetic_data_generation.ipynb` 생성
- [ ] 기존 섹션 13-15 복사 및 수정
- [ ] Triton 비활성화 코드 추가
- [ ] 데이터 로드 로직 추가 (시작 부분)
- [ ] LLM 생성 데이터 저장 (끝 부분)
- [ ] 실행 및 검증

### Phase 4: 노트북 3 생성 🎯
- [ ] `03_llm_enhanced_training.ipynb` 생성
- [ ] 기존 섹션 16-17 복사 및 수정
- [ ] 데이터 로드 로직 추가
- [ ] 성능 비교 결과 저장
- [ ] 시각화 결과 저장
- [ ] 실행 및 검증

### Phase 5: 통합 테스트 🧪
- [ ] 전체 파이프라인 실행 (1→2→3)
- [ ] 데이터 무결성 확인
- [ ] 성능 비교 (기존 vs 분리)
- [ ] 메모리 사용량 측정
- [ ] 실행 시간 측정
- [ ] 문서화 업데이트

---

## 📚 파일 구조 최종 모습

```
opensearch-neural-pre-train/
├── notebooks/
│   ├── 01_neural_sparse_base_training.ipynb       # 🆕 기본 학습
│   ├── 02_llm_synthetic_data_generation.ipynb     # 🆕 LLM 합성 데이터
│   ├── 03_llm_enhanced_training.ipynb             # 🆕 확장 학습
│   └── korean_neural_sparse_training_v2_llm.ipynb # 기존 (보관용)
├── src/
│   ├── dataset_manager.py                         # 🆕 데이터 관리
│   ├── llm_loader.py
│   ├── synthetic_data_generator.py
│   ├── cross_lingual_synonyms.py
│   └── opensearch_sparse_encoder.py
├── dataset/                                        # 🆕 공유 데이터 저장소
│   ├── base_model/                                # 노트북 1 결과
│   │   ├── korean_documents.json
│   │   ├── idf_statistics.pkl
│   │   ├── trend_keywords.json
│   │   ├── bilingual_synonyms.json
│   │   ├── qd_pairs_base.pkl
│   │   └── neural_sparse_v1_model/
│   ├── llm_generated/                             # 노트북 2 결과
│   │   ├── synthetic_qd_pairs.pkl
│   │   ├── enhanced_synonyms.json
│   │   └── generation_metadata.json
│   ├── enhanced_model/                            # 노트북 3 결과
│   │   ├── neural_sparse_v2_model/
│   │   ├── training_history.json
│   │   └── performance_comparison.json
│   ├── evaluation/
│   │   ├── v1_metrics.json
│   │   ├── v2_metrics.json
│   │   └── comparison_plots/
│   └── metadata.json
├── plan.md                                         # 이 문서
├── plan_old.md                                     # 이전 plan 백업
└── requirements.txt
```

### 데이터 크기 예상

```
dataset/
├── base_model/              (~2GB)
│   ├── korean_documents.json       (500MB)
│   ├── idf_statistics.pkl          (50MB)
│   ├── trend_keywords.json         (10MB)
│   ├── bilingual_synonyms.json     (5MB)
│   ├── qd_pairs_base.pkl           (300MB)
│   └── neural_sparse_v1_model/     (1GB)
├── llm_generated/           (~1.5GB)
│   ├── synthetic_qd_pairs.pkl      (1GB)
│   ├── enhanced_synonyms.json      (20MB)
│   └── generation_metadata.json    (1MB)
└── enhanced_model/          (~2GB)
    ├── neural_sparse_v2_model/     (1GB)
    ├── training_history.json       (10MB)
    └── evaluation/                 (100MB - plots)

Total: ~5.5GB (모델 제외 시 ~3.5GB)
```

---

## 💡 추가 개선 사항

### 1. 데이터 버전 관리
```python
# dataset/metadata.json 예시
{
  "version": "1.0.0",
  "created_at": "2025-01-14T10:30:00",
  "python_version": "3.12",
  "gpu": "NVIDIA GB10",
  "datasets": {
    "base_model": {
      "version": "1.0.0",
      "created_by": "01_neural_sparse_base_training.ipynb",
      "created_at": "2025-01-14T10:30:00",
      "num_documents": 10000,
      "num_qd_pairs": 30000
    },
    "llm_generated": {
      "version": "1.0.0",
      "created_by": "02_llm_synthetic_data_generation.ipynb",
      "created_at": "2025-01-14T11:00:00",
      "llm_model": "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
      "num_synthetic_pairs": 10000,
      "num_enhanced_synonyms": 5000
    },
    "enhanced_model": {
      "version": "1.0.0",
      "created_by": "03_llm_enhanced_training.ipynb",
      "created_at": "2025-01-14T12:00:00",
      "total_training_pairs": 40000,
      "v1_mrr": 0.85,
      "v2_mrr": 0.92
    }
  }
}
```

### 2. 자동 의존성 체크 함수
```python
# src/dataset_manager.py에 추가
def check_dependencies(self, required: List[Tuple[str, str]]) -> bool:
    """노트북 실행 전 필요한 데이터 확인"""
    missing = []
    for subdir, filename in required:
        if not self.check_data_exists(filename, subdir):
            missing.append(f"{subdir}/{filename}")

    if missing:
        print("=" * 70)
        print("❌ Missing required data files:")
        print("=" * 70)
        for f in missing:
            print(f"   - {f}")
        print("\n💡 Please run previous notebooks first:")
        print("   1. 01_neural_sparse_base_training.ipynb")
        print("   2. 02_llm_synthetic_data_generation.ipynb")
        print("=" * 70)
        return False

    print("✅ All dependencies satisfied")
    return True
```

### 3. 진행 상황 추적
```python
# dataset/progress.json 예시
{
  "01_base_training": {
    "status": "completed",
    "started_at": "2025-01-14T10:00:00",
    "completed_at": "2025-01-14T10:45:00",
    "duration_minutes": 45,
    "success": true
  },
  "02_llm_generation": {
    "status": "completed",
    "started_at": "2025-01-14T11:00:00",
    "completed_at": "2025-01-14T11:15:00",
    "duration_minutes": 15,
    "success": true
  },
  "03_enhanced_training": {
    "status": "in_progress",
    "started_at": "2025-01-14T12:00:00"
  }
}
```

---

## 🎯 성공 기준

- [ ] 각 노트북이 독립적으로 실행 가능
- [ ] 데이터 저장/로드가 정상 작동
- [ ] 전체 파이프라인 실행 시간이 기존 대비 효율적
- [ ] 메모리 사용량이 개선됨
- [ ] 모델 성능이 기존과 동일 또는 향상
- [ ] 모든 노트북이 문서화됨
- [ ] 오류 처리 및 의존성 체크 구현
- [ ] 데이터 버전 관리 시스템 작동

---

## 📝 다음 단계

### 즉시 구현 (우선순위 순서)

1. **`src/dataset_manager.py` 생성** ⚡ (15분)
   - 기본 클래스 구현
   - 저장/로드 메서드
   - 파일 확인 유틸리티

2. **노트북 1 생성** 📓 (30분)
   - 기존 섹션 1-12 복사
   - DatasetManager 통합
   - 저장 로직 추가

3. **노트북 1 테스트** 🧪 (60분)
   - 전체 실행
   - 데이터 저장 검증
   - 문제 수정

4. **노트북 2-3 생성** 📓 (30분)
   - 기존 섹션 복사
   - 로드/저장 로직 추가

5. **전체 파이프라인 테스트** 🎯 (90분)
   - 1→2→3 순차 실행
   - 성능 측정
   - 문서화 완료

---

## 🚀 시작하기

구현을 시작하시려면 다음 중 하나를 선택하세요:

1. **`src/dataset_manager.py` 구현 시작**
2. **노트북 1 생성 시작**
3. **전체 구현 계획 상세화**

어떤 것부터 진행하시겠습니까? 🤔
