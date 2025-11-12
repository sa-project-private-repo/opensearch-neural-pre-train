# ARM 시스템 설치 가이드 (Apple Silicon, ARM 서버)

이 가이드는 ARM 기반 시스템 (Apple M1/M2/M3, ARM 서버 등)에서 프로젝트를 설정하는 방법을 설명합니다.

## 🚨 ARM 시스템 특이사항

ARM 아키텍처에서는 일부 패키지 설치가 어려울 수 있습니다:
- ❌ `mecab-python3`: C++ 컴파일 필요, ARM에서 실패 가능
- ❌ `konlpy`: Java 의존성, 설치 복잡
- ⚠️ `hdbscan`: 선택적 클러스터링 라이브러리, 없어도 작동

## ✅ 빠른 설치 (권장)

### 1. Python 가상환경 생성

```bash
# Python 3.10+ 필요
python3 --version  # 확인

# venv 생성
python3 -m venv .venv

# 활성화
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
```

### 2. Minimal 의존성 설치

```bash
# ARM 호환 최소 의존성만 설치
pip install -r requirements-minimal.txt

# 또는 필수 패키지만 직접 설치
pip install torch transformers datasets numpy pandas scikit-learn tqdm rank-bm25 scipy pyyaml
```

### 3. 검증

```bash
# Python 환경에서 import 테스트
python -c "import torch; import transformers; print('✓ Core packages OK')"
python -c "from src.losses import in_batch_negatives_loss; print('✓ src modules OK')"
```

## 🧪 테스트 실행

### Phase 1 테스트 (손실 함수)

```bash
python test_korean_neural_sparse.py
```

**예상 출력**:
```
✓ 손실 함수 정의 완료
  - NEW: In-Batch Negatives Contrastive Loss
✓ 학습 완료!
```

### Phase 2 테스트 (시간 분석)

```bash
python test_temporal_features.py
```

**예상 출력**:
```
✓ Loaded 9,996 documents
✓ Temporal IDF calculated
✓ 34 trending tokens detected
```

## ⚠️ 문제 해결

### 문제 1: PyTorch 설치 실패

**증상**: `torch` 설치 중 오류

**해결**:
```bash
# Apple Silicon (M1/M2/M3)
pip install torch torchvision torchaudio

# ARM Linux 서버
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 문제 2: scipy 컴파일 오류

**증상**: `scipy` 설치 중 C 컴파일 오류

**해결**:
```bash
# macOS: Xcode Command Line Tools 설치
xcode-select --install

# Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# 그래도 안되면 conda 사용
conda install scipy
```

### 문제 3: rank-bm25 import 오류

**증상**: `from rank_bm25 import BM25Okapi` 실패

**해결**:
```bash
pip install --upgrade rank-bm25
```

## 🚀 Korean NLP 없이 작동 확인

프로젝트의 모든 핵심 기능은 **konlpy/mecab 없이** 작동합니다:

✅ **작동하는 기능**:
- In-batch negatives contrastive loss
- Temporal IDF 계산
- 자동 트렌드 감지
- BM25 hard negative mining
- 토큰 임베딩 클러스터링
- 동의어 자동 발견

❌ **작동하지 않는 기능** (선택적):
- Konlpy 형태소 분석기 (필수 아님)
- Mecab 토크나이저 (BERT tokenizer로 대체 가능)

## 📝 코드 수정 불필요

프로젝트는 이미 ARM 호환성을 고려하여 설계되었습니다:

```python
# src/negative_sampling.py
# BM25는 rank-bm25 패키지 사용 (ARM 호환)
from rank_bm25 import BM25Okapi

# src/temporal_clustering.py
# scipy와 sklearn 사용 (ARM 호환)
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
```

**BERT tokenizer**가 모든 한국어 처리를 담당하므로 mecab이 필요 없습니다!

## 🎯 권장 설정

### Apple Silicon (M1/M2/M3)

```bash
# 1. Homebrew로 Python 설치
brew install python@3.12

# 2. venv 생성
python3.12 -m venv .venv
source .venv/bin/activate

# 3. 최소 의존성 설치
pip install --upgrade pip
pip install -r requirements-minimal.txt

# 4. 테스트
python test_korean_neural_sparse.py
```

### ARM Linux 서버

```bash
# 1. Python 3.10+ 설치
sudo apt-get update
sudo apt-get install python3.12 python3.12-venv python3-pip

# 2. venv 생성
python3.12 -m venv .venv
source .venv/bin/activate

# 3. 개발 도구 설치 (scipy 컴파일용)
sudo apt-get install build-essential python3-dev

# 4. 의존성 설치
pip install -r requirements-minimal.txt

# 5. 테스트
python test_korean_neural_sparse.py
```

## ✅ 검증 체크리스트

설치 후 다음을 확인하세요:

- [ ] Python 3.10+ 버전 확인
- [ ] venv 활성화됨
- [ ] torch import 성공
- [ ] transformers import 성공
- [ ] src.losses import 성공
- [ ] test_korean_neural_sparse.py 실행 성공
- [ ] test_temporal_features.py 실행 성공

## 📚 추가 리소스

- PyTorch ARM 설치: https://pytorch.org/get-started/locally/
- Transformers 문서: https://huggingface.co/docs/transformers
- 프로젝트 이슈: GitHub Issues에 보고

---

**요약**: ARM 시스템에서는 `requirements-minimal.txt`를 사용하세요. mecab/konlpy 없이 모든 핵심 기능이 작동합니다!
