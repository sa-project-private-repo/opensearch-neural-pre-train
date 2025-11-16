# Korean Wikipedia Data Loading Guide

이 문서는 최신 한국어 Wikipedia 데이터를 로드하는 방법을 설명합니다.

## 📚 두 가지 데이터 소스

### Option 1: HuggingFace Dataset (기본, 권장)
- **날짜**: 2023-11-01
- **장점**: 빠름, 안정적
- **단점**: 2년 전 데이터
- **사용 시기**: 테스트, 빠른 프로토타이핑

### Option 2: Latest Wikimedia Dump
- **날짜**: 2025-11-01 (최신)
- **장점**: 가장 최신 데이터
- **단점**: 첫 실행시 느림 (20-60분)
- **사용 시기**: 프로덕션, 최신 데이터 필요

## 🚀 사용 방법

### 노트북에서 사용

```python
from src.wikipedia_loader import load_korean_wikipedia

# Option 1: 빠른 로딩 (HuggingFace, 2023 데이터)
docs = load_korean_wikipedia(
    max_documents=100000,
    use_latest=False  # 빠름
)

# Option 2: 최신 데이터 (첫 실행시 느림, 이후 캐시됨)
docs = load_korean_wikipedia(
    max_documents=100000,
    use_latest=True  # 최신, 캐시 지원
)
```

### 커맨드 라인에서 사용

```bash
# 테스트: 100개 문서만
python download_latest_wikipedia.py --test

# 전체: 100,000개 문서
python download_latest_wikipedia.py --max-docs 100000

# 다운로드만 (나중에 파싱)
python download_latest_wikipedia.py --max-docs 0

# 파싱만 (이미 다운로드된 경우)
python download_latest_wikipedia.py --skip-download --max-docs 100000
```

## 📊 성능 비교

| 옵션 | 첫 실행 시간 | 이후 실행 | 데이터 날짜 | 용량 |
|------|-------------|----------|------------|------|
| HuggingFace | ~5-10분 | ~30초 | 2023-11-01 | ~400MB |
| Latest Dump | ~20-60분 | ~30초 | 2025-11-01 | ~1-2GB |

## 💾 캐싱

Latest Dump 사용시 자동 캐싱:
- **위치**: `dataset/wikipedia_dumps/`
- **형식**: JSON (압축 해제됨)
- **재사용**: 두 번째 실행부터 즉시 로드

캐시 삭제:
```bash
rm -rf dataset/wikipedia_dumps/
```

## 🔧 고급 사용법

### 캐시 강제 갱신
```python
docs = load_korean_wikipedia(
    max_documents=100000,
    use_latest=True,
    force_download=True  # 기존 캐시 무시
)
```

### 최소 문서 길이 설정
```python
docs = load_korean_wikipedia(
    max_documents=100000,
    min_length=200,  # 200자 이상만
    use_latest=True
)
```

### 사용 가능한 소스 정보 확인
```python
from src.wikipedia_loader import get_wikipedia_info
import json

info = get_wikipedia_info()
print(json.dumps(info, indent=2, ensure_ascii=False))
```

## 📝 노트북 1 업데이트 내용

기존:
```python
ko_wiki = load_dataset("wikipedia", "20220301.ko", split="train[:100000]")
```

변경 후:
```python
from src.wikipedia_loader import load_korean_wikipedia

ko_wiki_docs = load_korean_wikipedia(
    max_documents=100000,
    use_latest=False  # 또는 True
)
```

## 🐛 문제 해결

### 다운로드 실패
```
❌ Download failed: HTTP Error 404
```
**해결**: HuggingFace 데이터로 자동 폴백됨

### 파싱 오류
```
❌ Parsing failed: ...
```
**해결**:
1. `mwparserfromhell` 설치 확인
2. 캐시 삭제 후 재시도
3. HuggingFace 옵션 사용

### 메모리 부족
```
MemoryError: ...
```
**해결**: `max_documents` 값 줄이기

## 📦 필요한 패키지

```bash
pip install datasets mwparserfromhell tqdm
```

모두 `requirements.txt`에 포함되어 있습니다.

## 🎯 권장 사항

### 개발/테스트
```python
docs = load_korean_wikipedia(max_documents=10000, use_latest=False)
```

### 프로덕션
```python
docs = load_korean_wikipedia(max_documents=100000, use_latest=True)
```

## 📚 추가 리소스

- [Wikimedia Dumps](https://dumps.wikimedia.org/kowiki/)
- [HuggingFace Datasets](https://huggingface.co/datasets/wikimedia/wikipedia)
- [mwparserfromhell](https://github.com/earwig/mwparserfromhell)
