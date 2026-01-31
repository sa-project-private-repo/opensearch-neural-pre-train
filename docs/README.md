# Korean Neural Sparse Model Documentation

## 개요
Korean Neural Sparse Model V28 기술 문서입니다.

## 문서 구조

### Concepts (개념)
- [01-neural-sparse-overview.md](./concepts/01-neural-sparse-overview.md) - Neural Sparse 검색 개요
- [02-splade-architecture.md](./concepts/02-splade-architecture.md) - SPLADE 아키텍처 상세
- [03-model-operation.md](./concepts/03-model-operation.md) - 모델 동작 원리
- [04-loss-functions.md](./concepts/04-loss-functions.md) - 손실 함수 상세

### Guides (가이드)
- [training-guide.md](./guides/training-guide.md) - 학습 가이드
- [opensearch-integration.md](./guides/opensearch-integration.md) - OpenSearch 통합 가이드
- [model-loading-guide.md](./guides/model-loading-guide.md) - 모델 로딩 가이드

### Reference (참조)
- [hyperparameters.md](./reference/hyperparameters.md) - 하이퍼파라미터 참조
- [korean-stopwords.md](./reference/korean-stopwords.md) - 한국어 불용어 처리

### Experiments (실험)
- [V28_EXPERIMENT.md](../experiments/V28_EXPERIMENT.md) - V28 실험 문서

### Archive
- 이전 문제 해결 기록 (./archive/)

## Version History

| Version | Key Features | Status |
|---------|--------------|--------|
| V22 | KoBERT backbone, curriculum learning | Completed |
| V24 | XLM-RoBERTa + BGE-M3 teacher | Completed |
| V25 | IDF-aware FLOPS | Completed |
| V26 | Enhanced IDF + Special Token Fix | Completed |
| V27 | Travel Domain Data | Training |
| **V28** | **Korean Language Filter + Context Gate** | **Ready** |

## Quick Start
1. concepts/01-neural-sparse-overview.md - 개념 이해
2. guides/training-guide.md - 학습 시작
3. guides/opensearch-integration.md - 배포
