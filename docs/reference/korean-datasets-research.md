# 한국어 데이터셋 확충 조사 보고서

SPLADE 기반 한국어 Neural Sparse 검색 모델의 성능 개선을 위해 추가 수집 가능한
한국어 데이터셋을 조사한 결과를 정리한다. 후속 구현 작업은 본 보고서의 우선
순위와 통합 매핑을 참조하여 `scripts/collect_korean_datasets.py` 와
`src/preprocessing/downloaders/` 를 확장한다.

## 1. 현재 코퍼스 인벤토리

### 1.1 `scripts/collect_korean_datasets.py` (13종)

| 이름 | HF 경로 | pair_type | 용도 |
|---|---|---|---|
| korquad2 | squad_kor_v2 | qa_long | QA, long context |
| klue_mrc | klue/mrc | qa_mrc | MRC |
| klue_sts | klue/sts | sts_similarity | STS |
| klue_nli | klue/nli | nli_entailment | NLI |
| koalpaca | beomi/KoAlpaca-v1.1a | instruction | instruction |
| open_orca_ko | kyujinpy/OpenOrca-KO | instruction | instruction |
| mc4_ko | mc4 (ko) | web_passage | 웹 passage |
| wikipedia_ko | wikimedia/wikipedia 20231101.ko | wiki_passage | 위키 |
| opus_en_ko | Helsinki-NLP/opus-100 | parallel | 병렬 |
| ko_triplet | nlpai-lab/ko-triplet-v1.0 | retrieval_triplet | **hard-neg 보유** |
| ko_wikidata_qa | maywell/ko_wikidata_QA | wikidata_qa | QA |
| ko_alpaca_bingsu | Bingsu/ko_alpaca_data | instruction | instruction |
| (ko_strategyqa) | - | - | 스킵(너무 작음) |

### 1.2 `src/preprocessing/downloaders/` (9종)

| 이름 | 클래스 | 출처 | 용도 |
|---|---|---|---|
| kor_nli | KorNLIDownloader | kor_nli | NLI |
| klue_nli | KLUENLIDownloader | klue/nli | NLI |
| kor_sts | KorSTSDownloader | kor_sts | STS |
| korquad | KorQuADDownloader | squad_kor_v1 | QA |
| klue_mrc | KLUEMRCDownloader | klue/mrc | MRC |
| nsmc | NSMCDownloader | nsmc | 감성분류 |
| ynat | YNATDownloader | klue/ynat | 뉴스토픽 |
| korean_instructions | KoreanInstructionsDownloader | instruction | instruction |
| persona_chat | PersonaChatDownloader | dialog | 대화 |

### 1.3 관련 기존 자산

- `scripts/prepare_korean_mlm_data.py` — Wikipedia/mC4 기반 MLM 코퍼스
- `scripts/mine_aihub_negatives.py` — AI Hub JSONL 용 BGE-M3 hard-neg 마이너
- `scripts/mine_hard_negatives.py`, `scripts/mine_multi_negatives.py`
- `scripts/finetune_doc2query.py`, `scripts/expand_documents.py`
- `benchmark/hf_data_loader.py` — ko-strategyqa, miracl-ko, mrtydi-ko, ecom-ko

## 2. 갭 분석

| 축 | 현재 상태 | 부족한 점 |
|---|---|---|
| **도메인** | 위키/뉴스/지시문 중심 | 법률·의료·금융·이커머스·IT·행정 문서 부재 |
| **형식** | `(query, positive)` 중심 | 트리플렛·리스트와이즈·점수 라벨 부족 |
| **Hard Negative** | ko-triplet 744K 만 제공 | 검색 전용 하드 네거티브 데이터가 단일 소스 |
| **문서 길이** | 짧은 응답/문장 위주 | 긴 passage(512+ tok)와 멀티 문단 부족 |
| **라이선스** | HF 공개 위주 | 대규모 AI Hub 자산 미통합 |
| **다국어 혼합** | OPUS-100 만 병렬 | 영-한 retrieval 트리플렛 없음 |
| **벤치마크 정렬** | 학습-평가 분리 | MIRACL/Mr.TyDi 학습 split 미사용 |

이 갭들이 V28~V35 벤치마크 정체의 주요 원인일 가능성이 높다. 특히
*검색 도메인의 hard negative* 가 가장 시급하다.

## 3. 신규 후보 카탈로그

### 3.1 그룹 A — HuggingFace 즉시 통합 (P0/P1)

| # | 데이터셋 | 출처 | 라이선스 | 규모 | pair_type | 통합 지점 | 우선순위 |
|---|---|---|---|---|---|---|---|
| A1 | `miracl/miracl` (ko train) | [HF](https://huggingface.co/datasets/miracl/miracl) | Apache-2.0 | ~12K queries / 1.4M passages | retrieval_triplet | `collect_korean_datasets.py` 신규 `collect_miracl_ko()` + hard-neg 마이닝 | **P0** |
| A2 | `castorini/mr-tydi` (korean train) | [HF](https://huggingface.co/datasets/castorini/mr-tydi) | Apache-2.0 | ~1.3K train queries + corpus | retrieval_triplet | 동 | **P0** |
| A3 | `williamjeong2/msmarco-triplets-ko-v1/v2` | [HF](https://huggingface.co/datasets/williamjeong2/msmarco-triplets-ko-v1) | MIT (원본 MS MARCO) | ~500K+ 트리플렛 | retrieval_triplet | 신규 `collect_msmarco_ko()` | **P0** |
| A4 | `unicamp-dl/mmarco` (korean) | [HF](https://huggingface.co/datasets/unicamp-dl/mmarco) | Apache-2.0 | ~500K queries 번역 | retrieval_triplet | 동 | **P0** |
| A5 | `nlpai-lab/kullm-v2` | HF: `nlpai-lab/kullm-v2` | CC-BY-NC | ~150K instruction | instruction | `collect_korean_datasets.py` 추가 | P1 |
| A6 | `heegyu/kowiki-sentences` | HF: `heegyu/kowiki-sentences` | CC-BY-SA | 수백만 문장 | wiki_passage | MLM prep + BM25 페어 생성 | P1 |
| A7 | `lcw99/wikipedia-korean-20240501` | HF: `lcw99/wikipedia-korean-20240501` | CC-BY-SA | 최신 스냅샷 | wiki_passage | `wikipedia_ko` 교체 | P1 |
| A8 | `BCCard/BCCard-Finance-Kor-QnA` | HF: `BCCard/BCCard-Finance-Kor-QnA` | 상업 가능(사용 약관 확인) | ~수천 QA | domain_qa_finance | 신규 downloader 클래스 | P1 |
| A9 | `mteb/KLUE-TC`, `mteb/KLUE-YNAT` retrieval variants | HF: `mteb/KLUE-TC` | CC-BY-SA | ~수만 | topic_match | eval + 증강 | P2 |
| A10 | `MarkrAI/KoCommercial-Dataset` | HF: `MarkrAI/KoCommercial-Dataset` | 상업 가능 | ~1.2M instruction | instruction | `collect_korean_datasets.py` | P1 |
| A11 | `maywell/ko_Ultrafeedback_binarized` | HF: `maywell/ko_Ultrafeedback_binarized` | MIT | ~수만 | preference→triplet | 변환 후 triplet | P2 |
| A12 | `nayohan/aihub-en-ko-translation-1.2m` | HF: `nayohan/aihub-en-ko-translation-1.2m` | AI Hub 약관 | 1.2M 병렬 | parallel | `collect_opus_en_ko` 확장 | P2 |
| A13 | `kozistr/korean-triplet-v1` 류 (있는 경우) | HF 탐색 | 확인 필요 | - | retrieval_triplet | 동 | P2 |

### 3.2 그룹 B — AI Hub (라이선스 신청 필요, P0/P1)

AI Hub(한국지능정보사회진흥원)는 대규모·도메인 특화 한국어 코퍼스를 제공하며
본 프로젝트에 이미 `aihubshell` 바이너리와 `scripts/mine_aihub_negatives.py`
파이프라인이 존재하지만 실제 자산은 미통합 상태다.

| # | 데이터셋 | AI Hub 카테고리 | 규모(추정) | pair_type | 통합 방식 | 우선순위 |
|---|---|---|---|---|---|---|
| B1 | 행정 문서 대상 기계독해 | [기계독해](https://aihub.or.kr/aidata/86) | 45만 QA | qa_mrc, domain_qa_public | `aihubshell` → `mine_aihub_negatives.py` | **P0** |
| B2 | 뉴스 기사 기계독해 | MRC | 수십만 | qa_mrc | 동 | **P0** |
| B3 | 도서자료 기계독해 | MRC | 수십만 | qa_long | 동 | P1 |
| B4 | 법률 지식베이스 QA | 전문분야 KB | 수만 | domain_qa_legal | 동 + 법률 토큰 IDF | **P0** |
| B5 | 금융/약관 QA | 전문분야 KB | 수만 | domain_qa_finance | 동 | **P0** |
| B6 | 의료 QA / EMR 요약 | 보건의료 | 수만 | domain_qa_medical | 동 | P1 |
| B7 | 특허/논문 요약 | 전문분야 | 수만 | qa_long | 동 | P1 |
| B8 | 일반상식 문장 생성 | 언어 | 수십만 | instruction | 변환 | P2 |
| B9 | 전문분야(IT/제조) 말뭉치 | 산업 | 대규모 | web_passage | MLM + doc2query | P2 |

**통합 파이프라인(B 공통):**
```
aihubshell 다운로드 → JSONL 정규화 → mine_aihub_negatives.py (BGE-M3)
  → data/v30.0/{dataset}.jsonl (query, docs, scores) → SPLADE 학습
```

### 3.3 그룹 C — 합성/증강 (P1)

기존 코드 자산을 활용해 무라이선스 부담으로 생성 가능한 데이터:

| # | 방법 | 입력 | 출력 | 기존 도구 | 우선순위 |
|---|---|---|---|---|---|
| C1 | doc2query 생성 | Wikipedia-ko, mC4-ko | (synthetic_query, passage) | `scripts/finetune_doc2query.py` | P1 |
| C2 | Document expansion | 짧은 passage | 확장된 positive | `scripts/expand_documents.py` | P1 |
| C3 | LLM 질의 생성 | 도메인 passage | 자연 질의 | 외부 API + 기존 jsonl 스키마 | P1 |
| C4 | BM25 + BGE-M3 재랭크 hard-neg | 기존 (query, positive) | 트리플렛 | `scripts/mine_hard_negatives.py` | **P0** |
| C5 | Cross-lingual pairing (en↔ko) | OPUS, translation | 이중언어 retrieval | 기존 `collect_opus_en_ko` 확장 | P2 |

## 4. 우선순위 매트릭스

영향도(I) × 통합 비용(C⁻¹) × 라이선스 안정성(L) 기준. 점수가 높을수록 먼저
진행.

| 우선순위 | 항목 | 근거 |
|---|---|---|
| **P0** | A1 MIRACL-ko train, A2 Mr.TyDi-ko train | 벤치마크와 동일 분포, Apache-2.0, 즉시 사용 |
| **P0** | A3/A4 MS MARCO-ko, mMARCO | 대규모 트리플렛으로 hard-neg 학습에 직접 기여 |
| **P0** | B1 AI Hub 행정 MRC, B2 뉴스 MRC | 45만 규모, 도메인 확장 + 기존 mine 스크립트 재사용 |
| **P0** | B4/B5 법률·금융 KB QA | 현재 완전히 부재한 도메인 |
| **P0** | C4 Hard-neg 마이닝 | 기존 (query,positive) 전부를 트리플렛으로 승격 |
| **P1** | A5~A10, B3/B6/B7, C1~C3 | 규모 또는 도메인 보완 |
| **P2** | A11~A13, B8/B9, C5 | 장기 개선, 보조적 |

## 5. 통합 매핑 (구현 지침)

### 5.1 `scripts/collect_korean_datasets.py` 확장

신규 collector 함수를 추가하고 `DATASET_COLLECTORS` 레지스트리(539행)에 등록.
`_make_record`(58행)와 `write_dataset`(543행) 재사용.

```
collect_miracl_ko()         → retrieval_triplet   (A1)
collect_mrtydi_ko()         → retrieval_triplet   (A2)
collect_msmarco_ko()        → retrieval_triplet   (A3/A4)
collect_kullm_v2()          → instruction         (A5)
collect_ko_commercial()     → instruction         (A10)
collect_wikipedia_ko_2024() → wiki_passage        (A7, 교체)
```

MIRACL/Mr.TyDi는 `positive_passages`/`negative_passages` 필드를 가지므로
`ko-triplet` 과 동일하게 `negative` 필드를 채워서 반환한다(`collect_ko_triplet`,
424~452행 패턴 참조).

### 5.2 `src/preprocessing/downloaders/` 확장

`BaseDownloader`/`RawSample`(base.py:9-24) 상속하는 신규 클래스:

```
src/preprocessing/downloaders/
  retrieval.py  ← 신규
    MiraclKoDownloader
    MrTydiKoDownloader
    MSMarcoKoDownloader
  domain.py     ← 신규
    BCCardFinanceQADownloader
    AIHubLegalQADownloader
    AIHubFinanceQADownloader
    AIHubMedicalQADownloader
```

`DOWNLOADER_REGISTRY`(__init__.py:30)에 추가하고 `get_downloader` 를 통해 접근.

### 5.3 AI Hub 파이프라인

1. `aihubshell -mode d -datasetkey <id>` 로 다운로드
2. 변환 스크립트 (신규 `scripts/convert_aihub_to_jsonl.py`) 로
   `{query, docs:[positive], scores}` 형식 생성
3. `scripts/mine_aihub_negatives.py --input_dir data/aihub/<dataset>`
   실행해 BGE-M3 hard-neg 추가
4. `data/v30.0/aihub_*.jsonl` 에 저장
5. 신규 config `configs/pretrain_korean_v2.yaml` 의 `train_files` 에 패턴 추가

### 5.4 신규 `pair_type` 어휘 확장

기존 `pair_type` 에 도메인 축 추가:

```
qa_long, qa_mrc, sts_similarity, nli_entailment, instruction,
web_passage, wiki_passage, parallel, retrieval_triplet, wikidata_qa,
domain_qa_legal,   ← 신규
domain_qa_finance, ← 신규
domain_qa_medical, ← 신규
domain_qa_public,  ← 신규
synthetic_doc2query ← 신규 (C1/C2)
```

커리큘럼/샘플러에서 도메인별 가중치 조정에 사용(`train_v35_phase*.yaml`
패턴 참고).

### 5.5 학습 config 연결

- `configs/pretrain_korean.yaml` 의 `train_patterns` 에 신규 JSONL 추가
- 신규 작성: `configs/pretrain_korean_v2.yaml` — 도메인 균형 샘플링 적용
- `configs/train_v35_phase1.yaml` 패턴으로 BGE-M3 KD 병행

## 6. 검증 절차

### 6.1 데이터 품질 검증 (수집 단계)

1. `is_korean_text` (collect_korean_datasets.py:39) 로 한국어 비율 ≥ 0.3 확인
2. 중복 제거: `prepare_korean_mlm_data.py` 의 `text_hash` 로직 재사용
3. 길이 분포 체크 (query < 200 chars, passage 50~2048 chars)
4. hard-neg 유효성: BGE-M3 score < positive score

### 6.2 학습 효과 검증

`benchmark/runner.py` 로 다음 4개 벤치마크 실행 (hf_data_loader.py:27-50):

| 벤치마크 | 메트릭 | V28 baseline | 목표(Δ) |
|---|---|---|---|
| ko-strategyqa | Recall@1, NDCG@10 | 30.4% / - | +10%p |
| miracl-ko (dev) | NDCG@10, Recall@100 | TBD | +3%p |
| mrtydi-ko (test) | MRR@100, Recall@100 | TBD | +3%p |
| ecom-ko | NDCG@10 | TBD | +5%p |

### 6.3 회귀 방지

- `make eval-v28-language` → 한국어 토큰 비율 ≥ 85% 유지
- FLOPS 희소도 측정 → V28 수준 유지
- 비한국어 토큰 활성 비율 회귀 없음

### 6.4 A/B 프로토콜

각 신규 데이터 그룹을 단계별로 투입하고 (`pretrain_korean_v2` → `_v3`...)
벤치마크 델타를 기록. 음(陰) 효과 데이터는 제거.

```
stage 0: V28 baseline
stage 1: + A1,A2,A3,A4 (HF retrieval)       → Δ 측정
stage 2: + C4 (hard-neg 마이닝 전체 데이터)  → Δ 측정
stage 3: + B1,B2 (AI Hub 공공 MRC)           → Δ 측정
stage 4: + B4,B5 (AI Hub 법률/금융)          → Δ 측정
stage 5: + C1 (doc2query 합성)               → Δ 측정
```

각 stage 에서 한국어 비율 회귀/FLOPS 회귀가 관찰되면 해당 데이터의
커리큘럼 가중치를 조정하거나 제외한다.

## 7. 후속 작업 (본 조사 범위 외)

이 보고서 승인 후 별도 plan 에서 순차 실행:

1. **P0 데이터 수집 구현** — collector/downloader 코드 추가
2. **AI Hub 라이선스 신청** — 법률/금융/행정 MRC 4건
3. **Hard-neg 마이닝 대규모 실행** — 기존 코퍼스 전체
4. **학습 config v2 작성 및 stage 1 학습**
5. **벤치마크 리포트 및 회귀 확인**

## 참고 링크

- [MIRACL Dataset (HuggingFace)](https://huggingface.co/datasets/miracl/miracl)
- [Mr. TyDi (HuggingFace)](https://huggingface.co/datasets/castorini/mr-tydi)
- [williamjeong2/msmarco-triplets-ko-v1](https://huggingface.co/datasets/williamjeong2/msmarco-triplets-ko-v1)
- [unicamp-dl/mmarco](https://huggingface.co/datasets/unicamp-dl/mmarco)
- [AI Hub 기계독해](https://aihub.or.kr/aidata/86)
- [AwesomeKorean_Data (songys)](https://github.com/songys/AwesomeKorean_Data)
- [mteb/KLUE-TC](https://huggingface.co/datasets/mteb/KLUE-TC)
- [SPLADE 공식 레포](https://github.com/naver/splade)
- [Training Sparse Encoders (HF Blog)](https://huggingface.co/blog/train-sparse-encoder)
