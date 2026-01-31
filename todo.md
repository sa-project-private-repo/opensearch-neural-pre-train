# Neural Sparse Model Training TODO

## 현재 상태

- **V27**: 학습 진행 중
- **V28**: 코드 완료, 학습 대기

---

## 할일 목록

### Phase 1: V27 완료 대기

- [ ] V27 학습 완료 확인
  - `outputs/train_v27/training_complete.flag` 파일 생성 여부 확인
  - 또는 `pgrep -f 'src.train v27'`로 프로세스 확인

### Phase 2: V28 학습

- [ ] V28 학습 시작
  ```bash
  # 자동 실행 (V27 완료 대기 후 자동 시작)
  nohup ./scripts/run_v28_after_v27.sh > outputs/v28_auto.log 2>&1 &

  # 수동 실행
  make train-v28
  ```

- [ ] 학습 모니터링
  ```bash
  make logs-v28
  make tensorboard-v28
  ```

### Phase 3: V28 검증

- [ ] V28a 한국어 토큰 비율 검증
  ```bash
  make eval-v28-language
  ```
  - 목표: >85% 한국어 토큰 비율

- [ ] V28b 컨텍스트 구분율 검증
  ```bash
  make eval-v28-context
  ```
  - 목표: <50% 토큰 오버랩 (동일 키워드, 다른 컨텍스트)

### Phase 4: 벤치마크

- [ ] Ko-StrategyQA 벤치마크
  ```bash
  make benchmark-ko-strategyqa-v28
  ```
  - 목표: Recall@1 > 40%

- [ ] 전체 벤치마크 실행
  ```bash
  make benchmark-all
  ```

### Phase 5: 배포

- [ ] HuggingFace 모델 export
  ```bash
  python scripts/export_huggingface.py --model outputs/train_v28/best_model --output huggingface/v28
  ```

- [ ] HuggingFace Hub 업로드
  ```bash
  huggingface-cli upload sewoong/korean-neural-sparse-encoder huggingface/v28
  ```

- [ ] README 업데이트
  - V28 벤치마크 결과 추가
  - 버전 히스토리 업데이트

---

## 성공 기준

| 지표 | V26 현재 | V28 목표 |
|------|----------|----------|
| 한국어 토큰 비율 | ~10% | >85% |
| 다국어 노이즈 | 91% | <5% |
| 컨텍스트 구분율 | 0% | >60% |
| Ko-StrategyQA Recall@1 | 30.4% | >40% |

---

## 유용한 명령어

```bash
# V27 상태 확인
pgrep -f 'src.train v27' && echo "V27 running" || echo "V27 not running"

# V28 전체 파이프라인
make v28-pipeline

# 로그 확인
tail -f outputs/train_v28/training.log

# 체크포인트 확인
ls -la outputs/train_v28/checkpoint_*
```

---

## 참고 문서

- [experiments/V28_EXPERIMENT.md](experiments/V28_EXPERIMENT.md) - V28 실험 상세
- [configs/train_v28.yaml](configs/train_v28.yaml) - V28 학습 설정
- [scripts/run_v28_after_v27.sh](scripts/run_v28_after_v27.sh) - 자동화 스크립트
