# Vocabulary 크기 불일치 문제 해결

## 🐛 문제 원인

Special tokens 추가 후 vocabulary 크기가 변경되었지만, 일부 코드에서 여전히 구 크기를 사용:

```python
tokenizer.add_tokens(TECHNICAL_SPECIAL_TOKENS)  # 33개 추가
# Vocab: 32,000 → 32,033

# 문제가 발생한 부분
tokenizer.vocab_size  # ❌ 여전히 32,000 반환
len(tokenizer)        # ✅ 32,033 반환 (정확함)
```

## ⚠️ 발생한 에러

```python
RuntimeError: The size of tensor a (32033) must match the size of tensor b (32000)
```

- `doc_sparse`: 32,033 차원 (모델 출력)
- `query_sparse`: 32,000 차원 (IDF lookup) ← 문제!

## ✅ 해결 방법

**5개 위치에서 수정** (4개 셀):

1. **Cell 7**: Tokenizer info 출력
   ```python
   # Before
   print(f"Vocab size: {tokenizer.vocab_size:,}")
   
   # After
   print(f"Vocab size: {len(tokenizer):,}")
   ```

2. **Cell 40**: `compute_query_representation` 함수 ⭐ **가장 중요**
   ```python
   # Before
   vocab_size = tokenizer.vocab_size  # 32000
   
   # After
   vocab_size = len(tokenizer)  # 32033
   ```

3. **Cell 44**: 모델 저장 config
   ```python
   # Before
   'vocab_size': tokenizer.vocab_size
   
   # After
   'vocab_size': len(tokenizer)
   ```

4. **Cell 46**: config.json 생성 (2곳)
   ```python
   # Before
   "vocab_size": tokenizer.vocab_size
   "embedding_dimension": {tokenizer.vocab_size}
   
   # After
   "vocab_size": len(tokenizer)
   "embedding_dimension": {len(tokenizer)}
   ```

5. **Cell 48**: Inference 테스트
   ```python
   # Before
   sparse_vec = np.zeros(tokenizer.vocab_size)
   
   # After
   sparse_vec = np.zeros(len(tokenizer))
   ```

## 📊 수정 결과

```
✓ 수정된 코드: 5개 위치 (4개 셀)
✓ 차원 일치: doc_sparse (32033) = query_sparse (32033)
✓ 학습 가능 상태
```

## 💡 중요 포인트

**항상 `len(tokenizer)` 사용:**
```python
# ❌ 잘못된 방법
vocab_size = tokenizer.vocab_size  # 원본 BERT vocab만

# ✅ 올바른 방법
vocab_size = len(tokenizer)  # 추가된 special tokens 포함
```

**이유:**
- `tokenizer.vocab_size`: 읽기 전용 속성, 원본 모델 vocabulary 크기
- `len(tokenizer)`: 현재 tokenizer의 실제 크기 (special tokens 포함)

## 🚀 다음 단계

학습 재시작:
```python
# 모든 차원이 32,033으로 일치됨
train_loss, ranking_loss, l0_loss, idf_penalty = train_epoch(...)
```

문제 해결 완료! ✅
