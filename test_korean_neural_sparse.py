#!/usr/bin/env python3
"""
OpenSearch Inference-Free Neural Sparse 모델 테스트 스크립트
한국어 샘플 데이터로 빠른 테스트를 수행합니다.
"""

import os
import json
import math
import numpy as np
from collections import Counter
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

# Import new loss functions
from src.losses import (
    neural_sparse_loss_with_regularization,
    compute_sparsity_metrics,
)

print("=" * 60)
print("OpenSearch Inference-Free Neural Sparse Model Test")
print("=" * 60)

# 한국어 샘플 데이터셋
SAMPLE_DOCUMENTS = [
    "인공지능은 컴퓨터 시스템이 인간의 지능을 모방하는 기술입니다.",
    "머신러닝은 데이터로부터 패턴을 학습하는 인공지능의 한 분야입니다.",
    "딥러닝은 인공 신경망을 사용하여 복잡한 문제를 해결합니다.",
    "자연어 처리는 컴퓨터가 인간의 언어를 이해하고 처리하는 기술입니다.",
    "OpenSearch는 강력한 검색 및 분석 엔진으로 다양한 기능을 제공합니다.",
    "벡터 검색은 의미적 유사성을 기반으로 문서를 검색합니다.",
    "Neural sparse 검색은 희소 벡터를 사용하여 효율적인 검색을 제공합니다.",
    "한국어 자연어 처리는 형태소 분석과 품사 태깅을 포함합니다.",
    "트랜스포머 아키텍처는 현대 자연어 처리의 핵심 기술입니다.",
    "BERT 모델은 양방향 인코더를 사용하여 문맥을 이해합니다.",
    "GPT는 생성형 언어 모델로 다양한 텍스트를 생성할 수 있습니다.",
    "LLM은 대규모 언어 모델을 의미하며 ChatGPT가 대표적입니다.",
    "임베딩은 텍스트를 벡터 공간으로 변환하는 과정입니다.",
    "검색 엔진 최적화는 웹사이트의 가시성을 높이는 작업입니다.",
    "데이터베이스는 구조화된 정보를 저장하고 관리하는 시스템입니다.",
    "클라우드 컴퓨팅은 인터넷을 통해 컴퓨팅 리소스를 제공합니다.",
    "빅데이터 분석은 대량의 데이터에서 인사이트를 추출합니다.",
    "파이썬은 데이터 과학과 머신러닝에 널리 사용되는 언어입니다.",
    "알고리즘은 문제를 해결하기 위한 단계적 절차입니다.",
    "소프트웨어 개발은 프로그램을 설계하고 구현하는 과정입니다.",
]

SAMPLE_QUERIES = [
    ("인공지능 기술", "인공지능은 컴퓨터 시스템이 인간의 지능을 모방하는 기술입니다."),
    ("머신러닝 학습", "머신러닝은 데이터로부터 패턴을 학습하는 인공지능의 한 분야입니다."),
    ("OpenSearch 검색", "OpenSearch는 강력한 검색 및 분석 엔진으로 다양한 기능을 제공합니다."),
    ("neural sparse", "Neural sparse 검색은 희소 벡터를 사용하여 효율적인 검색을 제공합니다."),
    ("한국어 처리", "한국어 자연어 처리는 형태소 분석과 품사 태깅을 포함합니다."),
    ("BERT 모델", "BERT 모델은 양방향 인코더를 사용하여 문맥을 이해합니다."),
    ("GPT ChatGPT", "GPT는 생성형 언어 모델로 다양한 텍스트를 생성할 수 있습니다."),
    ("임베딩 벡터", "임베딩은 텍스트를 벡터 공간으로 변환하는 과정입니다."),
]

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"   CUDA: {torch.cuda.get_device_name(0)}")

# Step 1: 토크나이저 로드
print("\n" + "=" * 60)
print("Step 1: 토크나이저 로드")
print("=" * 60)

MODEL_NAME = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"✓ 토크나이저 로드: {MODEL_NAME}")
print(f"  Vocab size: {tokenizer.vocab_size:,}")

# Step 2: IDF 계산
print("\n" + "=" * 60)
print("Step 2: IDF (Inverse Document Frequency) 계산")
print("=" * 60)

def calculate_idf(documents, tokenizer):
    """IDF 계산"""
    N = len(documents)
    df = Counter()

    print(f"문서 {N}개에서 IDF 계산 중...")

    for doc in documents:
        tokens = tokenizer.encode(doc, add_special_tokens=False, max_length=128, truncation=True)
        unique_tokens = set(tokens)
        for token_id in unique_tokens:
            df[token_id] += 1

    # IDF 계산
    idf_dict = {}
    for token_id, doc_freq in df.items():
        idf_score = math.log((N + 1) / (doc_freq + 1)) + 1.0
        idf_dict[token_id] = idf_score

    # 토큰 문자열로 변환
    idf_token_dict = {}
    for token_id, score in idf_dict.items():
        token_str = tokenizer.decode([token_id])
        idf_token_dict[token_str] = float(score)

    print(f"✓ {len(idf_token_dict):,}개 토큰의 IDF 계산 완료")
    print(f"  평균 IDF: {np.mean(list(idf_token_dict.values())):.4f}")

    return idf_token_dict, idf_dict

idf_token_dict, idf_id_dict = calculate_idf(SAMPLE_DOCUMENTS, tokenizer)

# 트렌드 키워드 부스팅
print("\n트렌드 키워드 부스팅 적용 중...")
TREND_BOOST = {
    'LLM': 1.5, 'GPT': 1.5, 'ChatGPT': 1.5,
    '생성형': 1.4, 'RAG': 1.4, 'OpenSearch': 1.3,
    '검색': 1.2, '인공지능': 1.2, 'AI': 1.2,
    'BERT': 1.2, '임베딩': 1.3, 'neural': 1.3, 'sparse': 1.3,
}

boost_count = 0
for keyword, boost_factor in TREND_BOOST.items():
    keyword_tokens = tokenizer.encode(keyword, add_special_tokens=False)
    for token_id in keyword_tokens:
        token_str = tokenizer.decode([token_id])
        if token_str in idf_token_dict:
            idf_token_dict[token_str] *= boost_factor
            boost_count += 1

print(f"✓ {boost_count}개 토큰에 트렌드 부스팅 적용")

# Step 3: 모델 정의
print("\n" + "=" * 60)
print("Step 3: OpenSearch 문서 인코더 모델 정의")
print("=" * 60)

class OpenSearchDocEncoder(nn.Module):
    """OpenSearch Neural Sparse Document Encoder"""
    def __init__(self, model_name="klue/bert-base"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModelForMaskedLM.from_pretrained(model_name)
        self.vocab_size = self.config.vocab_size
        self.activation = lambda x: torch.log1p(torch.relu(x))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits
        activated = self.activation(logits)
        sparse_vector = torch.max(activated * attention_mask.unsqueeze(-1), dim=1).values
        return sparse_vector

doc_encoder = OpenSearchDocEncoder(MODEL_NAME)
doc_encoder = doc_encoder.to(device)

print(f"✓ 모델 초기화 완료")
print(f"  Parameters: {sum(p.numel() for p in doc_encoder.parameters()):,}")

# Step 4: 학습 데이터 준비
print("\n" + "=" * 60)
print("Step 4: 학습 데이터 준비")
print("=" * 60)

# Query-Document pairs 생성
qd_pairs = []
for query, pos_doc in SAMPLE_QUERIES:
    qd_pairs.append((query, pos_doc, 1.0))  # Positive

    # Negative sampling
    for neg_doc in SAMPLE_DOCUMENTS:
        if neg_doc != pos_doc:
            qd_pairs.append((query, neg_doc, 0.0))  # Negative
            break  # 1개만

print(f"✓ {len(qd_pairs)}개 query-document pairs 생성")
print(f"  Positive: {len(SAMPLE_QUERIES)}")
print(f"  Negative: {len(qd_pairs) - len(SAMPLE_QUERIES)}")

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, qd_pairs, tokenizer, max_length=64):
        self.qd_pairs = qd_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qd_pairs)

    def __getitem__(self, idx):
        query, document, relevance = self.qd_pairs[idx]

        query_encoded = self.tokenizer(
            query, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        doc_encoded = self.tokenizer(
            document, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        return {
            'query_input_ids': query_encoded['input_ids'].squeeze(0),
            'query_attention_mask': query_encoded['attention_mask'].squeeze(0),
            'doc_input_ids': doc_encoded['input_ids'].squeeze(0),
            'doc_attention_mask': doc_encoded['attention_mask'].squeeze(0),
            'relevance': torch.tensor(relevance, dtype=torch.float32)
        }

dataset = SimpleDataset(qd_pairs, tokenizer)
# Increase batch size for better in-batch negatives
BATCH_SIZE = 8  # Increased from 4
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"✓ 데이터 로더 생성 (batch_size={BATCH_SIZE})")

# Step 5: 손실 함수 정의
print("\n" + "=" * 60)
print("Step 5: 손실 함수 정의")
print("=" * 60)

def compute_query_representation(query_tokens, idf_dict, vocab_size):
    """IDF lookup으로 쿼리 sparse vector 생성 (Inference-Free!)"""
    batch_size, seq_len = query_tokens.shape
    query_sparse = torch.zeros(batch_size, vocab_size, device=query_tokens.device)

    for b in range(batch_size):
        for token_id in query_tokens[b]:
            token_id = token_id.item()
            if token_id in idf_dict:
                query_sparse[b, token_id] = idf_dict[token_id]

    return query_sparse

print("✓ 손실 함수 정의 완료")
print("  - NEW: In-Batch Negatives Contrastive Loss")
print("  - L0 Regularization (Sparsity)")
print("  - IDF-aware Penalty (optional)")
print("\n⚠️  FIXED: Replaced BCE with proper contrastive loss!")

# Step 6: 학습 실행
print("\n" + "=" * 60)
print("Step 6: 모델 학습 (간단한 테스트)")
print("=" * 60)

optimizer = AdamW(doc_encoder.parameters(), lr=5e-5)
NUM_EPOCHS = 2

# Loss hyperparameters
LAMBDA_L0 = 5e-4  # Reduced from 1e-3 to allow less sparsity
LAMBDA_IDF = 1e-2
TEMPERATURE = 0.05

print(f"학습 설정:")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Learning rate: 5e-5")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Temperature: {TEMPERATURE}")
print(f"  Lambda L0: {LAMBDA_L0}")
print(f"  Lambda IDF: {LAMBDA_IDF}")

for epoch in range(NUM_EPOCHS):
    doc_encoder.train()
    total_loss_sum = 0
    total_ranking_loss = 0
    total_l0_loss = 0
    total_idf_penalty = 0

    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

    for batch_idx, batch in enumerate(loader):
        query_tokens = batch['query_input_ids'].to(device)
        doc_input_ids = batch['doc_input_ids'].to(device)
        doc_attention_mask = batch['doc_attention_mask'].to(device)
        relevance = batch['relevance'].to(device)

        # Document encoding
        doc_sparse = doc_encoder(doc_input_ids, doc_attention_mask)

        # Query encoding (IDF lookup - Inference-Free!)
        query_sparse = compute_query_representation(
            query_tokens, idf_id_dict, tokenizer.vocab_size
        )

        # NEW: Use improved loss function with in-batch negatives
        total_loss, loss_dict = neural_sparse_loss_with_regularization(
            doc_sparse=doc_sparse,
            query_sparse=query_sparse,
            relevance=relevance,
            idf_dict=idf_id_dict,
            lambda_l0=LAMBDA_L0,
            lambda_idf=LAMBDA_IDF,
            temperature=TEMPERATURE,
            use_in_batch_negatives=True,  # Key improvement!
        )

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(doc_encoder.parameters(), 1.0)
        optimizer.step()

        # Accumulate losses
        total_loss_sum += total_loss.item()
        total_ranking_loss += loss_dict['ranking'].item()
        total_l0_loss += loss_dict['l0'].item()
        total_idf_penalty += loss_dict['idf_penalty'].item()

        if (batch_idx + 1) % 2 == 0:
            print(
                f"  Batch {batch_idx + 1}/{len(loader)} - "
                f"Total: {total_loss.item():.4f}, "
                f"Ranking: {loss_dict['ranking'].item():.4f}, "
                f"L0: {loss_dict['l0'].item():.4f}"
            )

    # Epoch summary
    num_batches = len(loader)
    avg_total = total_loss_sum / num_batches
    avg_ranking = total_ranking_loss / num_batches
    avg_l0 = total_l0_loss / num_batches
    avg_idf = total_idf_penalty / num_batches

    print(f"\n✓ Epoch {epoch + 1} 완료:")
    print(f"  Total Loss: {avg_total:.4f}")
    print(f"  Ranking Loss: {avg_ranking:.4f}")
    print(f"  L0 Loss: {avg_l0:.4f}")
    print(f"  IDF Penalty: {avg_idf:.4f}")

    # Compute sparsity metrics
    doc_encoder.eval()
    with torch.no_grad():
        sample_batch = next(iter(loader))
        sample_docs = doc_encoder(
            sample_batch['doc_input_ids'].to(device),
            sample_batch['doc_attention_mask'].to(device)
        )
        sparsity_metrics = compute_sparsity_metrics(sample_docs)
        print(f"  Sparsity: {sparsity_metrics['sparsity']:.2%}")
        print(f"  Avg non-zero tokens: {sparsity_metrics['non_zero_count_mean']:.1f}")

print("\n✓ 학습 완료!")

# Step 7: 모델 저장
print("\n" + "=" * 60)
print("Step 7: 모델 저장 (OpenSearch 호환 형식)")
print("=" * 60)

OUTPUT_DIR = "./models/test_korean_neural_sparse_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# pytorch_model.bin
torch.save(doc_encoder.state_dict(), f"{OUTPUT_DIR}/pytorch_model.bin")
print(f"✓ pytorch_model.bin 저장")

# idf.json
with open(f"{OUTPUT_DIR}/idf.json", 'w', encoding='utf-8') as f:
    json.dump(idf_token_dict, f, ensure_ascii=False, indent=2)
print(f"✓ idf.json 저장 ({len(idf_token_dict):,} tokens)")

# Tokenizer
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✓ Tokenizer 파일 저장")

# config.json
config = {
    "model_type": "opensearch-neural-sparse-doc-encoder",
    "base_model": MODEL_NAME,
    "vocab_size": tokenizer.vocab_size,
    "mode": "doc-only",
    "output_format": "rank_features",
    "test_info": {
        "documents": len(SAMPLE_DOCUMENTS),
        "queries": len(SAMPLE_QUERIES),
        "epochs": NUM_EPOCHS,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
}

with open(f"{OUTPUT_DIR}/config.json", 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
print(f"✓ config.json 저장")

print(f"\n모델 저장 위치: {OUTPUT_DIR}/")

# Step 8: 추론 테스트
print("\n" + "=" * 60)
print("Step 8: 추론 테스트")
print("=" * 60)

doc_encoder.eval()

def encode_document(text, model, tokenizer, device):
    """문서 인코딩 (모델 사용)"""
    encoded = tokenizer(text, max_length=64, padding='max_length',
                       truncation=True, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        sparse_vec = model(input_ids, attention_mask)

    return sparse_vec.cpu().numpy()[0]

def encode_query_inference_free(text, tokenizer, idf_dict):
    """쿼리 인코딩 (IDF lookup - Inference-Free!)"""
    tokens = tokenizer.encode(text, add_special_tokens=False, max_length=64, truncation=True)
    sparse_vec = np.zeros(tokenizer.vocab_size)

    for token_id in tokens:
        token_str = tokenizer.decode([token_id])
        if token_str in idf_dict:
            sparse_vec[token_id] = idf_dict[token_str]

    return sparse_vec

def get_top_tokens(sparse_vec, tokenizer, top_k=10):
    """상위 토큰 추출"""
    top_indices = np.argsort(sparse_vec)[-top_k:][::-1]
    top_values = sparse_vec[top_indices]

    return [(tokenizer.decode([idx]), val) for idx, val in zip(top_indices, top_values) if val > 0]

# 테스트 쿼리
test_queries = [
    "인공지능 기술",
    "OpenSearch 검색 엔진",
    "한국어 자연어 처리",
    "LLM ChatGPT",
]

print("\n📝 쿼리 인코딩 테스트 (Inference-Free)")
print("-" * 60)

for query in test_queries:
    sparse_vec = encode_query_inference_free(query, tokenizer, idf_token_dict)
    non_zero = np.count_nonzero(sparse_vec)

    print(f"\nQuery: {query}")
    print(f"  Non-zero: {non_zero}/{len(sparse_vec)} ({non_zero/len(sparse_vec)*100:.2f}%)")
    print(f"  상위 토큰:")

    top_tokens = get_top_tokens(sparse_vec, tokenizer, top_k=5)
    for i, (token, value) in enumerate(top_tokens, 1):
        print(f"    {i}. {token:15s} ({value:.4f})")

# 테스트 문서
test_docs = [
    "OpenSearch는 neural sparse 검색을 지원합니다.",
    "인공지능과 머신러닝은 데이터 과학의 핵심입니다.",
]

print("\n\n📄 문서 인코딩 테스트 (Model Inference)")
print("-" * 60)

for doc in test_docs:
    sparse_vec = encode_document(doc, doc_encoder, tokenizer, device)
    non_zero = np.count_nonzero(sparse_vec)
    l1_norm = np.sum(np.abs(sparse_vec))

    print(f"\nDocument: {doc}")
    print(f"  Non-zero: {non_zero}/{len(sparse_vec)} ({non_zero/len(sparse_vec)*100:.2f}%)")
    print(f"  L1 Norm: {l1_norm:.2f}")
    print(f"  상위 토큰:")

    top_tokens = get_top_tokens(sparse_vec, tokenizer, top_k=5)
    for i, (token, value) in enumerate(top_tokens, 1):
        print(f"    {i}. {token:15s} ({value:.4f})")

# Step 9: 검색 시뮬레이션
print("\n" + "=" * 60)
print("Step 9: 검색 시뮬레이션")
print("=" * 60)

# 모든 문서 인코딩
print("\n모든 샘플 문서 인코딩 중...")
doc_vectors = []
for doc in SAMPLE_DOCUMENTS:
    vec = encode_document(doc, doc_encoder, tokenizer, device)
    doc_vectors.append(vec)
doc_vectors = np.array(doc_vectors)

print(f"✓ {len(doc_vectors)}개 문서 인코딩 완료")

# 검색 테스트
search_queries = [
    "인공지능 머신러닝",
    "OpenSearch neural sparse 검색",
    "한국어 처리",
]

print("\n🔍 검색 결과:")
print("=" * 60)

for query in search_queries:
    print(f"\nQuery: '{query}'")

    # 쿼리 인코딩
    query_vec = encode_query_inference_free(query, tokenizer, idf_token_dict)

    # 유사도 계산 (dot product)
    similarities = np.dot(doc_vectors, query_vec)

    # 상위 3개 결과
    top_indices = np.argsort(similarities)[-3:][::-1]

    print("상위 3개 결과:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. [Score: {similarities[idx]:.4f}] {SAMPLE_DOCUMENTS[idx][:60]}...")

# 최종 요약
print("\n" + "=" * 60)
print("✅ 테스트 완료!")
print("=" * 60)

print(f"""
테스트 요약:
  ✓ 모델: {MODEL_NAME}
  ✓ 샘플 문서: {len(SAMPLE_DOCUMENTS)}개
  ✓ 샘플 쿼리: {len(SAMPLE_QUERIES)}개
  ✓ 학습 Epochs: {NUM_EPOCHS}
  ✓ IDF 토큰: {len(idf_token_dict):,}개
  ✓ 저장 위치: {OUTPUT_DIR}/

OpenSearch 통합:
  1. {OUTPUT_DIR}/를 압축하여 OpenSearch에 업로드
  2. Doc-only mode로 설정
  3. rank_features 타입 매핑 사용
  4. Neural sparse 쿼리 실행

다음 단계:
  - 더 많은 데이터로 재학습
  - Knowledge distillation 적용
  - BEIR 벤치마크 평가
  - OpenSearch 실제 배포
""")

print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
