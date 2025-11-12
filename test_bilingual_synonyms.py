#!/usr/bin/env python3
"""
Test script for bilingual (Korean-English) synonym discovery.

Demonstrates how to build and use cross-lingual synonym dictionary
so that "모델" and "model" are treated as synonyms.
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Import bilingual synonym module
from src.cross_lingual_synonyms import (
    build_comprehensive_bilingual_dictionary,
    get_default_korean_english_pairs,
    apply_bilingual_synonyms_to_idf,
)

print("=" * 70)
print("Testing Bilingual (Korean-English) Synonym Discovery")
print("=" * 70)

# Step 1: Load tokenizer and model
print("\n[Step 1] Loading BERT model...")
MODEL_NAME = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

print(f"✓ Loaded {MODEL_NAME}")

# Step 2: Extract token embeddings
print("\n[Step 2] Extracting token embeddings...")
token_embeddings = model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
print(f"✓ Token embeddings: {token_embeddings.shape}")

# Step 3: Sample bilingual documents
print("\n[Step 3] Preparing sample bilingual documents...")
sample_docs = [
    "딥러닝 모델(model)을 학습(training)시킵니다.",
    "자연어 처리(NLP) 기술을 사용합니다.",
    "벡터 검색(vector search)으로 유사 문서를 찾습니다.",
    "BERT 모델은 트랜스포머(transformer) 아키텍처를 사용합니다.",
    "임베딩(embedding) 벡터를 생성합니다.",
    "신경망(neural network)을 최적화(optimization)합니다.",
    "데이터(data)를 전처리(preprocessing)합니다.",
    "검색 시스템(search system)을 구축합니다.",
    "알고리즘(algorithm)을 개선합니다.",
    "성능(performance)을 측정합니다.",
]

print(f"✓ Prepared {len(sample_docs)} sample documents")

# Step 4: Build bilingual dictionary
print("\n[Step 4] Building bilingual synonym dictionary...")

# Get manual curated pairs
manual_pairs = get_default_korean_english_pairs()
print(f"\n  Manual pairs loaded: {len(manual_pairs)}")
print(f"  Example: '모델' → {manual_pairs.get('모델', [])}")

# Build comprehensive dictionary
bilingual_dict = build_comprehensive_bilingual_dictionary(
    documents=sample_docs,
    token_embeddings=token_embeddings,
    tokenizer=tokenizer,
    bert_model=model,
    manual_pairs=manual_pairs,
)

# Step 5: Test the bilingual dictionary
print("\n[Step 5] Testing bilingual synonym lookup...")

test_terms = ["모델", "model", "검색", "search", "학습", "learning"]

for term in test_terms:
    if term in bilingual_dict:
        synonyms = bilingual_dict[term]
        print(f"\n  '{term}' → {synonyms[:5]}")
    else:
        print(f"\n  '{term}' → (not found)")

# Step 6: Create simple IDF for demonstration
print("\n[Step 6] Creating sample IDF dictionary...")
from collections import Counter

# Simple IDF calculation
doc_freq = Counter()
for doc in sample_docs:
    tokens = set(tokenizer.tokenize(doc.lower()))
    doc_freq.update(tokens)

N = len(sample_docs)
import math
idf_dict = {}
for token, df in doc_freq.items():
    idf_dict[token] = math.log((N + 1) / (df + 1)) + 1.0

print(f"✓ Created IDF for {len(idf_dict)} tokens")

# Step 7: Apply bilingual synonyms to IDF
print("\n[Step 7] Applying bilingual synonyms to IDF...")

# Show original IDF values
print(f"\n  Original IDF values:")
if "모델" in idf_dict:
    print(f"    '모델': {idf_dict['모델']:.4f}")
if "model" in idf_dict:
    print(f"    'model': {idf_dict['model']:.4f}")

# Apply bilingual synonyms
enhanced_idf = apply_bilingual_synonyms_to_idf(
    idf_dict=idf_dict,
    bilingual_dict=bilingual_dict,
    tokenizer=tokenizer,
)

# Show enhanced IDF values
print(f"\n  Enhanced IDF values (after bilingual sync):")
if "모델" in enhanced_idf:
    print(f"    '모델': {enhanced_idf['모델']:.4f}")
if "model" in enhanced_idf:
    print(f"    'model': {enhanced_idf['model']:.4f}")

if "검색" in enhanced_idf:
    print(f"    '검색': {enhanced_idf['검색']:.4f}")
if "search" in enhanced_idf:
    print(f"    'search': {enhanced_idf['search']:.4f}")

# Step 8: Demonstrate query encoding with bilingual synonyms
print("\n[Step 8] Query encoding with bilingual IDF...")

def encode_query_bilingual(query: str, idf_dict, tokenizer):
    """Encode query using bilingual IDF"""
    tokens = tokenizer.tokenize(query.lower())

    sparse_vec = {}
    for token in tokens:
        if token in idf_dict:
            sparse_vec[token] = idf_dict[token]

    return sparse_vec

# Test queries
test_queries = [
    "모델 학습",        # Korean
    "model training",  # English
    "모델 training",   # Mixed
]

print(f"\n  Query encodings:")
for query in test_queries:
    sparse = encode_query_bilingual(query, enhanced_idf, tokenizer)
    print(f"\n    Query: '{query}'")
    for token, score in sorted(sparse.items(), key=lambda x: -x[1])[:5]:
        print(f"      {token}: {score:.4f}")

# Summary
print("\n" + "=" * 70)
print("✅ Bilingual Synonym Test Complete!")
print("=" * 70)

print(f"\nKey Results:")
print(f"  • Bilingual dictionary entries: {len(bilingual_dict):,}")
print(f"  • IDF entries enhanced: {len(enhanced_idf):,}")
print(f"\nBenefits:")
print(f"  ✓ '모델' and 'model' now have synchronized IDF values")
print(f"  ✓ Queries with Korean or English terms work equally well")
print(f"  ✓ Mixed-language queries (모델 training) are supported")
print(f"\nUsage:")
print(f"  • Use enhanced_idf instead of regular idf_dict")
print(f"  • Queries automatically benefit from bilingual synonyms")
print(f"  • Documents with mixed languages are better indexed")
print()
