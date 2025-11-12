#!/usr/bin/env python3
"""
Fix duplicate section numbering in section 7
"""

import json
import re

# Read notebook
with open('korean_neural_sparse_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find and renumber section 7 subsections
section_mapping = [
    ("### 7.1. BERT 토큰 임베딩 추출", "### 7.1. BERT 토큰 임베딩 추출"),
    ("### 7.2. 유사 토큰 발견 함수", "### 7.2. 유사 토큰 발견 함수 (find_similar_tokens)"),
    ("### 7.3. 코퍼스 기반 동의어 자동 발견 (build_synonym_dict_from_corpus)", "### 7.3. 코퍼스 기반 동의어 발견 함수"),
]

print("Renumbering section 7 subsections...")

# First pass: mark the first 7.3 as the function definition
found_first_73 = False
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])

        # The second 7.3 should become 7.4
        if '### 7.3. 수집 데이터 기반 동의어' in source:
            cell['source'] = ['### 7.4. 수집 데이터 기반 동의어 자동 발견']
            print(f"  Cell {i}: 7.3 -> 7.4 (수집 데이터 기반)")

        # 7.4 should become 7.5
        elif '### 7.4. Synonym-Aware IDF' in source:
            cell['source'] = ['### 7.5. Synonym-Aware IDF 생성']
            print(f"  Cell {i}: 7.4 -> 7.5")

        # 7.5 should become 7.6
        elif '### 7.5. 동의어 기반 학습 데이터' in source:
            cell['source'] = ['### 7.6. 동의어 기반 학습 데이터 확장']
            print(f"  Cell {i}: 7.5 -> 7.6")

        # 7.6 should become 7.7
        elif source.strip().startswith('### 7.6. 동의어 정보 요약'):
            cell['source'] = ['### 7.7. 동의어 정보 요약\n', '\n', '자동 발견된 동의어 정보를 요약합니다.']
            print(f"  Cell {i}: 7.6 -> 7.7")

# Save
nb['cells'] = cells
with open('korean_neural_sparse_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n✓ Section numbering fixed!")
print("\nFinal section 7 structure:")
print("  7.1. BERT 토큰 임베딩 추출")
print("  7.2. 유사 토큰 발견 함수 (find_similar_tokens)")
print("  7.3. 코퍼스 기반 동의어 발견 함수 (build_synonym_dict_from_corpus)")
print("  7.4. 수집 데이터 기반 동의어 자동 발견")
print("  7.5. Synonym-Aware IDF 생성")
print("  7.6. 동의어 기반 학습 데이터 확장")
print("  7.7. 동의어 정보 요약")
