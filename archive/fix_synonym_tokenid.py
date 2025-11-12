#!/usr/bin/env python3
"""
Fix build_synonym_dict_from_corpus to handle token IDs (integers) instead of token strings
"""

import json

# Read notebook
with open('korean_neural_sparse_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with build_synonym_dict_from_corpus function
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def build_synonym_dict_from_corpus' in source:
            print(f"Found build_synonym_dict_from_corpus at cell {i}")

            # Replace the function with fixed version
            new_source = '''def build_synonym_dict_from_corpus(documents, tokenizer, embeddings,
                                   idf_dict, top_n=500, threshold=0.75):
    """
    수집된 문서 코퍼스에서 중요 토큰들의 동의어를 자동 발견

    Args:
        documents: 문서 리스트
        tokenizer: Tokenizer
        embeddings: Token embeddings
        idf_dict: IDF 딕셔너리 (token_id -> idf_score or token_str -> idf_score)
        top_n: 상위 N개 IDF 토큰 대상
        threshold: 유사도 임계값

    Returns:
        synonym_dict: {token: [similar_tokens]}
    """
    print(f"\\n📖 수집된 데이터에서 중요 토큰 추출...")

    # IDF 기반 중요 토큰 선정
    sorted_idf = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)

    # 필터링: subword(##로 시작), 특수문자, 단일 문자 제외
    important_tokens = []
    for token_or_id, idf_score in sorted_idf:
        if len(important_tokens) >= top_n:
            break

        # token_or_id가 정수(token ID)인 경우 문자열로 변환
        if isinstance(token_or_id, int):
            token = tokenizer.decode([token_or_id]).strip()
        else:
            token = token_or_id

        # 필터링 조건
        if (not token.startswith('##') and
            len(token) > 1 and
            not token in [',', '.', '!', '?', ':', ';', '-', '(', ')', '[', ']']):
            important_tokens.append(token)

    print(f"  중요 토큰 {len(important_tokens)}개 선정 완료")
    print(f"  상위 10개: {important_tokens[:10]}")

    # 각 토큰에 대해 유사 토큰 찾기
    print(f"\\n🔎 유사 토큰 자동 발견 중... (threshold={threshold})")
    synonym_dict = {}

    for token in tqdm(important_tokens, desc="Finding synonyms"):
        similar = find_similar_tokens(token, tokenizer, embeddings,
                                      top_k=5, threshold=threshold)
        if similar:
            synonym_dict[token] = [t for t, _ in similar]

    return synonym_dict, important_tokens


print("✓ 동의어 발견 함수 정의 완료")'''

            cell['source'] = [new_source]
            print("✓ Function updated")
            break

# Save notebook
with open('korean_neural_sparse_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n✓ Notebook saved successfully!")
