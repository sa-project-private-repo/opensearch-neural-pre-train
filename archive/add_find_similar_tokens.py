#!/usr/bin/env python3
"""
Add the missing find_similar_tokens function to the notebook
"""

import json

# Read notebook
with open('korean_neural_sparse_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find section 7.2 markdown cell
target_cell_idx = None
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell.get('source', []))
        if '### 7.2.' in source and '유사 토큰' in source:
            target_cell_idx = i
            print(f"Found section 7.2 at cell {i}")
            break

if target_cell_idx is None:
    print("Error: Could not find section 7.2")
    exit(1)

# Create the find_similar_tokens function cell
find_similar_tokens_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def find_similar_tokens(token, tokenizer, embeddings, top_k=10, threshold=0.75):\n",
        "    \"\"\"\n",
        "    주어진 토큰과 유사한 토큰들을 찾습니다.\n",
        "    \n",
        "    Args:\n",
        "        token: 검색할 토큰 (문자열)\n",
        "        tokenizer: Tokenizer\n",
        "        embeddings: Token embeddings (numpy array)\n",
        "        top_k: 반환할 유사 토큰 개수\n",
        "        threshold: 유사도 임계값 (0~1)\n",
        "    \n",
        "    Returns:\n",
        "        List of (token, similarity_score) tuples\n",
        "    \"\"\"\n",
        "    # 토큰 -> ID 변환\n",
        "    token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))\n",
        "    if not token_id:\n",
        "        return []\n",
        "    token_id = token_id[0]  # 첫 번째 토큰 ID 사용\n",
        "    \n",
        "    # Token embedding 추출\n",
        "    token_emb = embeddings[token_id]\n",
        "    \n",
        "    # 모든 토큰과의 코사인 유사도 계산\n",
        "    similarities = np.dot(embeddings, token_emb) / (\n",
        "        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(token_emb) + 1e-10\n",
        "    )\n",
        "    \n",
        "    # 상위 top_k+1 개 추출 (자기 자신 제외)\n",
        "    top_indices = np.argsort(similarities)[-(top_k+1):][::-1]\n",
        "    \n",
        "    similar_tokens = []\n",
        "    for idx in top_indices:\n",
        "        sim_score = float(similarities[idx])\n",
        "        if sim_score >= threshold and int(idx) != token_id:\n",
        "            similar_token = tokenizer.decode([int(idx)]).strip()\n",
        "            # 필터링: 빈 문자열, 특수문자만 있는 경우 제외\n",
        "            if similar_token and not similar_token in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']:\n",
        "                similar_tokens.append((similar_token, sim_score))\n",
        "    \n",
        "    return similar_tokens[:top_k]\n",
        "\n",
        "\n",
        "print(\"✓ find_similar_tokens 함수 정의 완료\")"
    ]
}

# Insert after section 7.2 header, before build_synonym_dict_from_corpus
cells.insert(target_cell_idx + 1, find_similar_tokens_cell)

# Update the section header to clarify it defines find_similar_tokens
cells[target_cell_idx]['source'] = ["### 7.2. 유사 토큰 발견 함수 (find_similar_tokens)"]

# Update the next section to 7.3
# Find the build_synonym_dict_from_corpus cell and update its header
for i in range(target_cell_idx + 2, len(cells)):
    if cells[i]['cell_type'] == 'markdown':
        source = ''.join(cells[i].get('source', []))
        if '### 7.3.' in source:
            # It's already 7.3, good
            break
        # If it's not labeled yet or has wrong number, create new markdown cell
        break

# Create/update section 7.3 header for build_synonym_dict_from_corpus
section_73_header = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 7.3. 코퍼스 기반 동의어 자동 발견 (build_synonym_dict_from_corpus)"
    ]
}

# Insert section 7.3 header before build_synonym_dict_from_corpus function
cells.insert(target_cell_idx + 2, section_73_header)

# Save notebook
nb['cells'] = cells
with open('korean_neural_sparse_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✓ find_similar_tokens function added successfully!")
print(f"  Total cells: {len(cells)}")
