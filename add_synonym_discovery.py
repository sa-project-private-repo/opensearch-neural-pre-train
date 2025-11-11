import json

# 노트북 읽기
with open('korean_neural_sparse_training.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# 새로 추가할 셀들
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. 자동 동의어 발견 및 데이터 확장\n",
            "\n",
            "수집된 데이터에서 토큰 임베딩 기반으로 동의어를 자동 발견하고,\n",
            "이를 활용하여 학습 데이터를 확장합니다."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### 5.1. BERT 토큰 임베딩 추출"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=\"*60)\n",
            "print(\"🔍 토큰 임베딩 기반 동의어 자동 발견\")\n",
            "print(\"=\"*60)\n",
            "\n",
            "# BERT 모델은 이미 doc_encoder에 로드되어 있음\n",
            "# Token embedding 추출\n",
            "token_embeddings = doc_encoder.bert.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()\n",
            "\n",
            "print(f\"✓ Token embeddings 추출 완료: {token_embeddings.shape}\")\n",
            "print(f\"  Vocab size: {token_embeddings.shape[0]:,}\")\n",
            "print(f\"  Embedding dim: {token_embeddings.shape[1]}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### 5.2. 유사 토큰 자동 발견 함수"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def find_similar_tokens(token, tokenizer, embeddings, top_k=10, threshold=0.75):\n",
            "    \"\"\"\n",
            "    주어진 토큰과 유사한 토큰들 찾기 (코사인 유사도 기반)\n",
            "    \n",
            "    Args:\n",
            "        token: 토큰 문자열\n",
            "        tokenizer: Tokenizer\n",
            "        embeddings: Token embeddings (vocab_size, embedding_dim)\n",
            "        top_k: 반환할 최대 개수\n",
            "        threshold: 최소 유사도 임계값\n",
            "    \n",
            "    Returns:\n",
            "        List of (token, similarity) tuples\n",
            "    \"\"\"\n",
            "    # 토큰 ID\n",
            "    token_id = tokenizer.convert_tokens_to_ids(token)\n",
            "    if token_id == tokenizer.unk_token_id:\n",
            "        return []\n",
            "    \n",
            "    # 해당 토큰의 임베딩\n",
            "    token_emb = embeddings[token_id]\n",
            "    \n",
            "    # 모든 토큰과의 코사인 유사도 계산\n",
            "    similarities = np.dot(embeddings, token_emb) / (\n",
            "        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(token_emb) + 1e-10\n",
            "    )\n",
            "    \n",
            "    # 상위 k개 추출\n",
            "    top_indices = np.argsort(similarities)[-top_k-1:-1][::-1]\n",
            "    \n",
            "    similar_tokens = []\n",
            "    for idx in top_indices:\n",
            "        sim_score = float(similarities[idx])\n",
            "        if sim_score >= threshold and int(idx) != token_id:\n",
            "            similar_token = tokenizer.decode([int(idx)])\n",
            "            similar_tokens.append((similar_token, sim_score))\n",
            "    \n",
            "    return similar_tokens\n",
            "\n",
            "\n",
            "def build_synonym_dict_from_corpus(documents, tokenizer, embeddings, \n",
            "                                   idf_dict, top_n=500, threshold=0.75):\n",
            "    \"\"\"\n",
            "    수집된 문서 코퍼스에서 중요 토큰들의 동의어를 자동 발견\n",
            "    \n",
            "    Args:\n",
            "        documents: 문서 리스트\n",
            "        tokenizer: Tokenizer\n",
            "        embeddings: Token embeddings\n",
            "        idf_dict: IDF 딕셔너리\n",
            "        top_n: 상위 N개 IDF 토큰 대상\n",
            "        threshold: 유사도 임계값\n",
            "    \n",
            "    Returns:\n",
            "        synonym_dict: {token: [similar_tokens]}\n",
            "    \"\"\"\n",
            "    print(f\"\\n📖 수집된 데이터에서 중요 토큰 추출...\")\n",
            "    \n",
            "    # IDF 기반 중요 토큰 선정\n",
            "    sorted_idf = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)\n",
            "    \n",
            "    # 필터링: subword(##로 시작), 특수문자, 단일 문자 제외\n",
            "    important_tokens = []\n",
            "    for token, idf_score in sorted_idf:\n",
            "        if len(important_tokens) >= top_n:\n",
            "            break\n",
            "        \n",
            "        # 필터링 조건\n",
            "        if (not token.startswith('##') and \n",
            "            len(token) > 1 and \n",
            "            not token in [',', '.', '!', '?', ':', ';', '-', '(', ')', '[', ']']):\n",
            "            important_tokens.append(token)\n",
            "    \n",
            "    print(f\"  중요 토큰 {len(important_tokens)}개 선정 완료\")\n",
            "    print(f\"  상위 10개: {important_tokens[:10]}\")\n",
            "    \n",
            "    # 각 토큰에 대해 유사 토큰 찾기\n",
            "    print(f\"\\n🔎 유사 토큰 자동 발견 중... (threshold={threshold})\")\n",
            "    synonym_dict = {}\n",
            "    \n",
            "    for token in tqdm(important_tokens, desc=\"Finding synonyms\"):\n",
            "        similar = find_similar_tokens(token, tokenizer, embeddings, \n",
            "                                      top_k=5, threshold=threshold)\n",
            "        if similar:\n",
            "            synonym_dict[token] = [t for t, _ in similar]\n",
            "    \n",
            "    return synonym_dict, important_tokens\n",
            "\n",
            "\n",
            "print(\"✓ 동의어 발견 함수 정의 완료\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### 5.3. 수집 데이터 기반 동의어 자동 발견"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 수집된 문서에서 동의어 자동 발견\n",
            "auto_synonym_dict, important_tokens = build_synonym_dict_from_corpus(\n",
            "    korean_data['documents'],\n",
            "    tokenizer,\n",
            "    token_embeddings,\n",
            "    idf_id_dict,  # ID 기반 IDF\n",
            "    top_n=500,    # 상위 500개 토큰\n",
            "    threshold=0.75  # 유사도 75% 이상\n",
            ")\n",
            "\n",
            "print(f\"\\n{'='*60}\")\n",
            "print(f\"✓ 자동 동의어 발견 완료!\")\n",
            "print(f\"{'='*60}\")\n",
            "print(f\"  발견된 동의어 그룹: {len(auto_synonym_dict):,}개\")\n",
            "print(f\"  총 동의어 쌍: {sum(len(v) for v in auto_synonym_dict.values()):,}개\")\n",
            "\n",
            "# 예시 출력\n",
            "print(f\"\\n📝 발견된 동의어 예시 (상위 20개):\")\n",
            "for i, (token, synonyms) in enumerate(list(auto_synonym_dict.items())[:20], 1):\n",
            "    if synonyms:\n",
            "        print(f\"  {i:2d}. {token:15s} → {', '.join(synonyms[:3])}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### 5.4. Synonym-Aware IDF 생성"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def create_synonym_aware_idf(original_idf, tokenizer, synonym_dict, method='max'):\n",
            "    \"\"\"\n",
            "    동의어 정보를 반영한 IDF 생성\n",
            "    \n",
            "    Args:\n",
            "        original_idf: 원본 IDF 딕셔너리\n",
            "        tokenizer: Tokenizer\n",
            "        synonym_dict: 동의어 사전 {token: [synonyms]}\n",
            "        method: 'max', 'mean' 중 선택\n",
            "    \n",
            "    Returns:\n",
            "        enhanced_idf: 강화된 IDF 딕셔너리\n",
            "    \"\"\"\n",
            "    enhanced_idf = original_idf.copy()\n",
            "    boost_count = 0\n",
            "    \n",
            "    for canonical, synonyms in synonym_dict.items():\n",
            "        all_tokens = [canonical] + synonyms\n",
            "        \n",
            "        # 각 토큰의 IDF 값 수집\n",
            "        idf_values = []\n",
            "        for token in all_tokens:\n",
            "            if token in original_idf:\n",
            "                idf_values.append(original_idf[token])\n",
            "        \n",
            "        if not idf_values:\n",
            "            continue\n",
            "        \n",
            "        # IDF 값 통합\n",
            "        if method == 'max':\n",
            "            shared_idf = max(idf_values)\n",
            "        else:  # mean\n",
            "            shared_idf = np.mean(idf_values)\n",
            "        \n",
            "        # 모든 동의어 토큰에 적용\n",
            "        for token in all_tokens:\n",
            "            if token in enhanced_idf:\n",
            "                enhanced_idf[token] = shared_idf\n",
            "                boost_count += 1\n",
            "    \n",
            "    print(f\"\\n✓ Synonym-Aware IDF 생성 완료\")\n",
            "    print(f\"  {boost_count:,}개 토큰에 동의어 정보 반영\")\n",
            "    \n",
            "    return enhanced_idf\n",
            "\n",
            "\n",
            "# Synonym-Aware IDF 생성\n",
            "print(\"\\n\" + \"=\"*60)\n",
            "print(\"🔄 동의어 정보를 반영한 IDF 생성 중...\")\n",
            "print(\"=\"*60)\n",
            "\n",
            "idf_token_dict_enhanced = create_synonym_aware_idf(\n",
            "    idf_token_dict_boosted,\n",
            "    tokenizer,\n",
            "    auto_synonym_dict,\n",
            "    method='max'\n",
            ")\n",
            "\n",
            "# IDF 변화 예시\n",
            "print(\"\\n📊 IDF 변화 예시:\")\n",
            "sample_tokens = list(auto_synonym_dict.keys())[:5]\n",
            "for token in sample_tokens:\n",
            "    if token in idf_token_dict_boosted and token in idf_token_dict_enhanced:\n",
            "        original = idf_token_dict_boosted[token]\n",
            "        enhanced = idf_token_dict_enhanced[token]\n",
            "        change = \"↑\" if enhanced > original else \"→\"\n",
            "        print(f\"  {token:15s}: {original:.4f} {change} {enhanced:.4f}\")\n",
            "\n",
            "# Enhanced IDF를 기본 IDF로 사용\n",
            "idf_token_dict_boosted = idf_token_dict_enhanced.copy()\n",
            "print(\"\\n✓ Enhanced IDF를 기본 IDF로 설정 완료\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### 5.5. 동의어 기반 학습 데이터 확장"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def expand_data_with_synonyms(qd_pairs, documents, synonym_dict, \n",
            "                              tokenizer, expansion_ratio=0.2):\n",
            "    \"\"\"\n",
            "    동의어를 활용하여 학습 데이터 확장\n",
            "    \n",
            "    Args:\n",
            "        qd_pairs: 원본 query-document pairs\n",
            "        documents: 문서 리스트\n",
            "        synonym_dict: 동의어 사전\n",
            "        tokenizer: Tokenizer\n",
            "        expansion_ratio: 확장 비율 (0.2 = 20% 추가)\n",
            "    \n",
            "    Returns:\n",
            "        expanded_pairs: 확장된 pairs\n",
            "    \"\"\"\n",
            "    print(f\"\\n🔄 동의어 기반 데이터 확장 중... (expansion_ratio={expansion_ratio})\")\n",
            "    \n",
            "    expanded_pairs = list(qd_pairs)  # 원본 복사\n",
            "    expansion_count = int(len(qd_pairs) * expansion_ratio)\n",
            "    \n",
            "    added = 0\n",
            "    attempts = 0\n",
            "    max_attempts = expansion_count * 10\n",
            "    \n",
            "    while added < expansion_count and attempts < max_attempts:\n",
            "        attempts += 1\n",
            "        \n",
            "        # 랜덤 pair 선택\n",
            "        query, doc, relevance = qd_pairs[np.random.randint(len(qd_pairs))]\n",
            "        \n",
            "        # 쿼리 토큰화\n",
            "        query_tokens = tokenizer.tokenize(query)\n",
            "        \n",
            "        # 동의어로 대체 가능한 토큰 찾기\n",
            "        replaceable = [(i, token) for i, token in enumerate(query_tokens) \n",
            "                      if token in synonym_dict and synonym_dict[token]]\n",
            "        \n",
            "        if not replaceable:\n",
            "            continue\n",
            "        \n",
            "        # 랜덤하게 하나 선택하여 동의어로 대체\n",
            "        idx, token = replaceable[np.random.randint(len(replaceable))]\n",
            "        synonym = np.random.choice(synonym_dict[token])\n",
            "        \n",
            "        # 새 쿼리 생성\n",
            "        new_query_tokens = query_tokens.copy()\n",
            "        new_query_tokens[idx] = synonym\n",
            "        new_query = tokenizer.convert_tokens_to_string(new_query_tokens)\n",
            "        \n",
            "        # 중복 체크\n",
            "        if new_query != query and new_query.strip():\n",
            "            expanded_pairs.append((new_query, doc, relevance))\n",
            "            added += 1\n",
            "    \n",
            "    print(f\"✓ 데이터 확장 완료!\")\n",
            "    print(f\"  원본: {len(qd_pairs):,} pairs\")\n",
            "    print(f\"  확장: {len(expanded_pairs):,} pairs (+{added:,})\")\n",
            "    print(f\"  증가율: {(len(expanded_pairs) / len(qd_pairs) - 1) * 100:.1f}%\")\n",
            "    \n",
            "    return expanded_pairs\n",
            "\n",
            "\n",
            "# 학습 데이터 확장\n",
            "print(\"\\n\" + \"=\"*60)\n",
            "print(\"📈 동의어 기반 학습 데이터 확장\")\n",
            "print(\"=\"*60)\n",
            "\n",
            "korean_data['qd_pairs_expanded'] = expand_data_with_synonyms(\n",
            "    korean_data['qd_pairs'],\n",
            "    korean_data['documents'],\n",
            "    auto_synonym_dict,\n",
            "    tokenizer,\n",
            "    expansion_ratio=0.15  # 15% 확장\n",
            ")\n",
            "\n",
            "# 확장 예시 출력\n",
            "print(\"\\n📝 확장된 쿼리 예시:\")\n",
            "original_count = len(korean_data['qd_pairs'])\n",
            "for i, (query, doc, rel) in enumerate(korean_data['qd_pairs_expanded'][original_count:original_count+5]):\n",
            "    print(f\"  {i+1}. {query[:60]}...\")\n",
            "\n",
            "# 확장된 데이터를 기본으로 사용\n",
            "korean_data['qd_pairs'] = korean_data['qd_pairs_expanded']\n",
            "print(f\"\\n✓ 확장된 데이터를 학습에 사용\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5.6. 동의어 정보 요약\n",
            "\n",
            "자동 발견된 동의어 정보를 요약합니다."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"\\n\" + \"=\"*60)\n",
            "print(\"📊 동의어 발견 및 데이터 확장 요약\")\n",
            "print(\"=\"*60)\n",
            "\n",
            "print(f\"\\n1️⃣ 동의어 발견 결과:\")\n",
            "print(f\"  - 분석 대상 토큰: {len(important_tokens):,}개\")\n",
            "print(f\"  - 발견된 동의어 그룹: {len(auto_synonym_dict):,}개\")\n",
            "print(f\"  - 총 동의어 쌍: {sum(len(v) for v in auto_synonym_dict.values()):,}개\")\n",
            "print(f\"  - 평균 동의어 수: {np.mean([len(v) for v in auto_synonym_dict.values()]):.2f}개/그룹\")\n",
            "\n",
            "print(f\"\\n2️⃣ IDF 강화 결과:\")\n",
            "changes = 0\n",
            "for token in auto_synonym_dict.keys():\n",
            "    if token in idf_token_dict:\n",
            "        changes += 1\n",
            "print(f\"  - IDF 업데이트된 토큰: {changes:,}개\")\n",
            "\n",
            "print(f\"\\n3️⃣ 데이터 확장 결과:\")\n",
            "print(f\"  - 원본 pairs: {len(korean_data['qd_pairs_expanded']) - len(korean_data['qd_pairs']):,}개\")\n",
            "print(f\"  - 최종 pairs: {len(korean_data['qd_pairs']):,}개\")\n",
            "\n",
            "print(f\"\\n✅ 동의어 기반 데이터 확장 완료!\")\n",
            "print(f\"   학습 데이터가 더 풍부해졌습니다.\")\n",
            "print(\"=\"*60)"
        ]
    }
]

# 삽입할 위치 찾기 (## 5. OpenSearch 문서 인코더 모델 정의 앞)
insert_position = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown' and 'source' in cell:
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if '## 5. OpenSearch 문서 인코더 모델 정의' in source:
            insert_position = i
            print(f"삽입 위치 찾음: 셀 {i} 앞")
            break

if insert_position is None:
    print("❌ 삽입 위치를 찾을 수 없습니다. 마지막에 추가합니다.")
    insert_position = len(notebook['cells'])

# 새 셀들 삽입
for offset, new_cell in enumerate(new_cells):
    notebook['cells'].insert(insert_position + offset, new_cell)

print(f"\n✓ {len(new_cells)}개 셀 추가 완료 (위치: {insert_position})")

# 기존 섹션 번호 업데이트 (5 → 6, 6 → 7, etc.)
print("\n섹션 번호 업데이트 중...")
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown' and 'source' in cell:
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

        # ## 5. 이후의 섹션들을 +1씩 증가
        if i > insert_position + len(new_cells) - 1:
            for old_num in range(12, 4, -1):  # 12부터 5까지 역순으로
                old_header = f"## {old_num}."
                new_header = f"## {old_num + 1}."
                if old_header in source:
                    new_source = source.replace(old_header, new_header)
                    if isinstance(cell['source'], list):
                        cell['source'] = [line + '\n' for line in new_source.split('\n')[:-1]]
                        cell['source'].append(new_source.split('\n')[-1])
                    else:
                        cell['source'] = new_source
                    print(f"  셀 {i}: {old_header} → {new_header}")
                    break

# 노트북 저장
with open('korean_neural_sparse_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("\n✓ 노트북 파일 저장 완료!")
print(f"  총 셀 개수: {len(notebook['cells'])}")
