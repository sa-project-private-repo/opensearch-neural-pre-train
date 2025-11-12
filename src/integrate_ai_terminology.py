#!/usr/bin/env python3
"""
AI 도메인 용어집을 training notebook에 통합
"""

import json
import sys

# Read notebook
with open('korean_neural_sparse_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find section 5 (OpenSearch 문서 인코더 모델 정의)
section_5_idx = None
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if '## 5. OpenSearch 문서 인코더 모델 정의' in source:
            section_5_idx = i
            print(f"Found section 5 at cell {i}")
            break

if section_5_idx is None:
    print("Error: Could not find section 5")
    sys.exit(1)

# Create new section 4.5: AI 도메인 용어집 통합
new_cells = []

# Section header
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4.5. AI 도메인 특화 용어집 통합\n",
        "\n",
        "AI/ML/LLM 도메인에 특화된 용어집을 로드하고,\n",
        "기술 용어를 tokenizer special tokens로 추가하여 분절을 방지합니다."
    ]
})

# Load terminology module
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# AI 도메인 용어집 로드\n",
        "from ai_domain_terminology import (\n",
        "    AI_TERMINOLOGY,\n",
        "    TECHNICAL_SPECIAL_TOKENS,\n",
        "    AI_SYNONYMS\n",
        ")\n",
        "\n",
        "print(\"=\"*60)\n",
        "print(\"🤖 AI 도메인 용어집 로드\")\n",
        "print(\"=\"*60)\n",
        "print(f\"✓ 주요 용어 카테고리: {len(AI_TERMINOLOGY)}개\")\n",
        "print(f\"✓ Special tokens: {len(TECHNICAL_SPECIAL_TOKENS)}개\")\n",
        "print(f\"✓ 동의어 매핑: {len(AI_SYNONYMS)}개\")\n",
        "print()\n",
        "print(\"📝 샘플 용어:\")\n",
        "for i, (term, synonyms) in enumerate(list(AI_TERMINOLOGY.items())[:5]):\n",
        "    print(f\"  {term}: {', '.join(synonyms[:3])}\")\n"
    ]
})

# Add special tokens to tokenizer
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 4.5.1. Tokenizer에 기술 용어 추가\n",
        "\n",
        "ChatGPT, OpenSearch 등 기술 용어가 분절되는 것을 방지하기 위해\n",
        "special tokens로 추가합니다."
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"=\"*60)\n",
        "print(\"🔧 Tokenizer에 기술 용어 추가\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "# 현재 vocabulary 크기\n",
        "original_vocab_size = len(tokenizer)\n",
        "print(f\"Original vocab size: {original_vocab_size:,}\")\n",
        "\n",
        "# Special tokens 추가\n",
        "num_added = tokenizer.add_tokens(TECHNICAL_SPECIAL_TOKENS)\n",
        "new_vocab_size = len(tokenizer)\n",
        "\n",
        "print(f\"Added {num_added} new tokens\")\n",
        "print(f\"New vocab size: {new_vocab_size:,}\")\n",
        "print()\n",
        "\n",
        "# 추가된 토큰 샘플 확인\n",
        "print(\"✓ 추가된 기술 용어 샘플:\")\n",
        "for token in TECHNICAL_SPECIAL_TOKENS[:10]:\n",
        "    token_id = tokenizer.convert_tokens_to_ids(token)\n",
        "    print(f\"  {token:20s} -> ID: {token_id}\")\n",
        "\n",
        "print()\n",
        "print(\"🧪 분절 방지 테스트:\")\n",
        "test_texts = [\n",
        "    \"ChatGPT는 강력한 LLM입니다\",\n",
        "    \"OpenSearch 벡터검색 기능\",\n",
        "    \"RAG 파이프라인 구축\",\n",
        "]\n",
        "\n",
        "for text in test_texts:\n",
        "    tokens = tokenizer.tokenize(text)\n",
        "    print(f\"  '{text}'\")\n",
        "    print(f\"    → {tokens}\")\n"
    ]
})

# Merge domain synonyms with auto-discovered synonyms
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 4.5.2. 도메인 동의어 매핑 생성\n",
        "\n",
        "AI 도메인 용어집의 동의어를 활용하여 검색 성능을 향상시킵니다."
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"=\"*60)\n",
        "print(\"🔗 도메인 동의어 매핑\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "# AI_SYNONYMS를 일반 dictionary 형태로 변환\n",
        "domain_synonym_dict = {}\n",
        "\n",
        "for main_term, synonyms in AI_TERMINOLOGY.items():\n",
        "    # Main term을 소문자로\n",
        "    main_key = main_term.lower()\n",
        "    synonym_list = [s.lower() for s in synonyms]\n",
        "    \n",
        "    domain_synonym_dict[main_key] = synonym_list\n",
        "    \n",
        "    # 양방향 매핑: 각 synonym도 main term을 가리킴\n",
        "    for syn in synonym_list:\n",
        "        if syn not in domain_synonym_dict:\n",
        "            domain_synonym_dict[syn] = [main_key]\n",
        "        elif main_key not in domain_synonym_dict[syn]:\n",
        "            domain_synonym_dict[syn].append(main_key)\n",
        "\n",
        "print(f\"✓ 도메인 동의어 딕셔너리 생성 완료\")\n",
        "print(f\"  총 {len(domain_synonym_dict):,}개 항목\")\n",
        "print()\n",
        "print(\"📝 샘플 동의어 매핑:\")\n",
        "samples = [\n",
        "    \"검색\", \"인공지능\", \"llm\", \"chatgpt\", \"임베딩\",\n",
        "    \"rag\", \"프롬프트\", \"딥러닝\", \"머신러닝\"\n",
        "]\n",
        "for term in samples:\n",
        "    if term in domain_synonym_dict:\n",
        "        syns = domain_synonym_dict[term][:3]  # 상위 3개만\n",
        "        print(f\"  {term:15s} → {', '.join(syns)}\")\n"
    ]
})

# Summary section
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 4.5.3. 용어집 통합 요약"
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"=\"*60)\n",
        "print(\"✅ AI 도메인 용어집 통합 완료!\")\n",
        "print(\"=\"*60)\n",
        "print()\n",
        "print(\"📊 통합 결과:\")\n",
        "print(f\"  • Tokenizer vocabulary: {original_vocab_size:,} → {new_vocab_size:,} (+{num_added})\")\n",
        "print(f\"  • AI 도메인 용어: {len(AI_TERMINOLOGY):,}개 카테고리\")\n",
        "print(f\"  • 동의어 매핑: {len(domain_synonym_dict):,}개 항목\")\n",
        "print(f\"  • Special tokens: {len(TECHNICAL_SPECIAL_TOKENS)}개\")\n",
        "print()\n",
        "print(\"🎯 주요 개선 사항:\")\n",
        "print(\"  1. 기술 용어 분절 방지 (ChatGPT, OpenSearch, LLM 등)\")\n",
        "print(\"  2. AI 도메인 동의어 자동 매핑 (검색↔Search↔탐색)\")\n",
        "print(\"  3. 한국어-영어 용어 양방향 연결\")\n",
        "print()\n",
        "print(\"💡 다음 단계:\")\n",
        "print(\"  → 섹션 7에서 도메인 동의어와 자동 발견 동의어를 결합\")\n"
    ]
})

# Insert new cells before section 5
cells = cells[:section_5_idx] + new_cells + cells[section_5_idx:]

# Save notebook
nb['cells'] = cells
with open('korean_neural_sparse_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✓ AI 도메인 용어집이 notebook에 통합되었습니다!")
print(f"  • 새로운 섹션: 4.5. AI 도메인 특화 용어집 통합")
print(f"  • 추가된 셀: {len(new_cells)}개")
print(f"  • 총 셀 개수: {len(cells)}개")
