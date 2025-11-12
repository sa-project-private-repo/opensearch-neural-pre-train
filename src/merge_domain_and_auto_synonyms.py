#!/usr/bin/env python3
"""
도메인 동의어와 자동 발견 동의어를 결합
"""

import json
import sys

# Read notebook
with open('korean_neural_sparse_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find section 7.4 (수집 데이터 기반 동의어 자동 발견)
# 그 다음 코드 셀 뒤에 새로운 섹션 삽입
target_idx = None
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if '### 7.4. 수집 데이터 기반 동의어 자동 발견' in source:
            # Find the next code cell
            for j in range(i+1, len(cells)):
                if cells[j]['cell_type'] == 'code':
                    target_idx = j + 1
                    print(f"Will insert after cell {j}")
                    break
            break

if target_idx is None:
    print("Error: Could not find section 7.4")
    sys.exit(1)

# Create new section: 도메인 동의어 + 자동 발견 동의어 결합
new_cells = []

# Section header
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 7.4.1. 도메인 동의어 + 자동 발견 동의어 결합\n",
        "\n",
        "AI 도메인 전문 용어와 자동 발견된 동의어를 결합하여\n",
        "더 포괄적인 동의어 사전을 구성합니다."
    ]
})

# Merge synonyms code
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"=\"*60)\n",
        "print(\"🔗 도메인 동의어 + 자동 발견 동의어 결합\")\n",
        "print(\"=\"*60)\n",
        "print()\n",
        "\n",
        "# 1. 도메인 동의어 통계\n",
        "print(f\"📚 도메인 동의어 (AI 용어집):\")\n",
        "print(f\"  항목 수: {len(domain_synonym_dict):,}개\")\n",
        "print()\n",
        "\n",
        "# 2. 자동 발견 동의어 통계\n",
        "print(f\"🔍 자동 발견 동의어 (코퍼스 기반):\")\n",
        "print(f\"  항목 수: {len(auto_synonym_dict):,}개\")\n",
        "print()\n",
        "\n",
        "# 3. 결합 전략: 도메인 우선, 자동 발견 보완\n",
        "merged_synonym_dict = {}\n",
        "\n",
        "# 먼저 도메인 동의어 추가 (신뢰도 높음)\n",
        "for term, synonyms in domain_synonym_dict.items():\n",
        "    merged_synonym_dict[term] = list(set(synonyms))  # 중복 제거\n",
        "\n",
        "# 자동 발견 동의어 추가 (도메인 동의어와 중복되지 않는 것만)\n",
        "added_from_auto = 0\n",
        "for term, synonyms in auto_synonym_dict.items():\n",
        "    term_lower = term.lower()\n",
        "    \n",
        "    if term_lower in merged_synonym_dict:\n",
        "        # 기존 항목에 새로운 동의어 추가\n",
        "        existing = set(merged_synonym_dict[term_lower])\n",
        "        new_synonyms = [s.lower() for s in synonyms if s.lower() not in existing]\n",
        "        if new_synonyms:\n",
        "            merged_synonym_dict[term_lower].extend(new_synonyms)\n",
        "            added_from_auto += len(new_synonyms)\n",
        "    else:\n",
        "        # 새로운 항목 추가\n",
        "        merged_synonym_dict[term_lower] = [s.lower() for s in synonyms]\n",
        "        added_from_auto += len(synonyms)\n",
        "\n",
        "print(f\"✅ 결합 완료:\")\n",
        "print(f\"  총 항목 수: {len(merged_synonym_dict):,}개\")\n",
        "print(f\"  도메인 동의어 기여: {len(domain_synonym_dict):,}개 항목\")\n",
        "print(f\"  자동 발견 기여: {added_from_auto:,}개 동의어 추가\")\n",
        "print()\n",
        "\n",
        "# 4. 샘플 확인\n",
        "print(\"📝 결합 동의어 샘플:\")\n",
        "sample_terms = ['검색', '인공지능', 'llm', 'chatgpt', '임베딩', 'rag']\n",
        "for term in sample_terms:\n",
        "    if term in merged_synonym_dict:\n",
        "        syns = merged_synonym_dict[term][:5]  # 상위 5개\n",
        "        print(f\"  {term:15s} → {', '.join(syns)}\")\n"
    ]
})

# Insert new cells
cells = cells[:target_idx] + new_cells + cells[target_idx:]

# Save notebook
nb['cells'] = cells
with open('korean_neural_sparse_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✓ 도메인 동의어 결합 섹션 추가 완료!")
print(f"  • 추가된 셀: {len(new_cells)}개")
print(f"  • 총 셀 개수: {len(cells)}개")
