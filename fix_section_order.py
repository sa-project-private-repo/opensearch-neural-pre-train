#!/usr/bin/env python3
"""
Fix the section ordering in korean_neural_sparse_training.ipynb
Move synonym discovery section to after model definition and dataset preparation
"""

import json
import re

# Read notebook
with open('korean_neural_sparse_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find the synonym discovery cells (should be section 5)
synonym_section_start = None
synonym_section_end = None

for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if '## 5.' in source and '자동 동의어 발견' in source:
            synonym_section_start = i
            print(f"Found synonym section start at cell {i}")
            break

if synonym_section_start is None:
    print("Error: Could not find synonym discovery section")
    exit(1)

# Find the end of synonym discovery section (next ## heading)
for i in range(synonym_section_start + 1, len(cells)):
    if cells[i]['cell_type'] == 'markdown':
        source = ''.join(cells[i]['source'])
        if source.strip().startswith('## 6.'):
            synonym_section_end = i
            print(f"Found synonym section end at cell {i}")
            break

if synonym_section_end is None:
    print("Error: Could not find end of synonym discovery section")
    exit(1)

# Extract synonym discovery cells
synonym_cells = cells[synonym_section_start:synonym_section_end]
print(f"Synonym section has {len(synonym_cells)} cells")

# Find where to insert: after "학습 데이터셋 준비" section
# That should be section 7 now
insert_position = None

for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if '## 7.' in source and '학습 데이터셋 준비' in source:
            # Find the end of this section
            for j in range(i + 1, len(cells)):
                if cells[j]['cell_type'] == 'markdown':
                    next_source = ''.join(cells[j]['source'])
                    if next_source.strip().startswith('## 8.'):
                        insert_position = j
                        print(f"Will insert synonym section at cell {j}")
                        break
            break

if insert_position is None:
    print("Error: Could not find insertion position")
    exit(1)

# Remove synonym cells from original position
remaining_cells = cells[:synonym_section_start] + cells[synonym_section_end:]

# Adjust insert position due to removal
if insert_position > synonym_section_end:
    insert_position -= len(synonym_cells)

# Insert synonym cells at new position
new_cells = remaining_cells[:insert_position] + synonym_cells + remaining_cells[insert_position:]

# Update section numbers
# Old section 5 (synonym) should become section 8
# Old sections 6, 7 should become 5, 6
# Old section 8+ should become 9+

def update_section_number(source, old_num, new_num):
    """Update section number in markdown heading"""
    pattern = f'## {old_num}\\.'
    replacement = f'## {new_num}.'
    return re.sub(pattern, replacement, source)

# Renumber sections
section_mapping = {
    6: 5,  # OpenSearch 문서 인코더 모델 정의: 6 -> 5
    7: 6,  # 학습 데이터셋 준비: 7 -> 6
    5: 7,  # 토큰 임베딩 기반 동의어 (was 5, now should be 7)
    8: 8,  # 모델 학습 stays 8
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
}

print("\nRenumbering sections...")
for i, cell in enumerate(new_cells):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        # Check each possible section number
        for old_num, new_num in section_mapping.items():
            pattern = f'## {old_num}\\.'
            if re.search(pattern, source):
                new_source = re.sub(pattern, f'## {new_num}.', source)
                cell['source'] = [new_source]
                print(f"  Cell {i}: ## {old_num}. -> ## {new_num}.")
                break

# Update notebook
nb['cells'] = new_cells

# Save
with open('korean_neural_sparse_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✓ Notebook reordered successfully!")
print(f"  Total cells: {len(new_cells)}")
print(f"\nNew section order:")
print("  Section 5: OpenSearch 문서 인코더 모델 정의")
print("  Section 6: 학습 데이터셋 준비")
print("  Section 7: 토큰 임베딩 기반 동의어 자동 발견")
print("  Section 8: 모델 학습")
