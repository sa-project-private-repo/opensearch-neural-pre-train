#!/usr/bin/env python3
"""
섹션 7.5와 7.6에서 merged_synonym_dict를 사용하도록 업데이트
그리고 모델 embedding resize 코드 추가
"""

import json
import re

# Read notebook
with open('korean_neural_sparse_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

print("Updating synonym dictionary usage...")

# Find and update cells that use auto_synonym_dict
updated_count = 0
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Replace auto_synonym_dict with merged_synonym_dict in sections 7.5+
        if 'auto_synonym_dict' in source and ('create_synonym_aware_idf' in source or 'expand_data_with_synonyms' in source):
            # Update the variable name
            new_source = source.replace('auto_synonym_dict', 'merged_synonym_dict')
            cell['source'] = [new_source]
            updated_count += 1
            print(f"  Updated cell {i}: auto_synonym_dict → merged_synonym_dict")

print(f"✓ Updated {updated_count} cells")

# Find section 5 (model definition) to add embedding resize
section_5_found = False
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if '## 5. OpenSearch 문서 인코더 모델 정의' in source:
            section_5_found = True
            print(f"\nFound section 5 at cell {i}")

            # Find the code cell that creates doc_encoder
            for j in range(i+1, min(i+5, len(cells))):
                if cells[j]['cell_type'] == 'code':
                    code_source = ''.join(cells[j]['source'])
                    if 'doc_encoder = OpenSearchDocEncoder' in code_source and 'doc_encoder.to(device)' in code_source:
                        # Add embedding resize after model creation
                        lines = code_source.split('\n')

                        # Find the line with doc_encoder.to(device)
                        insert_idx = None
                        for k, line in enumerate(lines):
                            if 'doc_encoder.to(device)' in line:
                                insert_idx = k + 1
                                break

                        if insert_idx:
                            # Insert resize code
                            new_lines = lines[:insert_idx] + [
                                "",
                                "# Tokenizer에 special tokens 추가로 인해 embedding resize 필요",
                                "if len(tokenizer) > doc_encoder.vocab_size:",
                                "    print(f\"Resizing model embeddings: {doc_encoder.vocab_size} → {len(tokenizer)}\")",
                                "    doc_encoder.bert.resize_token_embeddings(len(tokenizer))",
                                "    doc_encoder.vocab_size = len(tokenizer)",
                                "    print(f\"✓ Model embeddings resized\")",
                            ] + lines[insert_idx:]

                            cells[j]['source'] = ['\n'.join(new_lines)]
                            print(f"  Added embedding resize code to cell {j}")
                        break
            break

if not section_5_found:
    print("Warning: Could not find section 5 to add embedding resize")

# Save notebook
nb['cells'] = cells
with open('korean_neural_sparse_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✓ Notebook updated successfully!")
print(f"  • Synonym dict usage updated: {updated_count} cells")
print(f"  • Model embedding resize: added")
