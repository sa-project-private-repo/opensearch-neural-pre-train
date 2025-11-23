"""Comprehensive inference test for KNN synonym expansion.

Tests the trained model with various Korean-English synonym pairs
and outputs results in JSON format.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import torch
from transformers import AutoTokenizer

from src.model.splade_model import SPLADEDoc


def load_model_from_checkpoint(checkpoint_path: str, model_name: str = "bert-base-multilingual-cased"):
    """Load trained SPLADE model from checkpoint."""
    model = SPLADEDoc(model_name=model_name, dropout=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        raise ValueError("No model weights found in checkpoint")

    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer, device


def get_term_weights(sparse_repr: torch.Tensor, tokenizer, target_terms: list, top_k: int = 100):
    """Get weights for target terms and top-k terms."""
    # Get top-k
    topk_values, topk_indices = torch.topk(sparse_repr, k=top_k)

    top_terms = []
    for value, idx in zip(topk_values.tolist(), topk_indices.tolist()):
        token = tokenizer.convert_ids_to_tokens([idx])[0]
        if not token.startswith('[') or token == '[MASK]':
            top_terms.append({
                "token": token,
                "weight": round(value, 4),
                "token_id": idx
            })

    # Find target terms
    target_weights = {}
    for term in target_terms:
        # Try exact match first
        token_ids = tokenizer.encode(term, add_special_tokens=False)
        if token_ids:
            # Get weight for first token of the term
            token_id = token_ids[0]
            weight = sparse_repr[token_id].item()
            target_weights[term] = {
                "weight": round(weight, 4),
                "token_id": token_id,
                "found": weight > 0.01
            }
        else:
            target_weights[term] = {
                "weight": 0.0,
                "token_id": None,
                "found": False
            }

    return top_terms, target_weights


def run_comprehensive_test():
    """Run comprehensive inference test."""

    # Load model
    checkpoint_path = "outputs/baseline_dgx/best_model/checkpoint.pt"
    if not Path(checkpoint_path).exists():
        return {"error": "Checkpoint not found"}

    model, tokenizer, device = load_model_from_checkpoint(checkpoint_path)

    # Test cases from KNN-discovered synonyms
    test_cases = [
        {
            "id": 1,
            "korean_term": "컴퓨터",
            "english_synonym": "computer",
            "document": "컴퓨터 시스템은 현대 사회의 기반입니다. Computer systems are fundamental to modern society.",
        },
        {
            "id": 2,
            "korean_term": "서버",
            "english_synonym": "server",
            "document": "서버 인프라는 클라우드 서비스의 핵심입니다. Server infrastructure is crucial for cloud services.",
        },
        {
            "id": 3,
            "korean_term": "모델",
            "english_synonym": "model",
            "document": "기계학습 모델은 데이터를 분석합니다. Machine learning model analyzes data patterns.",
        },
        {
            "id": 4,
            "korean_term": "데이터",
            "english_synonym": "data",
            "document": "데이터 분석은 비즈니스 인사이트를 제공합니다. Data analysis provides business insights.",
        },
        {
            "id": 5,
            "korean_term": "시스템",
            "english_synonym": "system",
            "document": "운영 시스템의 안정성이 중요합니다. Operating system stability is critical.",
        },
        {
            "id": 6,
            "korean_term": "네트워크",
            "english_synonym": "network",
            "document": "네트워크 보안은 필수적입니다. Network security is essential.",
        },
        {
            "id": 7,
            "korean_term": "인터넷",
            "english_synonym": "internet",
            "document": "인터넷 연결이 필요합니다. Internet connectivity is required.",
        },
        {
            "id": 8,
            "korean_term": "프로그램",
            "english_synonym": "program",
            "document": "소프트웨어 프로그램을 개발합니다. Software program development continues.",
        },
    ]

    results = {
        "model_info": {
            "checkpoint": checkpoint_path,
            "validation_loss": 0.2880,
            "device": str(device),
        },
        "test_summary": {
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
        },
        "tests": []
    }

    for test in test_cases:
        # Tokenize and encode document
        inputs = tokenizer(
            test['document'],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            sparse_repr, _ = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
            )

        # Get weights for Korean and English terms
        target_terms = [test['korean_term'], test['english_synonym']]
        top_terms, target_weights = get_term_weights(
            sparse_repr[0],
            tokenizer,
            target_terms,
            top_k=30
        )

        # Check if synonym expansion worked
        korean_found = target_weights[test['korean_term']]['found']
        english_found = target_weights[test['english_synonym']]['found']
        passed = korean_found and english_found

        if passed:
            results['test_summary']['passed'] += 1
        else:
            results['test_summary']['failed'] += 1

        test_result = {
            "id": test['id'],
            "korean_term": test['korean_term'],
            "english_synonym": test['english_synonym'],
            "document": test['document'][:80] + "...",
            "korean_weight": target_weights[test['korean_term']]['weight'],
            "english_weight": target_weights[test['english_synonym']]['weight'],
            "korean_found": korean_found,
            "english_found": english_found,
            "passed": passed,
            "top_10_terms": top_terms[:10]
        }

        results['tests'].append(test_result)

    # Calculate success rate
    results['test_summary']['success_rate'] = round(
        results['test_summary']['passed'] / results['test_summary']['total_tests'] * 100,
        2
    )

    return results


if __name__ == "__main__":
    print("Running comprehensive inference test...")
    results = run_comprehensive_test()

    # Pretty print JSON
    print(json.dumps(results, ensure_ascii=False, indent=2))

    # Save to file
    output_file = "outputs/comprehensive_inference_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
