#!/usr/bin/env python3
"""
BF16 모델 테스트 - Triton 없이 작동하는 모델

FP8 모델은 Triton이 필수이지만, BF16 모델은 순수 PyTorch로 작동합니다.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_bf16_model():
    """BF16 모델로 쿼리 생성 테스트"""
    print("="*70)
    print("🧪 Testing BF16 Model (No Triton Required)")
    print("="*70)

    from src.llm_loader import check_gpu_memory

    # Check GPU
    stats = check_gpu_memory()
    if not stats.get('available'):
        print("❌ No GPU available")
        return False

    print("\n" + "="*70)
    print("📥 Loading Qwen2.5-14B-Instruct (BF16)")
    print("="*70)
    print("Model size: ~28GB VRAM")
    print("Quantization: BF16 (no FP8, no Triton)")
    print("Expected load time: ~3-5 minutes")
    print("="*70)

    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_name = "Qwen/Qwen2.5-14B-Instruct"

    print("\n1️⃣ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer):,})")

    print("\n2️⃣ Loading model...")
    start_time = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # BF16 precision
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f}s")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"📊 GPU Memory: {allocated:.2f} GB")

    # Test generation
    print("\n" + "="*70)
    print("3️⃣ Testing Query Generation")
    print("="*70)

    from src.synthetic_data_generator import generate_queries_from_document

    doc = "OpenSearch는 Apache 2.0 라이선스의 오픈 소스 검색 및 분석 엔진입니다. " \
          "Elasticsearch와 호환되며 대규모 데이터 검색, 로그 분석, 실시간 모니터링에 사용됩니다."

    print(f"📝 Document: {doc[:80]}...")
    print("⏳ Generating queries...")

    start_time = time.time()

    queries = generate_queries_from_document(
        document=doc,
        llm_model=model,
        llm_tokenizer=tokenizer,
        num_queries=3,
        max_new_tokens=150,
        temperature=0.8,
        verbose=True,
    )

    gen_time = time.time() - start_time

    print(f"\n✅ Generation completed in {gen_time:.2f}s")
    print(f"📊 Generated {len(queries)} queries:")
    for i, q in enumerate(queries, 1):
        print(f"   {i}. {q}")

    print("\n" + "="*70)
    print("✅ BF16 Model Test Passed!")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Load time: {load_time:.2f}s")
    print(f"Query gen time: {gen_time:.2f}s")
    print(f"GPU Memory: {allocated:.2f} GB")
    print("="*70)

    return True


if __name__ == "__main__":
    try:
        success = test_bf16_model()
        if success:
            print("\n💡 Next steps:")
            print("   1. Update notebook 2 to use BF16 model:")
            print("      model, tokenizer = load_qwen3_awq('Qwen/Qwen2.5-14B-Instruct')")
            print("   2. No Triton environment variables needed")
            print("   3. Pure PyTorch - reliable and stable")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
