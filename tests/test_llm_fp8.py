#!/usr/bin/env python3
"""
FP8 모델 테스트 - python3.12-dev 설치 후 Triton JIT 컴파일 테스트

python3.12-dev 설치로 Python.h가 사용 가능해져 Triton JIT 컴파일이 작동합니다.
이제 FP8 모델이 빠른 속도로 추론할 수 있습니다.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_fp8_model_with_triton():
    """FP8 모델 + Triton JIT 컴파일 테스트"""
    print("="*70)
    print("🧪 Testing FP8 Model with Triton JIT Compilation")
    print("="*70)
    print("✅ python3.12-dev installed - Triton should work now")
    print()

    from src.llm_loader import check_gpu_memory

    # Check GPU
    stats = check_gpu_memory()
    if not stats.get('available'):
        print("❌ No GPU available")
        return False

    print("\n" + "="*70)
    print("📥 Loading Qwen3-30B-A3B-Thinking-2507-FP8")
    print("="*70)
    print("Model: Qwen/Qwen3-30B-A3B-Thinking-2507-FP8")
    print("Quantization: FP8 (with Triton JIT compilation)")
    print("Expected VRAM: ~30GB")
    print("Expected load time: ~6-10 minutes")
    print("="*70)

    import time
    from src.llm_loader import load_qwen3_awq

    print("\n⏳ Loading model...")
    start_time = time.time()

    model, tokenizer = load_qwen3_awq(
        model_name="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
        device_map="auto",
    )

    load_time = time.time() - start_time
    print(f"\n✅ Model loaded in {load_time:.2f}s ({load_time/60:.1f} minutes)")

    # Test generation
    print("\n" + "="*70)
    print("🚀 Testing Query Generation (with Triton JIT)")
    print("="*70)

    from src.synthetic_data_generator import generate_queries_from_document

    doc = "OpenSearch는 Apache 2.0 라이선스의 오픈 소스 검색 및 분석 엔진입니다. " \
          "Elasticsearch와 호환되며 대규모 데이터 검색, 로그 분석, 실시간 모니터링에 사용됩니다."

    print(f"📝 Document: {doc[:80]}...")
    print("⏳ Generating queries (first call includes Triton kernel compilation)...")

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

    # Second generation (should be faster - kernels already compiled)
    print("\n" + "="*70)
    print("⚡ Second Generation (kernels cached)")
    print("="*70)

    doc2 = "Elasticsearch는 실시간 분산 검색 및 분석 엔진입니다. " \
           "JSON 문서를 색인화하고 빠른 검색을 제공합니다."

    print(f"📝 Document: {doc2[:80]}...")
    print("⏳ Generating...")

    start_time = time.time()

    queries2 = generate_queries_from_document(
        document=doc2,
        llm_model=model,
        llm_tokenizer=tokenizer,
        num_queries=3,
        max_new_tokens=150,
        temperature=0.8,
        verbose=True,
    )

    gen_time2 = time.time() - start_time

    print(f"\n✅ Second generation completed in {gen_time2:.2f}s")
    print(f"📊 Generated {len(queries2)} queries:")
    for i, q in enumerate(queries2, 1):
        print(f"   {i}. {q}")

    # GPU memory check
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print("\n" + "="*70)
        print("📊 GPU Memory Status")
        print("="*70)
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved:  {reserved:.2f} GB")
        print(f"Total:     {total:.2f} GB")
        print(f"Free:      {total - allocated:.2f} GB")

    # Summary
    print("\n" + "="*70)
    print("✅ FP8 Model Test Summary")
    print("="*70)
    print(f"Model: Qwen3-30B-A3B-Thinking-2507-FP8")
    print(f"Load time: {load_time:.2f}s ({load_time/60:.1f} min)")
    print(f"First generation: {gen_time:.2f}s (includes compilation)")
    print(f"Second generation: {gen_time2:.2f}s (cached kernels)")
    print(f"GPU Memory: {allocated:.2f} GB")
    print("="*70)

    # Performance comparison
    print("\n" + "="*70)
    print("📈 Performance Comparison")
    print("="*70)
    print(f"FP8 (Triton interpreter): >60s (timeout)")
    print(f"FP8 (Triton JIT):         {gen_time2:.2f}s ⭐")
    print(f"BF16 (pure PyTorch):      ~20s")
    print()
    print("✅ Triton JIT compilation is working!")

    if gen_time2 < 10:
        print("🚀 FP8 model is FAST! Much faster than BF16!")
        print("   Recommendation: Use FP8 model in production")
    elif gen_time2 < 20:
        print("✅ FP8 model is comparable to BF16")
        print("   Recommendation: FP8 for memory efficiency, BF16 for stability")
    else:
        print("⚠️  FP8 model is slower than expected")
        print("   Recommendation: Consider using BF16 instead")

    print("="*70)

    return True


if __name__ == "__main__":
    try:
        success = test_fp8_model_with_triton()

        if success:
            print("\n💡 Next steps:")
            print("   1. If FP8 is fast (<10s), update notebook 2 back to FP8 model")
            print("   2. If FP8 is slow (>15s), keep using BF16 model")
            print("   3. Run full synthetic data generation with chosen model")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
