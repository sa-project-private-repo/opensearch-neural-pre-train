#!/usr/bin/env python3
"""
Qwen2.5-32B-Instruct (BF16) 테스트

더 큰 BF16 모델로 테스트합니다. FP8보다 안정적이며 1 GPU로 실행 가능합니다.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_qwen32b_bf16():
    """Qwen2.5-32B BF16 모델 테스트"""
    print("="*70)
    print("🧪 Testing Qwen2.5-32B-Instruct (BF16)")
    print("="*70)

    from src.llm_loader import check_gpu_memory

    # Check GPU
    stats = check_gpu_memory()
    if not stats.get('available'):
        print("❌ No GPU available")
        return False

    total_vram = stats['devices'][0]['total_gb']
    print(f"\n⚠️  Model size check:")
    print(f"   Expected VRAM: ~64GB (32B params, BF16)")
    print(f"   Available VRAM: {total_vram:.2f}GB")

    if total_vram < 60:
        print(f"\n⚠️  VRAM might be tight!")
        print(f"   This model requires about 64GB VRAM")

    print("\n" + "="*70)
    print("📥 Loading Qwen2.5-32B-Instruct (BF16)")
    print("="*70)
    print("Model: Qwen/Qwen2.5-32B-Instruct")
    print("Size: 32B parameters")
    print("Precision: BF16 (stable, no Triton)")
    print("Expected VRAM: ~64GB")
    print("Expected load time: ~10-15 minutes (first download)")
    print("="*70)

    import time
    from src.llm_loader import load_qwen3_awq

    print("\n⏳ Loading model...")
    start_time = time.time()

    try:
        model, tokenizer = load_qwen3_awq(
            model_name="Qwen/Qwen2.5-32B-Instruct",
            device_map="auto",
        )

        load_time = time.time() - start_time
        print(f"\n✅ Model loaded in {load_time:.2f}s ({load_time/60:.1f} minutes)")

    except Exception as e:
        load_time = time.time() - start_time
        print(f"\n❌ Model loading failed after {load_time:.2f}s")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check GPU memory after loading
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print("\n" + "="*70)
        print("📊 GPU Memory After Loading")
        print("="*70)
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved:  {reserved:.2f} GB")
        print(f"Total:     {total:.2f} GB")
        print(f"Free:      {total - allocated:.2f} GB")
        print(f"Usage:     {allocated/total*100:.1f}%")

    # Test generation
    print("\n" + "="*70)
    print("🚀 Testing Query Generation")
    print("="*70)

    from src.synthetic_data_generator import generate_queries_from_document

    doc = "OpenSearch는 Apache 2.0 라이선스의 오픈 소스 검색 및 분석 엔진입니다. " \
          "Elasticsearch와 호환되며 대규모 데이터 검색, 로그 분석, 실시간 모니터링에 사용됩니다."

    print(f"📝 Document: {doc[:80]}...")
    print("⏳ Generating queries...")

    start_time = time.time()

    try:
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

        # Second generation for speed comparison
        print("\n" + "="*70)
        print("⚡ Second Generation Test")
        print("="*70)

        doc2 = "Elasticsearch는 실시간 분산 검색 및 분석 엔진입니다. " \
               "JSON 문서를 색인화하고 빠른 검색을 제공합니다."

        print(f"📝 Document: {doc2[:80]}...")

        start_time2 = time.time()
        queries2 = generate_queries_from_document(
            document=doc2,
            llm_model=model,
            llm_tokenizer=tokenizer,
            num_queries=3,
            max_new_tokens=150,
            temperature=0.8,
            verbose=False,
        )
        gen_time2 = time.time() - start_time2

        print(f"✅ Completed in {gen_time2:.2f}s")
        print(f"📊 Generated {len(queries2)} queries:")
        for i, q in enumerate(queries2, 1):
            print(f"   {i}. {q}")

        # Summary
        print("\n" + "="*70)
        print("✅ Qwen2.5-32B Test Summary")
        print("="*70)
        print(f"Model: Qwen2.5-32B-Instruct (BF16)")
        print(f"Load time: {load_time/60:.1f} minutes")
        print(f"First generation: {gen_time:.2f}s")
        print(f"Second generation: {gen_time2:.2f}s")
        print(f"GPU Memory: {allocated:.2f} GB ({allocated/total*100:.1f}%)")
        print("="*70)

        # Comparison with 14B
        print("\n" + "="*70)
        print("📈 Comparison with 14B Model")
        print("="*70)
        print(f"Qwen2.5-14B-Instruct (BF16): 27.51 GB, ~20s/query")
        print(f"Qwen2.5-32B-Instruct (BF16): {allocated:.2f} GB, ~{gen_time2:.0f}s/query")
        print()

        if gen_time2 < 30:
            print("✅ 32B model is reasonably fast!")
            print("   Recommendation: Use 32B for better quality")
        elif gen_time2 < 60:
            print("⚠️  32B model is slower than 14B")
            print("   Recommendation: Use 14B for speed, 32B for quality")
        else:
            print("❌ 32B model is too slow")
            print("   Recommendation: Stick with 14B model")

        print("="*70)

        return True

    except Exception as e:
        gen_time = time.time() - start_time
        print(f"\n❌ Generation failed after {gen_time:.2f}s")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print("Qwen2.5-32B-Instruct (BF16) Test")
        print("="*70)
        print("This is a larger BF16 model that should work on 1 GPU")
        print("Expected to use ~64GB VRAM")
        print("="*70)

        success = test_qwen32b_bf16()

        if success:
            print("\n🎉 Test passed!")
            print("   32B model is working and ready to use")
            print("   Consider updating notebook 2 if quality is better")
        else:
            print("\n❌ Test failed")
            print("   Recommendation: Keep using 14B model")

    except KeyboardInterrupt:
        print("\n\n❌ Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
