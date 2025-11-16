#!/usr/bin/env python3
"""
Qwen3-30B-A3B-Instruct-2507-FP8 테스트

FP8 Instruct 버전 모델을 테스트합니다.
Thinking 버전과 달리 일반 instruction-following 모델입니다.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_qwen30b_instruct_fp8():
    """Qwen3-30B Instruct FP8 모델 테스트"""
    print("="*70)
    print("🧪 Testing Qwen3-30B-A3B-Instruct-2507-FP8")
    print("="*70)
    print("Model: Instruction-following variant (not thinking)")
    print("="*70)

    from src.llm_loader import check_gpu_memory

    # Check GPU
    stats = check_gpu_memory()
    if not stats.get('available'):
        print("❌ No GPU available")
        return False

    total_vram = stats['devices'][0]['total_gb']
    print(f"\n⚠️  Model size check:")
    print(f"   Expected VRAM: ~30GB (30B params, FP8)")
    print(f"   Available VRAM: {total_vram:.2f}GB")

    print("\n" + "="*70)
    print("📥 Loading Qwen3-30B-A3B-Instruct-2507-FP8")
    print("="*70)
    print("Model: Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    print("Size: 30B parameters")
    print("Quantization: FP8")
    print("Expected VRAM: ~30GB")
    print("Expected load time: ~5-10 minutes")
    print("="*70)

    import time
    from src.llm_loader import load_qwen3_awq

    print("\n⏳ Loading model...")
    print("📝 Note: This will likely fail with Triton PTX error due to sm_121a")
    start_time = time.time()

    try:
        model, tokenizer = load_qwen3_awq(
            model_name="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
            device_map="auto",
        )

        load_time = time.time() - start_time
        print(f"\n✅ Model loaded in {load_time:.2f}s ({load_time/60:.1f} minutes)")

    except Exception as e:
        load_time = time.time() - start_time
        print(f"\n❌ Model loading failed after {load_time:.2f}s")
        print(f"Error: {e}")

        if "ptxas" in str(e) or "sm_121a" in str(e):
            print("\n" + "="*70)
            print("🔍 Diagnosis: Expected Triton PTX Error")
            print("="*70)
            print("This is the same issue as Qwen3-30B-A3B-Thinking-2507-FP8:")
            print("- GPU: NVIDIA GB10 (sm_121a)")
            print("- Triton: Does not support sm_121a")
            print("- PTXAS: Cannot compile for compute capability 12.1")
            print()
            print("❌ Conclusion: FP8 models cannot run on this GPU")
            print("✅ Solution: Use BF16 models instead")
            print("="*70)

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

        # Summary
        print("\n" + "="*70)
        print("✅ Qwen3-30B-Instruct-FP8 Test Summary")
        print("="*70)
        print(f"Model: Qwen3-30B-A3B-Instruct-2507-FP8")
        print(f"Load time: {load_time:.2f}s ({load_time/60:.1f} min)")
        print(f"Generation time: {gen_time:.2f}s")
        print(f"GPU Memory: {allocated:.2f} GB")
        print("="*70)

        print("\n🎉 SUCCESS! FP8 model is working!")
        print("   This is unexpected but great news!")

        return True

    except Exception as e:
        gen_time = time.time() - start_time
        print(f"\n❌ Generation failed after {gen_time:.2f}s")
        print(f"Error: {e}")

        if "ptxas" in str(e) or "sm_121a" in str(e):
            print("\n" + "="*70)
            print("🔍 Diagnosis: Triton PTX Compilation Error")
            print("="*70)
            print("Issue: GPU compute capability 12.1 not supported")
            print("GPU: NVIDIA GB10 (sm_121a)")
            print()
            print("This confirms that FP8 models cannot run on this GPU.")
            print("="*70)

        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print("Qwen3-30B-A3B-Instruct-2507-FP8 Test")
        print("="*70)
        print("Testing Instruct variant (not Thinking)")
        print("Expected to fail with same Triton PTX error")
        print("="*70)

        success = test_qwen30b_instruct_fp8()

        if success:
            print("\n🎉 Unexpected success!")
            print("   FP8 model is working on this GPU")
            print("   Consider using this for production")
        else:
            print("\n❌ Test failed as expected")
            print("   Recommendation: Use BF16 model")
            print("   - Qwen2.5-14B-Instruct (BF16) is proven to work")
            print("   - Or try Qwen2.5-32B-Instruct (BF16)")

    except KeyboardInterrupt:
        print("\n\n❌ Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
