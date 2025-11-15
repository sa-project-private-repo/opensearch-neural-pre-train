#!/usr/bin/env python3
"""
LLM 생성 테스트 - 합성 데이터 생성이 멈추는 문제 디버깅

이 스크립트는 다음을 테스트합니다:
1. LLM 모델 로드 가능 여부
2. 간단한 텍스트 생성 가능 여부
3. 한글 문서에서 쿼리 생성 가능 여부
4. 각 단계별 소요 시간 측정
"""

import os
import sys
import time
from pathlib import Path

# Disable Triton to avoid compilation errors (ARM aarch64)
# MUST be set before importing torch or transformers
os.environ["TRITON_INTERPRET"] = "1"  # Use interpreter mode
os.environ["DISABLE_TRITON"] = "1"     # Completely disable
print("🔧 Triton disabled (TRITON_INTERPRET=1, DISABLE_TRITON=1)")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_gpu_availability():
    """GPU 사용 가능 여부 확인"""
    print("="*70)
    print("1️⃣ Testing GPU Availability")
    print("="*70)

    import torch

    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")

        allocated = torch.cuda.memory_allocated(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU Memory: {allocated:.2f} / {total:.2f} GB")
        return True
    else:
        print("❌ CUDA not available")
        return False


def test_model_loading():
    """LLM 모델 로드 테스트"""
    print("\n" + "="*70)
    print("2️⃣ Testing Model Loading")
    print("="*70)

    try:
        from src.llm_loader import load_qwen3_awq

        print("⏳ Loading Qwen3 model (this may take a few minutes)...")
        start_time = time.time()

        model, tokenizer = load_qwen3_awq()

        load_time = time.time() - start_time
        print(f"\n✅ Model loaded successfully in {load_time:.2f}s")

        # Check model device
        if hasattr(model, 'device'):
            print(f"   Model device: {model.device}")

        return model, tokenizer

    except Exception as e:
        print(f"\n❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_simple_generation(model, tokenizer):
    """간단한 텍스트 생성 테스트"""
    print("\n" + "="*70)
    print("3️⃣ Testing Simple Text Generation")
    print("="*70)

    if model is None or tokenizer is None:
        print("⚠️  Skipped: Model not loaded")
        return False

    try:
        from src.llm_loader import generate_text

        prompt = "1부터 5까지 숫자를 나열하세요:"
        print(f"📝 Prompt: {prompt}")
        print("⏳ Generating...")

        start_time = time.time()

        # Set a timeout mechanism
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Generation took too long (>60s)")

        # Set 60 second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)

        try:
            generated = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=50,
                temperature=0.7,
            )
            signal.alarm(0)  # Cancel timeout

            gen_time = time.time() - start_time

            print(f"\n✅ Generation completed in {gen_time:.2f}s")
            print(f"📄 Output: {generated}")
            return True

        except TimeoutError as e:
            signal.alarm(0)
            print(f"\n❌ Generation timeout: {e}")
            return False

    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_korean_query_generation(model, tokenizer):
    """한글 문서에서 쿼리 생성 테스트"""
    print("\n" + "="*70)
    print("4️⃣ Testing Korean Query Generation")
    print("="*70)

    if model is None or tokenizer is None:
        print("⚠️  Skipped: Model not loaded")
        return False

    try:
        from src.synthetic_data_generator import generate_queries_from_document

        # Sample Korean document
        doc = "OpenSearch는 Apache 2.0 라이선스의 오픈 소스 검색 및 분석 엔진입니다. " \
              "Elasticsearch와 호환되며 대규모 데이터 검색, 로그 분석, 실시간 모니터링에 사용됩니다."

        print(f"📝 Document: {doc[:80]}...")
        print("⏳ Generating queries (max 60s timeout)...")

        start_time = time.time()

        # Set timeout
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Query generation took too long (>60s)")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)

        try:
            queries = generate_queries_from_document(
                document=doc,
                llm_model=model,
                llm_tokenizer=tokenizer,
                num_queries=3,
                max_new_tokens=150,
                temperature=0.8,
                verbose=True,  # Enable verbose logging
            )
            signal.alarm(0)  # Cancel timeout

            gen_time = time.time() - start_time

            print(f"\n✅ Query generation completed in {gen_time:.2f}s")
            print(f"📊 Generated {len(queries)} queries:")
            for i, q in enumerate(queries, 1):
                print(f"   {i}. {q}")

            return True

        except TimeoutError as e:
            signal.alarm(0)
            print(f"\n❌ Query generation timeout: {e}")
            print("   🔍 This suggests the LLM is taking too long to respond")
            print("   💡 Possible causes:")
            print("      - Model too large for available GPU memory")
            print("      - Inference is extremely slow")
            print("      - Model not properly loaded to GPU")
            return False

    except Exception as e:
        print(f"\n❌ Query generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_memory_after_generation(model):
    """생성 후 GPU 메모리 사용량 확인"""
    print("\n" + "="*70)
    print("5️⃣ GPU Memory After Generation")
    print("="*70)

    import torch

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available")
        return

    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"GPU Memory Status:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Total:     {total:.2f} GB")
    print(f"  Free:      {total - allocated:.2f} GB")

    if allocated > total * 0.9:
        print("\n⚠️  WARNING: GPU memory usage is very high (>90%)")
        print("   This may cause slow inference or OOM errors")


def main():
    """메인 테스트 실행"""
    print("\n" + "="*70)
    print("🧪 LLM Generation Debugging Test")
    print("="*70)
    print("This test will help identify why LLM generation is stuck")
    print("="*70)

    results = {}

    # Test 1: GPU
    results['gpu'] = test_gpu_availability()

    # Test 2: Model loading
    model, tokenizer = test_model_loading()
    results['model_load'] = (model is not None)

    if not results['model_load']:
        print("\n" + "="*70)
        print("❌ Cannot proceed: Model loading failed")
        print("="*70)
        return

    # Test 3: Simple generation
    results['simple_gen'] = test_simple_generation(model, tokenizer)

    # Test 4: Korean query generation
    results['korean_gen'] = test_korean_query_generation(model, tokenizer)

    # Test 5: GPU memory
    test_gpu_memory_after_generation(model)

    # Summary
    print("\n" + "="*70)
    print("📊 Test Summary")
    print("="*70)
    print(f"GPU Available:         {'✅' if results.get('gpu') else '❌'}")
    print(f"Model Loading:         {'✅' if results.get('model_load') else '❌'}")
    print(f"Simple Generation:     {'✅' if results.get('simple_gen') else '❌'}")
    print(f"Korean Query Gen:      {'✅' if results.get('korean_gen') else '❌'}")

    print("\n" + "="*70)
    print("🔍 Diagnosis")
    print("="*70)

    if not results.get('gpu'):
        print("❌ No GPU available - LLM inference will be very slow or fail")
        print("   💡 Solution: Ensure CUDA is properly installed")

    elif not results.get('model_load'):
        print("❌ Model failed to load")
        print("   💡 Solution: Check model name and available memory")

    elif not results.get('simple_gen'):
        print("❌ Simple generation failed or timed out")
        print("   💡 Possible causes:")
        print("      1. Model is too large for GPU memory")
        print("      2. Inference is extremely slow")
        print("      3. Model not properly configured for generation")
        print("   💡 Solution: Try a smaller model or check GPU memory")

    elif not results.get('korean_gen'):
        print("❌ Korean query generation failed or timed out")
        print("   💡 Possible causes:")
        print("      1. Prompt is too long")
        print("      2. Model struggles with Korean text")
        print("      3. max_new_tokens too high")
        print("   💡 Solution:")
        print("      - Reduce max_new_tokens from 150 to 50")
        print("      - Simplify the prompt")
        print("      - Try English prompts first")

    else:
        print("✅ All tests passed!")
        print("   The LLM generation is working correctly in isolation.")
        print("   💡 If notebook is still stuck, possible causes:")
        print("      1. Jupyter kernel issue - restart kernel")
        print("      2. Model was loaded multiple times - check GPU memory")
        print("      3. Interference from other notebook cells")

    print("="*70)


if __name__ == "__main__":
    main()
