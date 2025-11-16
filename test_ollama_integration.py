#!/usr/bin/env python3
"""
Quick test script for Ollama integration.
"""

import sys
from src.llm_loader import load_ollama_model, generate_text

def main():
    print("="*70)
    print("Testing Ollama Integration")
    print("="*70)

    # Load model
    print("\n1. Loading Ollama model...")
    try:
        model, tokenizer = load_ollama_model(
            model_name="qwen3:30b-a3b-thinking-2507-fp16"
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1

    # Test simple generation
    print("\n2. Testing text generation...")
    test_prompt = "안녕하세요. 이것은 테스트입니다. 간단히 응답해주세요."

    try:
        response = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=test_prompt,
            max_new_tokens=50,
            temperature=0.7,
        )
        print(f"✅ Generation successful!")
        print(f"\nPrompt: {test_prompt}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test query generation
    print("\n3. Testing query generation from document...")
    from src.synthetic_data_generator import generate_queries_from_document
    from src.llm_loader import OllamaModel

    test_doc = "OpenSearch는 Apache Lucene 기반의 오픈소스 검색 및 분석 엔진입니다."

    print(f"   Model type before call: {type(model)}")
    print(f"   Is OllamaModel: {isinstance(model, OllamaModel)}")

    try:
        queries = generate_queries_from_document(
            document=test_doc,
            llm_model=model,
            llm_tokenizer=tokenizer,
            num_queries=3,
            verbose=True,
        )
        print(f"\n✅ Query generation successful!")
        print(f"\nDocument: {test_doc}")
        print(f"Generated queries:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
    except Exception as e:
        print(f"❌ Query generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
