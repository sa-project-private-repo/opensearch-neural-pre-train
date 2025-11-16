#!/usr/bin/env python3
"""
Test script to verify Korean query generation and lowercase synonyms.

This script tests:
1. Korean query generation from documents
2. Lowercase synonym handling
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_loader import load_ollama_model
from src.synthetic_data_generator import generate_queries_from_document
from src.cross_lingual_synonyms import get_default_korean_english_pairs


def test_korean_query_generation():
    """Test that queries are generated in Korean."""
    print("=" * 70)
    print("Test 1: Korean Query Generation")
    print("=" * 70)

    # Load Ollama model
    print("\n📥 Loading Ollama model...")
    try:
        llm_model, llm_tokenizer = load_ollama_model(
            model_name="qwen3:30b-a3b-thinking-2507-fp16",
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Test document (Korean)
    test_doc = """
    OpenSearch는 Apache 2.0 라이선스로 제공되는 오픈 소스 검색 및 분석 엔진입니다.
    Elasticsearch를 포크하여 만들어졌으며, 강력한 검색 기능과 실시간 분석을 제공합니다.
    분산 아키텍처로 대규모 데이터를 효율적으로 처리할 수 있습니다.
    """

    print(f"\n📄 Test document: {test_doc[:100]}...")
    print(f"\n🔄 Generating queries...")

    try:
        queries = generate_queries_from_document(
            document=test_doc,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            num_queries=3,
            verbose=True,
        )

        print(f"\n✅ Generated {len(queries)} queries:")
        for i, query in enumerate(queries, 1):
            print(f"   {i}. {query}")

        # Check if queries contain Korean characters
        korean_count = 0
        english_count = 0

        for query in queries:
            has_korean = any('\uac00' <= c <= '\ud7a3' for c in query)
            has_english = any(c.isascii() and c.isalpha() for c in query)

            if has_korean:
                korean_count += 1
            if has_english and not has_korean:
                english_count += 1

        print(f"\n📊 Language analysis:")
        print(f"   Korean queries: {korean_count}/{len(queries)}")
        print(f"   English-only queries: {english_count}/{len(queries)}")

        if korean_count >= len(queries) * 0.8:  # At least 80% Korean
            print("\n✅ Test PASSED: Queries are in Korean")
            return True
        else:
            print("\n❌ Test FAILED: Not enough Korean queries")
            return False

    except Exception as e:
        print(f"\n❌ Error generating queries: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lowercase_synonyms():
    """Test that all synonyms are lowercase."""
    print("\n" + "=" * 70)
    print("Test 2: Lowercase Synonyms")
    print("=" * 70)

    synonym_dict = get_default_korean_english_pairs()

    print(f"\n📊 Loaded {len(synonym_dict)} synonym entries")

    uppercase_found = []

    for korean, english_list in synonym_dict.items():
        for eng in english_list:
            # Check if it's ASCII (English)
            if eng.isascii() and eng.isalpha():
                # Check if it has uppercase
                if eng != eng.lower():
                    uppercase_found.append((korean, eng))

    print(f"\n📋 Sample synonyms:")
    for i, (kor, eng_list) in enumerate(list(synonym_dict.items())[:5], 1):
        eng_str = ", ".join(eng_list)
        print(f"   {i}. {kor} → {eng_str}")

    if uppercase_found:
        print(f"\n❌ Test FAILED: Found {len(uppercase_found)} uppercase entries:")
        for kor, eng in uppercase_found[:10]:
            print(f"   {kor} → {eng}")
        return False
    else:
        print("\n✅ Test PASSED: All English synonyms are lowercase")
        return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("🧪 Testing Korean Generation and Lowercase Synonyms")
    print("=" * 70)

    results = []

    # Test 1: Lowercase synonyms (quick)
    results.append(("Lowercase Synonyms", test_lowercase_synonyms()))

    # Test 2: Korean query generation (requires LLM)
    results.append(("Korean Query Generation", test_korean_query_generation()))

    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n✅ All tests PASSED!")
        return 0
    else:
        print("\n❌ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
