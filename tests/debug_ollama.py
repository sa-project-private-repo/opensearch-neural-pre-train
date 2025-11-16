#!/usr/bin/env python3
"""Debug script to test Ollama chat API."""

import ollama

client = ollama.Client(host="http://localhost:11434")

# Test 1: generate API
print("="*70)
print("Test 1: generate API")
print("="*70)

response1 = client.generate(
    model="qwen3:30b-a3b-thinking-2507-fp16",
    prompt="한국어로 '검색 엔진'에 대한 쿼리 3개를 생성하세요.",
    options={"temperature": 0.3, "num_predict": 100}
)

print(f"Type: {type(response1)}")
print(f"Response: {response1}")
if hasattr(response1, 'response'):
    print(f"response field: {response1.response}")
if hasattr(response1, 'thinking'):
    print(f"thinking field: {response1.thinking}")

# Test 2: chat API
print("\n" + "="*70)
print("Test 2: chat API with system prompt")
print("="*70)

response2 = client.chat(
    model="qwen3:30b-a3b-thinking-2507-fp16",
    messages=[
        {'role': 'system', 'content': '당신은 한국어 검색 쿼리 생성 전문가입니다. 오직 한국어로만 답변하세요.'},
        {'role': 'user', 'content': '검색 엔진에 대한 쿼리 3개를 생성하세요.'}
    ],
    options={"temperature": 0.3, "num_predict": 100}
)

print(f"Type: {type(response2)}")
print(f"Response: {response2}")
if isinstance(response2, dict):
    print(f"Keys: {response2.keys()}")
    if 'message' in response2:
        print(f"Message: {response2['message']}")
        if isinstance(response2['message'], dict):
            print(f"Message content: {response2['message'].get('content', 'NO CONTENT')}")
