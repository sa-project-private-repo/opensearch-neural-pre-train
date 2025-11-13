"""
LLM Model Loader for ARM aarch64 + Python 3.12

This module provides loaders for Qwen3 and gpt-oss-20b models
optimized for ARM architecture with CUDA support.

Supported models:
- Qwen3-30B-A3B-Thinking-2507-AWQ (4-bit, ~16GB) - Reasoning optimized
- Qwen2.5-14B-Instruct-AWQ (4-bit, ~4GB)
- Qwen2.5-7B-Instruct-AWQ (4-bit, ~2GB)
- gpt-oss-20b (GGUF Q4, ~5GB) - requires llama-cpp-python

Requirements:
- Python 3.12
- autoawq==0.2.7
- transformers==4.46.3
- accelerate==1.1.1
- llama-cpp-python==0.3.4 (optional, for gpt-oss-20b)
"""

from typing import Tuple, Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def check_gpu_memory() -> Dict[str, Any]:
    """
    Check available GPU memory and return statistics.

    Returns:
        Dictionary with GPU memory information

    Example:
        >>> stats = check_gpu_memory()
        >>> print(f"Available: {stats['free_gb']:.2f} GB")
    """
    if not torch.cuda.is_available():
        print("⚠️  No CUDA GPU detected")
        return {
            'available': False,
            'device_count': 0,
        }

    device_count = torch.cuda.device_count()
    stats = {
        'available': True,
        'device_count': device_count,
        'devices': []
    }

    print("="*70)
    print("🖥️  GPU Memory Status")
    print("="*70)

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = props.total_memory / 1024**3
        free = total - allocated

        device_info = {
            'id': i,
            'name': props.name,
            'total_gb': total,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': free,
        }
        stats['devices'].append(device_info)

        print(f"\nGPU {i}: {props.name}")
        print(f"  Total:     {total:>8.2f} GB")
        print(f"  Allocated: {allocated:>8.2f} GB")
        print(f"  Free:      {free:>8.2f} GB")

    print("="*70)
    return stats


def load_qwen3_awq(
    model_name: str = "QuantTrio/Qwen3-30B-A3B-Thinking-2507-AWQ",
    device_map: str = "auto",
    low_cpu_mem_usage: bool = True,
) -> Tuple[Any, Any]:
    """
    Load Qwen3 model with AWQ 4-bit quantization (ARM compatible).

    Args:
        model_name: Hugging Face model name
            - "QuantTrio/Qwen3-30B-A3B-Thinking-2507-AWQ" (~16GB VRAM) - Reasoning optimized ⭐
            - "cpatonn/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit" (~16GB VRAM) - Alternative
            - "Qwen/Qwen2.5-14B-Instruct-AWQ" (~4GB VRAM)
            - "Qwen/Qwen2.5-7B-Instruct-AWQ" (~2GB VRAM)
        device_map: Device placement strategy ("auto", "cuda", "cpu")
        low_cpu_mem_usage: Use low CPU memory during loading

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> model, tokenizer = load_qwen3_awq()
        >>> response = model.generate(**tokenizer("Hello", return_tensors="pt"))
    """
    print("\n" + "="*70)
    print(f"📥 Loading Qwen3 Model (AWQ 4-bit)")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Device map: {device_map}")

    # Load tokenizer
    print("\n1️⃣ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer):,})")

    # Load model with AWQ quantization (Transformers native support)
    print("\n2️⃣ Loading model with AWQ quantization...")
    print("   Using Transformers native AWQ support (ARM compatible)")
    print("   This may take a few minutes...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=True,
        # Transformers 4.57+ has native AWQ support, no AutoAWQ needed
    )

    print(f"✓ Model loaded")

    # Print model info
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"\n📊 GPU Memory after loading: {allocated:.2f} GB")

    print("\n" + "="*70)
    print("✅ Qwen3 Model Ready!")
    print("="*70)

    return model, tokenizer


def load_gpt_oss_gguf(
    model_path: str,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,
    n_batch: int = 512,
) -> Any:
    """
    Load gpt-oss-20b model in GGUF format (requires llama-cpp-python).

    ⚠️  Requires llama-cpp-python installation:
        CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir

    Args:
        model_path: Path to .gguf file
        n_ctx: Context window size
        n_gpu_layers: Number of layers to offload to GPU (-1 = all)
        n_batch: Batch size for prompt processing

    Returns:
        Llama model instance

    Example:
        >>> llm = load_gpt_oss_gguf("gpt-oss-20b-Q4_0.gguf")
        >>> output = llm("Hello, how are you?", max_tokens=50)
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python not installed. Install with:\n"
            "CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python --no-cache-dir"
        )

    print("\n" + "="*70)
    print(f"📥 Loading gpt-oss-20b (GGUF)")
    print("="*70)
    print(f"Model path: {model_path}")
    print(f"Context size: {n_ctx}")
    print(f"GPU layers: {n_gpu_layers} ({'all' if n_gpu_layers == -1 else n_gpu_layers})")

    print("\nLoading model...")
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        verbose=False,
    )

    print("\n" + "="*70)
    print("✅ gpt-oss-20b Model Ready!")
    print("="*70)

    return llm


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """
    Generate text using Qwen3 model.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = deterministic)
        top_p: Nucleus sampling threshold
        do_sample: Whether to use sampling (False = greedy)

    Returns:
        Generated text (without prompt)

    Example:
        >>> text = generate_text(model, tokenizer, "Translate to English: 안녕하세요")
        >>> print(text)
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode and remove prompt
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = full_text[len(prompt):].strip()

    return generated_text


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> list[str]:
    """
    Generate text for multiple prompts in batch.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompts: List of input prompts
        max_new_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        do_sample: Whether to use sampling

    Returns:
        List of generated texts

    Example:
        >>> prompts = ["Hello", "How are you"]
        >>> responses = generate_batch(model, tokenizer, prompts)
    """
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode all outputs
    generated_texts = []
    for i, output in enumerate(outputs):
        full_text = tokenizer.decode(output, skip_special_tokens=True)
        # Remove prompt (approximate)
        generated_text = full_text[len(prompts[i]):].strip()
        generated_texts.append(generated_text)

    return generated_texts


# Prompt templates
QWEN3_CHAT_TEMPLATE = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

DOC_TO_QUERY_PROMPT = """다음 문서를 읽고 사용자가 이 문서를 찾기 위해 검색할 만한 쿼리를 {num_queries}개 생성하세요.
각 쿼리는 짧고 구체적이어야 합니다 (5-15단어).

문서: {document}

검색 쿼리 ({num_queries}개, 각 줄에 하나씩):
"""

SYNONYM_VERIFICATION_PROMPT = """다음 두 단어가 같은 의미를 가지거나 동의어 관계인지 판단하세요.

단어 1: {word1}
단어 2: {word2}

같은 의미이거나 동의어라면 "예", 아니면 "아니오"로 답하고 간단한 이유를 설명하세요.

답변:
"""


if __name__ == "__main__":
    # Test GPU memory check
    check_gpu_memory()

    print("\n" + "="*70)
    print("LLM Loader Module Test Complete")
    print("="*70)
    print("\nTo use:")
    print("  from src.llm_loader import load_qwen3_awq, generate_text")
    print("  model, tokenizer = load_qwen3_awq()")
    print("  text = generate_text(model, tokenizer, 'Hello!')")
