"""
LLM Model Loader for ARM aarch64 + Python 3.12

This module provides loaders for Qwen3 and gpt-oss-20b models
optimized for ARM architecture with CUDA support.

Supported models:
- Qwen3-30B-A3B-Thinking-2507-AWQ (4-bit, ~16GB) - Reasoning optimized
- Qwen2.5-14B-Instruct-AWQ (4-bit, ~4GB)
- Qwen2.5-7B-Instruct-AWQ (4-bit, ~2GB)
- gpt-oss-20b (GGUF Q4, ~5GB) - requires llama-cpp-python
- Ollama models (local server) - Zero setup, efficient

Requirements:
- Python 3.12
- autoawq==0.2.7
- transformers==4.46.3
- accelerate==1.1.1
- llama-cpp-python==0.3.4 (optional, for gpt-oss-20b)
- ollama==0.6.1 (optional, for ollama models)
"""

from typing import Tuple, Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# Ollama Model Classes (defined early for isinstance checks)
# ============================================================================

class OllamaModel:
    """
    Wrapper class for Ollama models to provide compatible interface.

    This class wraps an ollama.Client to provide the same interface
    as HuggingFace models used in this module.

    Attributes:
        client: Ollama client instance
        model_name: Name of the ollama model

    Example:
        >>> model = OllamaModel("qwen3:30b-a3b-thinking-2507-fp16")
        >>> response = model.generate("Hello, how are you?")
    """

    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        """
        Initialize Ollama model wrapper.

        Args:
            model_name: Name of the ollama model (e.g., "qwen3:30b")
            host: Ollama server URL
        """
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama not installed. Install with:\n"
                "pip install ollama"
            )

        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=host)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text using Ollama model.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate (mapped to num_predict)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            stream: Whether to stream the response
            system_prompt: Optional system prompt for better control

        Returns:
            Generated text
        """
        # Use chat API if system prompt provided (better for thinking models)
        if system_prompt:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                stream=stream,
                options={
                    "num_predict": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )

            if stream:
                text = ""
                for chunk in response:
                    text += chunk.get('message', {}).get('content', '')
                return text
            else:
                # Extract message (could be dict or object)
                message = response.get('message') if isinstance(response, dict) else getattr(response, 'message', None)
                if message:
                    # Try to get content and thinking from message
                    if isinstance(message, dict):
                        content = message.get('content', '')
                        thinking = message.get('thinking', '')
                    else:
                        content = getattr(message, 'content', '')
                        thinking = getattr(message, 'thinking', '')

                    # For thinking models, combine thinking + content
                    # (but content is usually what we want)
                    return content if content else thinking
                return ''
        else:
            # Use generate API
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=stream,
                options={
                    "num_predict": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )

            if stream:
                # For streaming, collect all chunks
                text = ""
                for chunk in response:
                    # Try both 'response' and 'thinking' fields
                    text += getattr(chunk, 'response', '') or getattr(chunk, 'thinking', '')
                return text
            else:
                # For thinking models, the output might be in 'thinking' field
                # Try 'response' first, fallback to 'thinking'
                result = getattr(response, 'response', '') or getattr(response, 'thinking', '')
                return result


class OllamaTokenizer:
    """
    Dummy tokenizer for Ollama models to maintain interface compatibility.

    Ollama handles tokenization internally, so this is just a placeholder
    to maintain compatibility with code expecting a tokenizer object.
    """

    def __init__(self):
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self.eos_token_id = 0

    def __len__(self):
        """Return arbitrary vocab size."""
        return 100000


# ============================================================================
# GPU Memory Functions
# ============================================================================

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
    model_name: str = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
    device_map: str = "auto",
    low_cpu_mem_usage: bool = True,
) -> Tuple[Any, Any]:
    """
    Load Qwen3 model with quantization (ARM aarch64 compatible).

    Args:
        model_name: Hugging Face model name
            - "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8" (~30GB VRAM) - FP8 quantization ⭐ ARM compatible
            - "Qwen/Qwen3-30B-A3B-Thinking-2507" (~60GB VRAM) - BF16, full precision
            - "Qwen/Qwen2.5-14B-Instruct" (~28GB VRAM) - Smaller alternative
        device_map: Device placement strategy ("auto", "cuda", "cpu")
        low_cpu_mem_usage: Use low CPU memory during loading

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> model, tokenizer = load_qwen3_awq()
        >>> response = model.generate(**tokenizer("Hello", return_tensors="pt"))

    Note:
        AWQ quantization requires autoawq package which doesn't support ARM aarch64.
        Using FP8 quantization instead, which is natively supported by Transformers.
        FP8 provides good compression (~30GB) while maintaining high quality.
    """
    # Detect model type from name
    is_fp8 = "FP8" in model_name
    is_bf16 = not is_fp8 and ("Instruct" in model_name or "Thinking" in model_name.replace("-FP8", ""))

    print("\n" + "="*70)
    if is_fp8:
        print(f"📥 Loading Qwen Model (FP8 Quantization)")
    elif is_bf16:
        print(f"📥 Loading Qwen Model (BF16 Precision)")
    else:
        print(f"📥 Loading Qwen Model")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Device map: {device_map}")

    if is_fp8:
        print(f"💡 Using FP8 quantization (ARM aarch64 compatible)")
    elif is_bf16:
        print(f"💡 Using BF16 precision (pure PyTorch, no Triton)")

    # Load tokenizer
    print("\n1️⃣ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer):,})")

    # Load model
    print("\n2️⃣ Loading model...")
    if is_fp8:
        print("   Using Transformers native FP8 support (no external deps needed)")
    elif is_bf16:
        print("   Using BF16 precision for stable inference")
    print("   This may take a few minutes...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if is_bf16 else None,  # Explicit BF16 for non-FP8 models
    )

    print(f"✓ Model loaded")

    # Print model info
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"\n📊 GPU Memory after loading: {allocated:.2f} GB")

    print("\n" + "="*70)
    print("✅ Model Ready!")
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
    tokenizer: Any = None,
    prompt: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Generate text using Qwen3 model or Ollama model.

    Args:
        model: Loaded model (HuggingFace or OllamaModel)
        tokenizer: Loaded tokenizer (or OllamaTokenizer, optional for Ollama)
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = deterministic)
        top_p: Nucleus sampling threshold
        do_sample: Whether to use sampling (False = greedy)
        system_prompt: System prompt for Ollama chat API (thinking models)

    Returns:
        Generated text (without prompt)

    Example:
        >>> text = generate_text(model, tokenizer, "Translate to English: 안녕하세요")
        >>> print(text)
    """
    # Check if this is an OllamaModel FIRST before using tokenizer
    if isinstance(model, OllamaModel):
        return model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
        )

    # HuggingFace model path - requires tokenizer
    if tokenizer is None:
        raise ValueError("tokenizer is required for HuggingFace models")

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


# ============================================================================
# Ollama Loader Functions
# ============================================================================

def load_ollama_model(
    model_name: str = "qwen3:30b-a3b-thinking-2507-fp16",
    host: str = "http://localhost:11434",
) -> Tuple[OllamaModel, OllamaTokenizer]:
    """
    Load an Ollama model running on local server.

    Args:
        model_name: Name of the ollama model
        host: Ollama server URL (default: http://localhost:11434)

    Returns:
        Tuple of (OllamaModel, OllamaTokenizer)

    Example:
        >>> model, tokenizer = load_ollama_model("qwen3:30b")
        >>> text = generate_text(model, tokenizer, "Hello!")

    Note:
        - Requires ollama to be running (`ollama serve`)
        - Model must be pulled first (`ollama pull <model>`)
        - Zero GPU setup needed, ollama handles everything
    """
    print("\n" + "="*70)
    print("📥 Loading Ollama Model")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Host: {host}")
    print("💡 Using Ollama (local server, zero setup)")

    # Check if ollama is available
    try:
        import ollama
    except ImportError:
        raise ImportError(
            "ollama not installed. Install with:\n"
            "pip install ollama"
        )

    # Try to connect and verify model exists
    print("\n1️⃣ Connecting to Ollama server...")
    try:
        client = ollama.Client(host=host)
        models_response = client.list()
        # models_response.models is a list of Model objects with .model attribute
        model_names = [m.model for m in models_response.models]

        if model_name not in model_names:
            print(f"⚠️  Warning: Model '{model_name}' not found in ollama")
            print(f"   Available models: {', '.join(model_names)}")
            print(f"\n   To pull the model, run:")
            print(f"   ollama pull {model_name}")
            raise ValueError(f"Model {model_name} not available")

        print(f"✓ Connected to Ollama server")
        print(f"✓ Model '{model_name}' is available")

    except Exception as e:
        print(f"❌ Failed to connect to Ollama server at {host}")
        print(f"   Error: {e}")
        print(f"\n   Make sure Ollama is running:")
        print(f"   ollama serve")
        raise

    # Create model wrapper
    print("\n2️⃣ Creating model wrapper...")
    model = OllamaModel(model_name=model_name, host=host)
    tokenizer = OllamaTokenizer()

    print(f"✓ Model wrapper created")

    print("\n" + "="*70)
    print("✅ Ollama Model Ready!")
    print("="*70)
    print("📊 Model uses Ollama server (no GPU memory tracked here)")
    print("    View Ollama logs for resource usage")
    print("="*70)

    return model, tokenizer


def generate_text_ollama(
    model: OllamaModel,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate text using Ollama model.

    Args:
        model: OllamaModel instance
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Returns:
        Generated text

    Example:
        >>> model, _ = load_ollama_model()
        >>> text = generate_text_ollama(model, "Hello!")
    """
    return model.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


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
    print("\nTo use Ollama:")
    print("  from src.llm_loader import load_ollama_model, generate_text")
    print("  model, tokenizer = load_ollama_model('qwen3:30b')")
    print("  text = generate_text(model, tokenizer, 'Hello!')")
