"""
Wikipedia Loader - Simple interface for loading Korean Wikipedia data.

This module provides a simple function to load Korean Wikipedia data,
with automatic downloading and caching.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_korean_wikipedia(
    max_documents: int = 100000,
    min_length: int = 100,
    cache_dir: str = "dataset/wikipedia_dumps",
    force_download: bool = False,
    use_latest: bool = True
) -> List[str]:
    """
    Load Korean Wikipedia documents.

    This function automatically:
    1. Downloads the dump if not cached
    2. Parses the dump if not cached
    3. Returns clean text documents

    Args:
        max_documents: Maximum number of documents to load
        min_length: Minimum text length for documents
        cache_dir: Directory for cache files
        force_download: Force re-download even if cached
        use_latest: Use latest dump (True) or HuggingFace dataset (False)

    Returns:
        List of document texts

    Example:
        >>> from src.wikipedia_loader import load_korean_wikipedia
        >>> docs = load_korean_wikipedia(max_documents=1000)
        >>> print(f"Loaded {len(docs)} documents")
    """
    if use_latest:
        return _load_from_dump(
            max_documents=max_documents,
            min_length=min_length,
            cache_dir=cache_dir,
            force_download=force_download
        )
    else:
        return _load_from_huggingface(max_documents=max_documents)


def _load_from_dump(
    max_documents: int,
    min_length: int,
    cache_dir: str,
    force_download: bool
) -> List[str]:
    """Load from Wikimedia dump (latest data)."""
    from src.wikipedia_downloader import WikipediaDownloader

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Check for cached JSON
    json_cache = cache_path / f"wikipedia_ko_{max_documents}.json"

    if json_cache.exists() and not force_download:
        print(f"✓ Loading from cache: {json_cache}")
        with open(json_cache, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
        print(f"✓ Loaded {len(cached_data):,} documents from cache")
        return [doc['text'] for doc in cached_data]

    # Download and parse
    print("📥 Downloading and parsing Korean Wikipedia dump...")
    print(f"   This may take 20-60 minutes for the first time")
    print(f"   (subsequent runs will use cached data)")

    downloader = WikipediaDownloader(dump_dir=str(cache_path))

    # Download dump
    print("\n1️⃣ Downloading dump...")
    try:
        dump_path = downloader.download_dump(dump_date='latest')
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\n💡 Falling back to HuggingFace dataset...")
        return _load_from_huggingface(max_documents)

    # Parse dump
    print("\n2️⃣ Parsing dump...")
    documents = []

    try:
        for i, doc in enumerate(downloader.parse_dump(
            dump_path=dump_path,
            max_documents=max_documents,
            min_length=min_length
        )):
            documents.append(doc)

            if (i + 1) % 5000 == 0:
                print(f"   Processed {i+1:,} documents...")

    except Exception as e:
        print(f"⚠️  Parsing stopped: {e}")
        if len(documents) == 0:
            print("💡 Falling back to HuggingFace dataset...")
            return _load_from_huggingface(max_documents)

    # Save to cache
    print(f"\n3️⃣ Saving to cache...")
    with open(json_cache, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"✅ Cached {len(documents):,} documents to {json_cache}")

    return [doc['text'] for doc in documents]


def _load_from_huggingface(max_documents: int) -> List[str]:
    """Load from HuggingFace datasets (2023 data, faster)."""
    from datasets import load_dataset

    print("📥 Loading from HuggingFace datasets (2023-11-01)...")

    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.ko",
            split=f"train[:{max_documents}]"
        )

        documents = []
        for item in dataset:
            text = item['text']
            if len(text) > 100:
                documents.append(text[:2000])

        print(f"✅ Loaded {len(documents):,} documents from HuggingFace")
        return documents

    except Exception as e:
        print(f"❌ HuggingFace loading failed: {e}")
        raise


def get_wikipedia_info() -> Dict[str, Any]:
    """
    Get information about available Wikipedia data sources.

    Returns:
        Dictionary with information about data sources
    """
    return {
        "sources": {
            "latest_dump": {
                "date": "2025-11-01 (latest)",
                "method": "load_korean_wikipedia(use_latest=True)",
                "pros": "Most recent data",
                "cons": "Slower first load (20-60 min download + parsing)",
                "caching": "Yes, subsequent loads are instant"
            },
            "huggingface": {
                "date": "2023-11-01",
                "method": "load_korean_wikipedia(use_latest=False)",
                "pros": "Fast, reliable",
                "cons": "2 years old data",
                "caching": "Built-in by datasets library"
            }
        },
        "recommendation": "Use latest_dump for production, huggingface for quick testing"
    }


# Convenience function for notebook usage
def quick_load(num_docs: int = 10000, latest: bool = False) -> List[str]:
    """
    Quick load function for notebooks.

    Args:
        num_docs: Number of documents to load
        latest: Use latest dump (slow first time) or HuggingFace (fast)

    Returns:
        List of document texts

    Example in notebook:
        >>> from src.wikipedia_loader import quick_load
        >>> docs = quick_load(10000, latest=False)  # Fast, 2023 data
        >>> # or
        >>> docs = quick_load(10000, latest=True)   # Latest, slower first time
    """
    return load_korean_wikipedia(
        max_documents=num_docs,
        use_latest=latest
    )


if __name__ == "__main__":
    # Test function
    print("Testing Wikipedia loader...")

    # Test HuggingFace (fast)
    print("\n1. Testing HuggingFace loader:")
    docs_hf = load_korean_wikipedia(max_documents=10, use_latest=False)
    print(f"   Loaded {len(docs_hf)} documents")
    print(f"   Sample: {docs_hf[0][:100]}...")

    # Show info
    print("\n2. Available sources:")
    import json
    info = get_wikipedia_info()
    print(json.dumps(info, indent=2, ensure_ascii=False))
