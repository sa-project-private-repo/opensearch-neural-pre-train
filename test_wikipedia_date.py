"""Test Wikipedia dataset loading with different dates."""

from datasets import load_dataset

# Test with the new date
date = "20251101"
language = "ko"

print(f"Testing Wikipedia dataset: {date}.{language}")
print("=" * 60)

try:
    dataset = load_dataset(
        "wikipedia",
        f"{date}.{language}",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    print("✓ Dataset loaded successfully!")
    print(f"Dataset: {dataset}")

    # Try to get first article
    print("\nFetching first article...")
    first_article = next(iter(dataset))
    print(f"✓ First article title: {first_article['title']}")
    print(f"✓ Text length: {len(first_article['text'])} characters")

except Exception as e:
    print(f"✗ Error: {type(e).__name__}")
    print(f"  {str(e)}")

    # Try alternative date format
    print(f"\nTrying alternative approach...")
    print("Available Wikipedia configurations might be limited.")
    print("Consider using wikimedia dumps directly.")
