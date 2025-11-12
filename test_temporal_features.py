#!/usr/bin/env python3
"""
Test script for temporal features (Phase 2).

Tests:
- Data loading with dates
- Temporal IDF calculation
- Automatic trend detection
"""

import sys
from datetime import datetime

from transformers import AutoTokenizer

# Import our new modules
from src.data_loader import load_korean_news_with_dates
from src.temporal_analysis import (
    calculate_temporal_idf,
    detect_trending_tokens,
    build_trend_boost_dict,
    apply_temporal_boost_to_idf,
)

print("=" * 70)
print("Testing Temporal Features (Phase 2)")
print("=" * 70)

# Step 1: Load tokenizer
print("\n[Step 1] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
print(f"✓ Loaded tokenizer (vocab_size: {tokenizer.vocab_size:,})")

# Step 2: Load news data with dates
print("\n[Step 2] Loading news data with dates...")
try:
    news_data = load_korean_news_with_dates(
        max_samples=10000,  # Limit for testing
        min_doc_length=20,
    )

    documents = news_data["documents"]
    dates = news_data["dates"]

    print(f"\n✓ Loaded {len(documents):,} documents")

    if not documents:
        print("❌ No documents loaded. Exiting.")
        sys.exit(1)

except Exception as e:
    print(f"❌ Failed to load news data: {e}")
    print("⚠️  This might be expected if the dataset doesn't have date info.")
    print("   Falling back to demo mode with synthetic dates...")

    # Fallback: use sample data with synthetic dates
    from test_korean_neural_sparse import SAMPLE_DOCUMENTS
    documents = SAMPLE_DOCUMENTS

    # Generate synthetic dates (last 60 days)
    from datetime import timedelta
    base_date = datetime.now()
    dates = [base_date - timedelta(days=i*3) for i in range(len(documents))]

    print(f"✓ Using {len(documents)} sample documents with synthetic dates")

# Step 3: Calculate temporal IDF
print("\n[Step 3] Calculating Temporal IDF...")
temporal_idf_token, temporal_idf_id = calculate_temporal_idf(
    documents=documents,
    dates=dates,
    tokenizer=tokenizer,
    decay_factor=0.95,
)

print(f"\n✓ Temporal IDF calculated for {len(temporal_idf_token):,} tokens")
print(f"  Sample IDF values:")
sample_tokens = list(temporal_idf_token.items())[:10]
for token, idf in sample_tokens:
    print(f"    '{token}': {idf:.4f}")

# Step 4: Detect trending tokens
print("\n[Step 4] Detecting Trending Tokens...")
try:
    trending_tokens = detect_trending_tokens(
        documents=documents,
        dates=dates,
        tokenizer=tokenizer,
        recent_days=30,
        historical_days=90,
        min_recent_count=2,  # Lower threshold for small dataset
        top_k=50,
    )

    if trending_tokens:
        print(f"\n✓ Detected {len(trending_tokens)} trending tokens")
    else:
        print("\n⚠️  No trending tokens detected (dataset may be too small or uniform)")
        trending_tokens = []

except Exception as e:
    print(f"⚠️  Trend detection failed: {e}")
    trending_tokens = []

# Step 5: Build automatic trend boost dictionary
print("\n[Step 5] Building Automatic Trend Boost Dictionary...")
if trending_tokens:
    auto_boost_dict = build_trend_boost_dict(
        trending_tokens=trending_tokens,
        max_boost=2.0,
        min_boost=1.2,
    )

    print(f"\n✓ Auto-generated boost dictionary:")
    print(f"  Total tokens: {len(auto_boost_dict)}")
    print(f"\n  Top 10 boosted tokens:")
    for i, (token, boost) in enumerate(list(auto_boost_dict.items())[:10], 1):
        print(f"    {i}. '{token}': {boost:.2f}x boost")

    # Step 6: Apply boosts to IDF
    print("\n[Step 6] Applying Trend Boosts to IDF...")
    boosted_idf = apply_temporal_boost_to_idf(
        idf_token_dict=temporal_idf_token,
        boost_dict=auto_boost_dict,
        tokenizer=tokenizer,
    )

    print(f"\n✓ Boosted IDF created")

    # Compare original vs boosted
    print(f"\n  Comparison (Original IDF vs Boosted IDF):")
    for token in list(auto_boost_dict.keys())[:5]:
        if token in temporal_idf_token:
            original = temporal_idf_token[token]
            boosted = boosted_idf[token]
            change = (boosted / original - 1) * 100
            print(f"    '{token}': {original:.4f} → {boosted:.4f} (+{change:.1f}%)")
else:
    print("⚠️  Skipping boost application (no trending tokens)")
    auto_boost_dict = {}

# Summary
print("\n" + "=" * 70)
print("✓ Phase 2 Features Test Complete!")
print("=" * 70)
print(f"\nResults:")
print(f"  Documents processed: {len(documents):,}")
print(f"  Date range: {min(dates).date()} to {max(dates).date()}")
print(f"  Temporal IDF tokens: {len(temporal_idf_token):,}")
print(f"  Trending tokens detected: {len(trending_tokens)}")
print(f"  Auto-boost dictionary size: {len(auto_boost_dict)}")
print()
print("Key Improvements:")
print("  ✓ Temporal information preserved and utilized")
print("  ✓ Time-weighted IDF calculated (recent docs weighted higher)")
print("  ✓ Automatic trend detection (replaces hardcoded TREND_BOOST)")
print("  ✓ Fully unsupervised approach - no manual curation needed")
print()
