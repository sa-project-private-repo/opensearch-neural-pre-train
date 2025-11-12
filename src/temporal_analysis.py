"""
Temporal analysis for neural sparse retrieval.

This module implements time-weighted IDF calculation and automatic trend
detection for Korean news documents.
"""

import math
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm


def calculate_temporal_idf(
    documents: List[str],
    dates: List[datetime],
    tokenizer,
    decay_factor: float = 0.95,
    base_date: Optional[datetime] = None,
) -> Tuple[Dict[str, float], Dict[int, float]]:
    """
    Calculate IDF with temporal weighting (recent documents weighted higher).

    Uses exponential decay: weight = decay_factor ^ days_old

    Args:
        documents: List of document texts
        dates: List of document dates
        tokenizer: Tokenizer for text processing
        decay_factor: Decay factor for temporal weighting (default: 0.95)
        base_date: Reference date (default: max date in dataset)

    Returns:
        idf_token_dict: {token_string: idf_score}
        idf_id_dict: {token_id: idf_score}

    Example:
        >>> idf_token, idf_id = calculate_temporal_idf(
        ...     documents, dates, tokenizer, decay_factor=0.95
        ... )
        >>> # Recent documents have higher influence on IDF
    """
    if base_date is None:
        base_date = max(dates)

    print(f"📊 Calculating Temporal IDF with decay_factor={decay_factor}")
    print(f"   Base date: {base_date.date()}")

    # Calculate document weights based on recency
    weights = []
    for date in dates:
        days_old = (base_date - date).days
        weight = decay_factor ** days_old
        weights.append(weight)

    total_weight = sum(weights)
    print(f"   Total weighted documents: {total_weight:.2f}")

    # Calculate weighted document frequency
    df = Counter()  # Document frequency (weighted)

    for doc, weight in tqdm(
        zip(documents, weights), total=len(documents), desc="Computing temporal IDF"
    ):
        tokens = tokenizer.encode(
            doc, add_special_tokens=False, max_length=128, truncation=True
        )
        unique_tokens = set(tokens)

        for token_id in unique_tokens:
            df[token_id] += weight  # Weighted count

    # Calculate IDF: log((N+1) / (df+1)) + 1
    idf_id_dict = {}
    for token_id, weighted_df in df.items():
        idf_score = math.log((total_weight + 1) / (weighted_df + 1)) + 1.0
        idf_id_dict[token_id] = idf_score

    # Convert to token strings
    idf_token_dict = {}
    for token_id, score in idf_id_dict.items():
        token_str = tokenizer.decode([token_id])
        idf_token_dict[token_str] = float(score)

    print(f"✓ Calculated IDF for {len(idf_token_dict):,} tokens")
    print(f"  Average IDF: {np.mean(list(idf_token_dict.values())):.4f}")

    return idf_token_dict, idf_id_dict


def calculate_windowed_idf(
    documents: List[str],
    dates: List[datetime],
    tokenizer,
    window_days: List[int] = [30, 90, 365],
) -> Dict[str, Dict[str, float]]:
    """
    Calculate IDF for multiple time windows.

    Args:
        documents: List of document texts
        dates: List of dates
        tokenizer: Tokenizer
        window_days: List of window sizes in days

    Returns:
        Dictionary mapping window names to IDF dictionaries

    Example:
        >>> windowed_idf = calculate_windowed_idf(
        ...     documents, dates, tokenizer, window_days=[30, 90, 365]
        ... )
        >>> recent_idf = windowed_idf['30_days']
    """
    base_date = max(dates)
    print(f"\n📅 Calculating Windowed IDF")
    print(f"   Base date: {base_date.date()}")

    windowed_idfs = {}

    for window_size in window_days:
        print(f"\n  Window: {window_size} days")
        cutoff_date = base_date - timedelta(days=window_size)

        # Filter documents in window
        window_docs = []
        for doc, date in zip(documents, dates):
            if date >= cutoff_date:
                window_docs.append(doc)

        if len(window_docs) == 0:
            print(f"    ⚠️  No documents in {window_size}-day window")
            continue

        print(f"    Documents in window: {len(window_docs)}")

        # Calculate IDF for this window
        N = len(window_docs)
        df = Counter()

        for doc in window_docs:
            tokens = tokenizer.encode(
                doc, add_special_tokens=False, max_length=128, truncation=True
            )
            unique_tokens = set(tokens)
            for token_id in unique_tokens:
                df[token_id] += 1

        # IDF calculation
        idf_dict = {}
        for token_id, doc_freq in df.items():
            idf_score = math.log((N + 1) / (doc_freq + 1)) + 1.0
            token_str = tokenizer.decode([token_id])
            idf_dict[token_str] = float(idf_score)

        window_name = f"{window_size}_days"
        windowed_idfs[window_name] = idf_dict
        print(f"    ✓ {len(idf_dict):,} tokens")

    return windowed_idfs


def detect_trending_tokens(
    documents: List[str],
    dates: List[datetime],
    tokenizer,
    recent_days: int = 30,
    historical_days: int = 365,
    min_recent_count: int = 5,
    top_k: int = 100,
) -> List[Tuple[str, float]]:
    """
    Automatically detect trending tokens based on frequency changes.

    Compares recent frequency vs historical frequency to find emerging terms.

    Args:
        documents: List of documents
        dates: List of dates
        tokenizer: Tokenizer
        recent_days: Recent period in days (default: 30)
        historical_days: Historical period in days (default: 365)
        min_recent_count: Minimum occurrences in recent period
        top_k: Number of top trending tokens to return

    Returns:
        List of (token, trend_score) tuples, sorted by trend score

    Example:
        >>> trending = detect_trending_tokens(
        ...     documents, dates, tokenizer, recent_days=30
        ... )
        >>> for token, score in trending[:10]:
        ...     print(f"{token}: {score:.2f}x increase")
    """
    base_date = max(dates)
    recent_cutoff = base_date - timedelta(days=recent_days)
    historical_cutoff = base_date - timedelta(days=historical_days)

    print(f"\n🔥 Detecting Trending Tokens")
    print(f"   Recent period: {recent_days} days")
    print(f"   Historical period: {historical_days} days")

    # Count tokens in recent and historical periods
    recent_counts = Counter()
    historical_counts = Counter()
    recent_doc_count = 0
    historical_doc_count = 0

    for doc, date in tqdm(
        zip(documents, dates), total=len(documents), desc="Analyzing trends"
    ):
        tokens = tokenizer.encode(
            doc, add_special_tokens=False, max_length=128, truncation=True
        )

        if date >= recent_cutoff:
            recent_doc_count += 1
            for token_id in tokens:
                recent_counts[token_id] += 1

        # Historical: between historical_cutoff and recent_cutoff (exclusive)
        if historical_cutoff <= date < recent_cutoff:
            historical_doc_count += 1
            for token_id in tokens:
                historical_counts[token_id] += 1

    print(f"   Recent documents: {recent_doc_count}")
    print(f"   Historical documents: {historical_doc_count}")

    # Calculate trend scores
    trend_scores = []

    for token_id, recent_count in recent_counts.items():
        if recent_count < min_recent_count:
            continue

        historical_count = historical_counts.get(token_id, 0)

        # Normalize by document counts
        recent_freq = recent_count / max(recent_doc_count, 1)
        historical_freq = historical_count / max(historical_doc_count, 1)

        # Trend score: ratio of recent to historical frequency
        # Add smoothing to avoid division by zero
        trend_score = (recent_freq + 1e-6) / (historical_freq + 1e-6)

        # Only include if trending up (score > 1.5)
        if trend_score > 1.5:
            token_str = tokenizer.decode([token_id])
            # Filter out subword tokens and special chars
            if not token_str.startswith("##") and len(token_str) > 1:
                trend_scores.append((token_str, trend_score, recent_count))

    # Sort by trend score
    trend_scores.sort(key=lambda x: x[1], reverse=True)

    # Return top-k with token and score
    top_trending = [(token, score) for token, score, _ in trend_scores[:top_k]]

    print(f"\n✓ Found {len(top_trending)} trending tokens")
    print(f"\n  Top 10 Trending:")
    for i, (token, score) in enumerate(top_trending[:10], 1):
        print(f"    {i}. {token}: {score:.2f}x")

    return top_trending


def build_trend_boost_dict(
    trending_tokens: List[Tuple[str, float]],
    max_boost: float = 2.0,
    min_boost: float = 1.2,
) -> Dict[str, float]:
    """
    Build boost dictionary from trending tokens.

    Args:
        trending_tokens: List of (token, trend_score) tuples
        max_boost: Maximum boost factor (default: 2.0)
        min_boost: Minimum boost factor (default: 1.2)

    Returns:
        Dictionary mapping tokens to boost factors

    Example:
        >>> trending = detect_trending_tokens(docs, dates, tokenizer)
        >>> boost_dict = build_trend_boost_dict(trending)
        >>> # Replace hardcoded TREND_BOOST with this
    """
    if not trending_tokens:
        return {}

    # Normalize trend scores to boost range
    max_score = max(score for _, score in trending_tokens)
    min_score = min(score for _, score in trending_tokens)
    score_range = max_score - min_score

    boost_dict = {}

    for token, score in trending_tokens:
        if score_range > 0:
            # Linear interpolation to [min_boost, max_boost]
            normalized = (score - min_score) / score_range
            boost_factor = min_boost + normalized * (max_boost - min_boost)
        else:
            boost_factor = max_boost

        boost_dict[token] = boost_factor

    print(f"\n✓ Created boost dictionary with {len(boost_dict)} tokens")
    print(f"  Boost range: {min_boost:.2f} - {max_boost:.2f}")

    return boost_dict


def apply_temporal_boost_to_idf(
    idf_token_dict: Dict[str, float],
    boost_dict: Dict[str, float],
    tokenizer,
) -> Dict[str, float]:
    """
    Apply trending token boosts to IDF scores.

    Args:
        idf_token_dict: Original IDF dictionary
        boost_dict: Boost factors for trending tokens
        tokenizer: Tokenizer

    Returns:
        Boosted IDF dictionary

    Example:
        >>> boosted_idf = apply_temporal_boost_to_idf(
        ...     idf_token_dict, boost_dict, tokenizer
        ... )
    """
    boosted_idf = idf_token_dict.copy()
    boost_count = 0

    for token, boost_factor in boost_dict.items():
        # Tokenize to handle subwords
        token_ids = tokenizer.encode(token, add_special_tokens=False)

        for token_id in token_ids:
            token_str = tokenizer.decode([token_id])
            if token_str in boosted_idf:
                boosted_idf[token_str] *= boost_factor
                boost_count += 1

    print(f"✓ Applied boost to {boost_count} tokens")

    return boosted_idf


def analyze_token_frequency_over_time(
    documents: List[str],
    dates: List[datetime],
    tokenizer,
    tokens_to_track: List[str],
    window_size: int = 30,
) -> Dict[str, List[Tuple[datetime, int]]]:
    """
    Analyze how specific token frequencies change over time.

    Args:
        documents: List of documents
        dates: List of dates
        tokenizer: Tokenizer
        tokens_to_track: List of tokens to analyze
        window_size: Rolling window size in days

    Returns:
        Dictionary mapping tokens to (date, count) time series

    Example:
        >>> freq_over_time = analyze_token_frequency_over_time(
        ...     documents, dates, tokenizer,
        ...     tokens_to_track=['AI', 'ChatGPT', 'LLM']
        ... )
    """
    # Create time series
    time_series = defaultdict(lambda: defaultdict(int))

    for doc, date in zip(documents, dates):
        tokens = tokenizer.encode(
            doc, add_special_tokens=False, max_length=128, truncation=True
        )
        token_strs = [tokenizer.decode([tid]) for tid in tokens]

        for track_token in tokens_to_track:
            if track_token in token_strs:
                window_start = date.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                time_series[track_token][window_start] += 1

    # Convert to sorted lists
    result = {}
    for token, date_counts in time_series.items():
        sorted_series = sorted(date_counts.items())
        result[token] = sorted_series

    return result
