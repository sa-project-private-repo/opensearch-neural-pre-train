"""
Data loading utilities with temporal information preservation.

This module loads Korean news datasets while preserving temporal metadata
for time-weighted analysis and trend detection.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm


def load_korean_news_with_dates(
    dataset_name: str = "heegyu/news-category-dataset",
    split: str = "train",
    max_samples: Optional[int] = None,
    min_doc_length: int = 10,
) -> Dict[str, List]:
    """
    Load Korean news dataset with date information preserved.

    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to load
        max_samples: Maximum number of samples (None for all)
        min_doc_length: Minimum document length in characters

    Returns:
        Dictionary containing:
            - documents: List of document texts
            - dates: List of datetime objects
            - categories: List of category labels
            - metadata: Additional metadata

    Example:
        >>> data = load_korean_news_with_dates(max_samples=10000)
        >>> print(f"Loaded {len(data['documents'])} documents")
        >>> print(f"Date range: {min(data['dates'])} to {max(data['dates'])}")
    """
    print(f"📰 Loading Korean news dataset: {dataset_name}")

    # Load dataset
    if max_samples:
        dataset = load_dataset(dataset_name, split=f"{split}[:{max_samples}]")
    else:
        dataset = load_dataset(dataset_name, split=split)

    documents = []
    dates = []
    categories = []

    print(f"Processing {len(dataset)} articles...")

    for item in tqdm(dataset, desc="Loading news"):
        # Extract text
        if "headline" in item:
            text = item["headline"]
        elif "text" in item:
            text = item["text"]
        elif "content" in item:
            text = item["content"]
        else:
            continue

        # Filter by length
        if len(text.strip()) < min_doc_length:
            continue

        # Extract date
        date = None
        if "date" in item:
            date = _parse_date(item["date"])
        elif "published_date" in item:
            date = _parse_date(item["published_date"])
        elif "publish_date" in item:
            date = _parse_date(item["publish_date"])
        elif "datetime" in item:
            date = _parse_date(item["datetime"])

        # Use current date as fallback
        if date is None:
            date = datetime.now()

        # Extract category
        category = item.get("category", "unknown")

        documents.append(text.strip())
        dates.append(date)
        categories.append(category)

    print(f"✓ Loaded {len(documents)} documents")

    if dates:
        date_range = max(dates) - min(dates)
        print(f"  Date range: {min(dates).date()} to {max(dates).date()}")
        print(f"  Time span: {date_range.days} days")

    if categories:
        category_counts = defaultdict(int)
        for cat in categories:
            category_counts[cat] += 1
        print(f"  Categories: {len(category_counts)}")
        for cat, count in sorted(
            category_counts.items(), key=lambda x: -x[1]
        )[:5]:
            print(f"    {cat}: {count}")

    return {
        "documents": documents,
        "dates": dates,
        "categories": categories,
        "dataset_name": dataset_name,
        "loaded_at": datetime.now(),
    }


def _parse_date(date_str: any) -> Optional[datetime]:
    """
    Parse date from various formats.

    Args:
        date_str: Date string or datetime object

    Returns:
        datetime object or None if parsing fails
    """
    if isinstance(date_str, datetime):
        return date_str

    if not isinstance(date_str, str):
        return None

    # Try common date formats
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y년 %m월 %d일",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue

    return None


def load_multiple_korean_datasets(
    max_samples_per_dataset: int = 50000,
) -> Dict[str, List]:
    """
    Load multiple Korean NLP datasets with temporal information.

    Args:
        max_samples_per_dataset: Maximum samples per dataset

    Returns:
        Combined dataset dictionary
    """
    all_documents = []
    all_dates = []
    all_categories = []
    all_sources = []

    print("=" * 60)
    print("Loading Multiple Korean Datasets")
    print("=" * 60)

    # 1. Korean News
    try:
        news_data = load_korean_news_with_dates(
            max_samples=max_samples_per_dataset
        )
        all_documents.extend(news_data["documents"])
        all_dates.extend(news_data["dates"])
        all_categories.extend(
            [f"news_{cat}" for cat in news_data["categories"]]
        )
        all_sources.extend(["news"] * len(news_data["documents"]))
    except Exception as e:
        print(f"⚠️  Failed to load news dataset: {e}")

    # 2. KLUE MRC (Question-Answer pairs)
    try:
        print(f"\n📚 Loading KLUE MRC dataset...")
        klue_mrc = load_dataset("klue", "mrc", split=f"train[:{max_samples_per_dataset}]")

        for item in tqdm(klue_mrc, desc="Processing KLUE MRC"):
            if "context" in item and len(item["context"].strip()) > 10:
                all_documents.append(item["context"].strip())
                all_dates.append(datetime.now())  # No date info
                all_categories.append("klue_mrc")
                all_sources.append("klue_mrc")

        print(f"✓ Loaded {len([s for s in all_sources if s == 'klue_mrc'])} KLUE MRC documents")
    except Exception as e:
        print(f"⚠️  Failed to load KLUE MRC: {e}")

    # 3. KorQuAD (Korean Question Answering)
    try:
        print(f"\n📚 Loading KorQuAD v1.0 dataset...")
        korquad = load_dataset("squad_kor_v1", split=f"train[:{max_samples_per_dataset}]")

        for item in tqdm(korquad, desc="Processing KorQuAD"):
            if "context" in item and len(item["context"].strip()) > 10:
                all_documents.append(item["context"].strip())
                all_dates.append(datetime.now())
                all_categories.append("korquad")
                all_sources.append("korquad")

        print(f"✓ Loaded {len([s for s in all_sources if s == 'korquad'])} KorQuAD documents")
    except Exception as e:
        print(f"⚠️  Failed to load KorQuAD: {e}")

    # Remove duplicates
    print(f"\n🔄 Removing duplicates...")
    unique_docs = {}
    for doc, date, cat, src in zip(
        all_documents, all_dates, all_categories, all_sources
    ):
        if doc not in unique_docs:
            unique_docs[doc] = (date, cat, src)

    final_documents = list(unique_docs.keys())
    final_dates = [unique_docs[doc][0] for doc in final_documents]
    final_categories = [unique_docs[doc][1] for doc in final_documents]
    final_sources = [unique_docs[doc][2] for doc in final_documents]

    print(
        f"✓ Deduplicated: {len(all_documents)} → {len(final_documents)} documents"
    )

    print("\n" + "=" * 60)
    print(f"✓ Total documents loaded: {len(final_documents):,}")
    print("=" * 60)

    return {
        "documents": final_documents,
        "dates": final_dates,
        "categories": final_categories,
        "sources": final_sources,
        "loaded_at": datetime.now(),
    }


def create_time_windows(
    dates: List[datetime], window_days: int = 30
) -> Dict[str, List[int]]:
    """
    Create time windows for temporal analysis.

    Args:
        dates: List of datetime objects
        window_days: Window size in days

    Returns:
        Dictionary mapping window names to document indices

    Example:
        >>> windows = create_time_windows(dates, window_days=30)
        >>> print(windows.keys())  # ['2024-01', '2024-02', ...]
    """
    if not dates:
        return {}

    windows = defaultdict(list)
    window_delta = timedelta(days=window_days)

    for idx, date in enumerate(dates):
        # Create window label (YYYY-MM format)
        window_label = date.strftime("%Y-%m")
        windows[window_label].append(idx)

    return dict(windows)


def get_recent_documents(
    documents: List[str],
    dates: List[datetime],
    recent_days: int = 30,
) -> Tuple[List[str], List[datetime]]:
    """
    Filter documents from recent time period.

    Args:
        documents: List of documents
        dates: List of dates
        recent_days: Number of recent days to include

    Returns:
        Filtered documents and dates

    Example:
        >>> recent_docs, recent_dates = get_recent_documents(
        ...     documents, dates, recent_days=30
        ... )
    """
    cutoff_date = max(dates) - timedelta(days=recent_days)

    recent_docs = []
    recent_dates_filtered = []

    for doc, date in zip(documents, dates):
        if date >= cutoff_date:
            recent_docs.append(doc)
            recent_dates_filtered.append(date)

    return recent_docs, recent_dates_filtered
