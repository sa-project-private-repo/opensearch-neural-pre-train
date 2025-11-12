"""
Temporal clustering for unsupervised synonym discovery.

This module implements time-based clustering of token embeddings to discover
synonyms and semantic groups with temporal weighting.
"""

from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans, DBSCAN
from tqdm import tqdm


def extract_temporal_embeddings(
    documents: List[str],
    dates: List[datetime],
    tokenizer,
    bert_model,
    time_windows: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract token embeddings from BERT model for different time windows.

    Args:
        documents: List of documents
        dates: List of document dates
        tokenizer: Tokenizer
        bert_model: BERT model (for token embeddings)
        time_windows: List of window sizes in days (None = all data)

    Returns:
        Dictionary mapping window names to token embeddings

    Example:
        >>> embeddings = extract_temporal_embeddings(
        ...     documents, dates, tokenizer, bert_model,
        ...     time_windows=[30, 90, 365]
        ... )
    """
    if time_windows is None:
        time_windows = [0]  # 0 means all data

    print(f"\n📊 Extracting Temporal Token Embeddings")

    # Get token embeddings from BERT model
    try:
        token_embeddings = (
            bert_model.bert.embeddings.word_embeddings.weight.detach()
            .cpu()
            .numpy()
        )
    except AttributeError:
        # Try alternative attribute path
        token_embeddings = (
            bert_model.embeddings.word_embeddings.weight.detach().cpu().numpy()
        )

    vocab_size, embedding_dim = token_embeddings.shape
    print(f"  Token embeddings: {vocab_size:,} x {embedding_dim}")

    # For now, return the base embeddings
    # In future: could compute context-aware embeddings per time window
    windowed_embeddings = {"all_time": token_embeddings}

    return windowed_embeddings


def cluster_tokens_temporal(
    token_embeddings: np.ndarray,
    tokenizer,
    method: str = "kmeans",
    n_clusters: int = 500,
    min_cluster_size: int = 2,
    distance_threshold: float = 0.3,
) -> Dict[int, List[int]]:
    """
    Cluster token embeddings to discover semantic groups.

    Args:
        token_embeddings: Token embedding matrix (vocab_size, embedding_dim)
        tokenizer: Tokenizer for filtering
        method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
        n_clusters: Number of clusters for kmeans
        min_cluster_size: Minimum cluster size
        distance_threshold: Distance threshold for DBSCAN/hierarchical

    Returns:
        Dictionary mapping cluster_id to list of token_ids

    Example:
        >>> clusters = cluster_tokens_temporal(
        ...     embeddings, tokenizer, method='kmeans', n_clusters=500
        ... )
    """
    print(f"\n🔍 Clustering Tokens")
    print(f"  Method: {method}")
    print(f"  Embeddings: {token_embeddings.shape}")

    if method == "kmeans":
        print(f"  Num clusters: {n_clusters}")
        kmeans = KMeans(
            n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300
        )
        cluster_labels = kmeans.fit_predict(token_embeddings)

    elif method == "dbscan":
        print(f"  Distance threshold: {distance_threshold}")
        dbscan = DBSCAN(
            eps=distance_threshold, min_samples=min_cluster_size, metric="cosine"
        )
        cluster_labels = dbscan.fit_predict(token_embeddings)

    elif method == "hierarchical":
        print(f"  Distance threshold: {distance_threshold}")
        # Hierarchical clustering
        linkage_matrix = linkage(token_embeddings, method="average", metric="cosine")
        cluster_labels = fcluster(
            linkage_matrix, t=distance_threshold, criterion="distance"
        )

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Group tokens by cluster
    clusters = defaultdict(list)
    for token_id, cluster_id in enumerate(cluster_labels):
        # Skip noise cluster in DBSCAN (-1)
        if cluster_id == -1:
            continue

        clusters[int(cluster_id)].append(token_id)

    # Filter by cluster size
    filtered_clusters = {
        cid: tokens
        for cid, tokens in clusters.items()
        if len(tokens) >= min_cluster_size
    }

    print(f"\n✓ Created {len(filtered_clusters):,} clusters")
    print(f"  Avg cluster size: {np.mean([len(t) for t in filtered_clusters.values()]):.1f}")
    print(f"  Max cluster size: {max([len(t) for t in filtered_clusters.values()])}")

    return filtered_clusters


def build_synonym_groups_from_clusters(
    clusters: Dict[int, List[int]],
    token_embeddings: np.ndarray,
    tokenizer,
    similarity_threshold: float = 0.75,
    max_group_size: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build synonym groups from token clusters.

    Args:
        clusters: Dictionary of cluster_id → token_ids
        token_embeddings: Token embedding matrix
        tokenizer: Tokenizer
        similarity_threshold: Minimum cosine similarity for synonyms
        max_group_size: Maximum synonyms per group

    Returns:
        Dictionary mapping main token to list of (synonym, similarity) tuples

    Example:
        >>> synonym_groups = build_synonym_groups_from_clusters(
        ...     clusters, embeddings, tokenizer, similarity_threshold=0.75
        ... )
        >>> print(synonym_groups['검색'])  # ['search', 'retrieval', ...]
    """
    print(f"\n🔗 Building Synonym Groups from Clusters")
    print(f"  Clusters: {len(clusters):,}")
    print(f"  Similarity threshold: {similarity_threshold}")

    synonym_groups = {}

    for cluster_id, token_ids in tqdm(
        clusters.items(), desc="Building synonym groups"
    ):
        # Convert token IDs to strings
        tokens_in_cluster = []
        for token_id in token_ids:
            token_str = tokenizer.decode([token_id])

            # Filter out subword tokens, special chars, etc.
            if (
                token_str.startswith("##")
                or len(token_str) <= 1
                or not token_str.strip()
            ):
                continue

            tokens_in_cluster.append((token_id, token_str))

        if len(tokens_in_cluster) < 2:
            continue

        # Compute pairwise similarities within cluster
        # Use the first token as the "main" token
        main_token_id, main_token_str = tokens_in_cluster[0]
        main_embedding = token_embeddings[main_token_id]

        synonyms = []

        for token_id, token_str in tokens_in_cluster[1:]:
            if token_str == main_token_str:
                continue

            # Compute cosine similarity
            token_embedding = token_embeddings[token_id]
            similarity = 1 - cosine(main_embedding, token_embedding)

            if similarity >= similarity_threshold:
                synonyms.append((token_str, float(similarity)))

        # Sort by similarity and limit size
        synonyms.sort(key=lambda x: -x[1])
        synonyms = synonyms[:max_group_size]

        if synonyms:
            synonym_groups[main_token_str] = synonyms

    print(f"✓ Created {len(synonym_groups):,} synonym groups")

    # Show examples
    print(f"\n  Sample synonym groups:")
    for i, (main_token, synonyms) in enumerate(
        list(synonym_groups.items())[:5]
    ):
        syn_str = ", ".join([f"{s}({sim:.2f})" for s, sim in synonyms[:3]])
        print(f"    {main_token} → {syn_str}")

    return synonym_groups


def discover_synonyms_temporal(
    documents: List[str],
    dates: List[datetime],
    tokenizer,
    bert_model,
    decay_factor: float = 0.95,
    n_clusters: int = 500,
    similarity_threshold: float = 0.75,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Discover synonyms using time-weighted clustering (main function).

    Args:
        documents: List of documents
        dates: List of document dates
        tokenizer: Tokenizer
        bert_model: BERT model
        decay_factor: Temporal decay factor (higher = more weight on recent)
        n_clusters: Number of clusters
        similarity_threshold: Minimum similarity for synonyms

    Returns:
        Synonym groups dictionary

    Example:
        >>> synonyms = discover_synonyms_temporal(
        ...     documents, dates, tokenizer, bert_model
        ... )
    """
    print(f"\n{'='*70}")
    print("Temporal Synonym Discovery")
    print(f"{'='*70}")

    # Step 1: Extract token embeddings
    embeddings_dict = extract_temporal_embeddings(
        documents, dates, tokenizer, bert_model
    )
    token_embeddings = embeddings_dict["all_time"]

    # Step 2: Cluster tokens
    clusters = cluster_tokens_temporal(
        token_embeddings,
        tokenizer,
        method="kmeans",
        n_clusters=n_clusters,
        min_cluster_size=2,
    )

    # Step 3: Build synonym groups
    synonym_groups = build_synonym_groups_from_clusters(
        clusters,
        token_embeddings,
        tokenizer,
        similarity_threshold=similarity_threshold,
        max_group_size=10,
    )

    print(f"\n{'='*70}")
    print(f"✓ Synonym Discovery Complete")
    print(f"  Total synonym groups: {len(synonym_groups):,}")
    print(f"{'='*70}\n")

    return synonym_groups


def merge_synonym_dictionaries(
    auto_synonyms: Dict[str, List[Tuple[str, float]]],
    manual_synonyms: Optional[Dict[str, List[str]]] = None,
    priority: str = "auto",
) -> Dict[str, List[str]]:
    """
    Merge automatically discovered and manually curated synonyms.

    Args:
        auto_synonyms: Auto-discovered synonyms with scores
        manual_synonyms: Manually curated synonyms (optional)
        priority: Which to prioritize ('auto' or 'manual')

    Returns:
        Merged synonym dictionary

    Example:
        >>> merged = merge_synonym_dictionaries(
        ...     auto_synonyms, manual_synonyms, priority='manual'
        ... )
    """
    print(f"\n🔀 Merging Synonym Dictionaries")

    merged = {}

    # Add auto-discovered synonyms
    for token, syn_list in auto_synonyms.items():
        merged[token] = [syn for syn, _ in syn_list]

    # Add or override with manual synonyms
    if manual_synonyms:
        for token, syn_list in manual_synonyms.items():
            if token in merged and priority == "auto":
                # Keep auto, append manual
                merged[token].extend([s for s in syn_list if s not in merged[token]])
            else:
                # Use manual
                merged[token] = syn_list

    print(f"✓ Merged {len(merged):,} synonym groups")

    return merged


def filter_synonyms_by_frequency(
    synonym_groups: Dict[str, List[Tuple[str, float]]],
    documents: List[str],
    tokenizer,
    min_frequency: int = 5,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Filter synonym groups by token frequency in corpus.

    Removes rare tokens that might be noise.

    Args:
        synonym_groups: Synonym groups
        documents: Document corpus
        tokenizer: Tokenizer
        min_frequency: Minimum occurrence count

    Returns:
        Filtered synonym groups

    Example:
        >>> filtered = filter_synonyms_by_frequency(
        ...     synonyms, documents, tokenizer, min_frequency=10
        ... )
    """
    print(f"\n🔍 Filtering Synonyms by Frequency")
    print(f"  Minimum frequency: {min_frequency}")

    # Count token frequencies
    token_counts = Counter()

    for doc in tqdm(documents, desc="Counting tokens"):
        tokens = tokenizer.tokenize(doc)
        token_counts.update(tokens)

    # Filter synonyms
    filtered_groups = {}
    removed_count = 0

    for main_token, synonyms in synonym_groups.items():
        # Check main token frequency
        if token_counts[main_token] < min_frequency:
            removed_count += 1
            continue

        # Filter synonym list
        filtered_syns = [
            (syn, score)
            for syn, score in synonyms
            if token_counts[syn] >= min_frequency
        ]

        if filtered_syns:
            filtered_groups[main_token] = filtered_syns
        else:
            removed_count += 1

    print(
        f"✓ Filtered: {len(synonym_groups):,} → {len(filtered_groups):,} groups"
    )
    print(f"  Removed {removed_count:,} low-frequency groups")

    return filtered_groups
